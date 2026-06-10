"""Tests for the GPU job queue.

Evidence harness contract (from Kynormous council pressure):
- Two jobs must not overlap on the GPU route
- Crash produces failure receipt naming the phase and last trustworthy evidence
- Wrong backend/fallback makes mismatch visible in receipt
- Stale lock recovery is explicit, not silent
- Cancel doesn't produce partial output pretending to be completion
- Caller-specified output paths work, outputs don't collapse into a shared singleton
"""

import fcntl
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


@pytest.fixture
def queue_dir(tmp_path):
    return tmp_path / "test_queue"


@pytest.fixture
def queue(queue_dir):
    return GPUQueue(queue_dir)


@pytest.fixture
def echo_job_types():
    """Job types that use simple echo/sleep commands for testing."""
    return {
        "echo": ["echo", "processed {input_path}"],
        "slow": ["sleep", "2"],
        "failing": ["false"],  # always exits 1
        "write_output": ["sh", "-c", "echo result > {output_dir}/result.txt"],
    }


def make_request(job_type="echo", input_path="/tmp/test.png", output_dir=None, **params):
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    return JobRequest(
        job_type=job_type,
        input_path=input_path,
        output_dir=output_dir,
        params=params,
    )


# --- Submit and basic lifecycle ---

class TestSubmit:
    def test_submit_creates_job_directory(self, queue):
        req = make_request()
        job_dir = queue.submit(req)
        assert job_dir.exists()
        assert (job_dir / "request.json").exists()
        assert (job_dir / "status.json").exists()

    def test_submit_status_is_pending(self, queue):
        req = make_request()
        queue.submit(req)
        state = queue.get_job(req.job_id)
        assert state is not None
        assert state.status == JobStatus.PENDING

    def test_submit_preserves_caller_output_dir(self, queue, tmp_path):
        """Caller-specified output paths must not be overwritten."""
        out1 = str(tmp_path / "output_a")
        out2 = str(tmp_path / "output_b")
        req1 = make_request(output_dir=out1)
        req2 = make_request(output_dir=out2)
        queue.submit(req1)
        queue.submit(req2)
        s1 = queue.get_job(req1.job_id)
        s2 = queue.get_job(req2.job_id)
        assert s1.output_dir == out1
        assert s2.output_dir == out2
        assert s1.output_dir != s2.output_dir


# --- Execution ---

class TestExecution:
    def test_run_one_succeeds(self, queue, echo_job_types):
        req = make_request()
        queue.submit(req)
        ran = queue.run_one(echo_job_types)
        assert ran is True
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.DONE
        assert state.exit_code == 0
        assert state.started_at is not None
        assert state.finished_at is not None

    def test_run_one_empty_queue(self, queue, echo_job_types):
        ran = queue.run_one(echo_job_types)
        assert ran is False

    def test_run_creates_receipt(self, queue, echo_job_types):
        req = make_request()
        queue.submit(req)
        queue.run_one(echo_job_types)
        receipt_path = queue.queue_dir / "done" / req.job_id / "receipt.json"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "done"
        assert receipt["effective_route"] is not None

    def test_run_captures_stdout(self, queue, echo_job_types):
        req = make_request(input_path="/tmp/myimage.png")
        queue.submit(req)
        queue.run_one(echo_job_types)
        stdout_path = queue.queue_dir / "done" / req.job_id / "stdout.log"
        assert stdout_path.exists()
        assert "processed /tmp/myimage.png" in stdout_path.read_text()

    def test_run_writes_to_caller_output_dir(self, queue, tmp_path):
        """Outputs land in the caller-specified directory, not a singleton."""
        out_dir = str(tmp_path / "my_specific_output")
        req = make_request(job_type="write_output", output_dir=out_dir)
        queue.submit(req)
        job_types = {"write_output": ["sh", "-c", "echo result > {output_dir}/result.txt"]}
        queue.run_one(job_types)
        assert (Path(out_dir) / "result.txt").exists()
        assert (Path(out_dir) / "result.txt").read_text().strip() == "result"

    def test_outputs_dont_collapse_to_singleton(self, queue, tmp_path):
        """Two jobs with different output dirs produce separate outputs."""
        out1 = str(tmp_path / "out_a")
        out2 = str(tmp_path / "out_b")
        job_types = {"write_output": ["sh", "-c", "echo {input_path} > {output_dir}/result.txt"]}

        req1 = make_request(job_type="write_output", input_path="image_a.png", output_dir=out1)
        req2 = make_request(job_type="write_output", input_path="image_b.png", output_dir=out2)
        queue.submit(req1)
        queue.submit(req2)
        queue.run_one(job_types)
        queue.run_one(job_types)

        assert (Path(out1) / "result.txt").read_text().strip() == "image_a.png"
        assert (Path(out2) / "result.txt").read_text().strip() == "image_b.png"


# --- Failure receipts ---

class TestFailure:
    def test_failed_job_has_receipt(self, queue, echo_job_types):
        """Crash produces failure receipt naming the phase."""
        req = make_request(job_type="failing")
        queue.submit(req)
        queue.run_one(echo_job_types)
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.FAILED
        assert state.exit_code != 0
        assert state.failure_phase == "execution"

        receipt_path = queue.queue_dir / "failed" / req.job_id / "receipt.json"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "failed"
        assert receipt["failure_phase"] == "execution"

    def test_unknown_job_type_fails_at_dispatch(self, queue):
        """Unknown job type fails with dispatch phase, not silently."""
        req = make_request(job_type="nonexistent_model")
        queue.submit(req)
        queue.run_one({})
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.FAILED
        assert state.failure_phase == "dispatch"
        assert "nonexistent_model" in state.error_message

    def test_failed_job_captures_stderr(self, queue):
        """Failure captures stderr for debugging."""
        job_types = {"stderr_test": ["sh", "-c", "echo 'error details' >&2; exit 1"]}
        req = make_request(job_type="stderr_test")
        queue.submit(req)
        queue.run_one(job_types)
        stderr_path = queue.queue_dir / "failed" / req.job_id / "stderr.log"
        assert stderr_path.exists()
        assert "error details" in stderr_path.read_text()

    def test_failed_receipt_records_effective_route(self, queue, echo_job_types):
        """Even on failure, the receipt records what command was attempted."""
        req = make_request(job_type="failing")
        queue.submit(req)
        queue.run_one(echo_job_types)
        receipt_path = queue.queue_dir / "failed" / req.job_id / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["effective_route"] is not None


# --- Serialization (two jobs must not overlap) ---

class TestSerialization:
    def test_second_job_blocked_while_first_runs(self, queue):
        """Two jobs must not overlap on the GPU route.

        We prove this by acquiring the flock in this test and verifying
        that run_one returns False (cannot acquire lock).
        """
        req = make_request()
        queue.submit(req)

        # Simulate a running job by holding the lock
        lock_fd = open(queue.lock_path, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        try:
            ran = queue.run_one({"echo": ["echo", "hello"]})
            assert ran is False  # Could not acquire lock
            # Job should still be pending
            state = queue.get_job(req.job_id)
            assert state.status == JobStatus.PENDING
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def test_fifo_order(self, queue, tmp_path):
        """Jobs execute in submission order (FIFO)."""
        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")
        req1 = make_request(job_type="write_output", input_path="first", output_dir=out1)
        time.sleep(0.01)  # ensure different submitted_at
        req2 = make_request(job_type="write_output", input_path="second", output_dir=out2)

        queue.submit(req1)
        queue.submit(req2)

        job_types = {"write_output": ["sh", "-c", "echo {input_path} > {output_dir}/result.txt"]}

        queue.run_one(job_types)
        # First job should be done
        assert queue.get_job(req1.job_id).status == JobStatus.DONE
        assert queue.get_job(req2.job_id).status == JobStatus.PENDING

        queue.run_one(job_types)
        assert queue.get_job(req2.job_id).status == JobStatus.DONE


# --- Cancel ---

class TestCancel:
    def test_cancel_pending_job(self, queue):
        req = make_request()
        queue.submit(req)
        result = queue.cancel(req.job_id)
        assert result is True
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.CANCELLED

    def test_cancel_nonexistent_job(self, queue):
        result = queue.cancel("nonexistent_id")
        assert result is False

    def test_cancelled_job_not_executed(self, queue, echo_job_types):
        """Cancel must not produce partial output pretending to be completion."""
        req = make_request()
        queue.submit(req)
        queue.cancel(req.job_id)

        ran = queue.run_one(echo_job_types)
        assert ran is False  # Queue is empty (cancelled job removed from pending)

        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.CANCELLED
        # No receipt in done/
        assert not (queue.queue_dir / "done" / req.job_id).exists()

    def test_cancelled_job_has_no_output(self, queue, tmp_path):
        """Cancelled job must not write to the output directory."""
        out_dir = str(tmp_path / "should_be_empty")
        os.makedirs(out_dir, exist_ok=True)
        req = make_request(job_type="write_output", output_dir=out_dir)
        queue.submit(req)
        queue.cancel(req.job_id)
        # No output files should exist
        assert len(list(Path(out_dir).iterdir())) == 0


# --- Stale lock recovery ---

class TestStaleRecovery:
    def test_recover_stale_running_job(self, queue):
        """Stale lock recovery is explicit, not silent."""
        req = make_request()
        queue.submit(req)

        # Manually move to running and set a dead PID
        job_dir = queue.queue_dir / "pending" / req.job_id
        running_dir = queue.queue_dir / "running" / req.job_id
        import shutil
        shutil.move(str(job_dir), str(running_dir))

        state = JobState.from_json((running_dir / "status.json").read_text())
        state.status = JobStatus.RUNNING
        state.started_at = time.time() - 100
        state.pid = 99999999  # PID that almost certainly doesn't exist
        (running_dir / "status.json").write_text(state.to_json())

        recovered = queue.recover_stale()
        assert req.job_id in recovered

        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.FAILED
        assert state.failure_phase == "stale_recovery"
        assert "99999999" in state.error_message

        # Must have a receipt
        receipt_path = queue.queue_dir / "failed" / req.job_id / "receipt.json"
        assert receipt_path.exists()
        receipt = json.loads(receipt_path.read_text())
        assert receipt["failure_phase"] == "stale_recovery"

    def test_no_false_stale_on_live_process(self, queue):
        """Don't recover our own process as stale."""
        req = make_request()
        queue.submit(req)

        job_dir = queue.queue_dir / "pending" / req.job_id
        running_dir = queue.queue_dir / "running" / req.job_id
        import shutil
        shutil.move(str(job_dir), str(running_dir))

        state = JobState.from_json((running_dir / "status.json").read_text())
        state.status = JobStatus.RUNNING
        state.started_at = time.time()
        state.pid = os.getpid()  # This process is alive
        (running_dir / "status.json").write_text(state.to_json())

        recovered = queue.recover_stale()
        assert len(recovered) == 0


# --- List ---

class TestList:
    def test_list_all(self, queue, echo_job_types):
        req1 = make_request()
        req2 = make_request()
        queue.submit(req1)
        queue.submit(req2)
        queue.run_one(echo_job_types)

        all_jobs = queue.list_jobs()
        assert len(all_jobs) == 2

    def test_list_by_status(self, queue, echo_job_types):
        req1 = make_request()
        req2 = make_request()
        queue.submit(req1)
        queue.submit(req2)
        queue.run_one(echo_job_types)

        done = queue.list_jobs(JobStatus.DONE)
        pending = queue.list_jobs(JobStatus.PENDING)
        assert len(done) == 1
        assert len(pending) == 1


# --- Effective route visibility ---

class TestEffectiveRoute:
    def test_effective_route_recorded(self, queue, echo_job_types):
        req = make_request(input_path="/tmp/test.png")
        queue.submit(req)
        queue.run_one(echo_job_types)
        state = queue.get_job(req.job_id)
        assert state.effective_route is not None
        assert "/tmp/test.png" in state.effective_route

    def test_param_cannot_override_input_path(self, queue):
        """User params must not shadow reserved keys (M1 regression)."""
        req = make_request(
            job_type="echo",
            input_path="/real/image.png",
            input_path_override="INJECTED",  # sneaky param
        )
        # Manually set the param to try to override
        req.params["input_path"] = "INJECTED"
        queue.submit(req)
        job_types = {"echo": ["echo", "{input_path}"]}
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        assert "INJECTED" not in state.effective_route
        assert "/real/image.png" in state.effective_route

    def test_param_cannot_override_output_dir(self, queue, tmp_path):
        """User params must not shadow output_dir."""
        real_out = str(tmp_path / "real_out")
        req = make_request(job_type="echo", output_dir=real_out)
        req.params["output_dir"] = "/tmp/evil"
        queue.submit(req)
        job_types = {"echo": ["echo", "{output_dir}"]}
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        assert "/tmp/evil" not in state.effective_route
        assert real_out in state.effective_route

    def test_effective_route_distinct_from_request(self, queue):
        """Requested route vs effective route are separate fields."""
        req = make_request(job_type="echo", input_path="/my/image.png")
        queue.submit(req)
        job_types = {"echo": ["echo", "processed {input_path}"]}
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        # request records the type; effective_route records the actual command
        assert state.job_type == "echo"
        assert "echo" in state.effective_route
        assert "/my/image.png" in state.effective_route
