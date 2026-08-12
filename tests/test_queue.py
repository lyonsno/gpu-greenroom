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
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from gpu_queue.models import CompletionOutboxRequest, JobRequest, JobState, JobStatus
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

    def test_list_tolerates_additive_status_fields(self, queue):
        """Newer workers may add status fields; older list readers must not crash."""
        req = make_request()
        queue.submit(req)
        status_file = queue.queue_dir / "pending" / req.job_id / "status.json"
        status = json.loads(status_file.read_text())
        status["worker_schema"] = "future-greenroom.v2"
        status["warnings"] = ["volatile_output"]
        status_file.write_text(json.dumps(status))

        [state] = queue.list_jobs(JobStatus.PENDING)

        assert state.job_id == req.job_id
        assert state.warnings == ["volatile_output"]


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

    def test_param_value_braces_not_expanded(self, queue, tmp_path):
        """Param values containing {braces} must not be format-expanded."""
        out = str(tmp_path / "out")
        req = make_request(job_type="echo", output_dir=out)
        req.params["seed"] = "{input_path}"  # sneaky value
        queue.submit(req)
        job_types = {"echo": {"cmd": ["sh", "-c", "echo {seed} > {output_dir}/val.txt"]}}
        queue.run_one(job_types)
        result = (Path(out) / "val.txt").read_text().strip()
        # Should be the literal string {input_path}, not the expanded path
        assert result == "{input_path}"

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


# --- Rich job type config ---

class TestRichJobTypeConfig:
    def test_dict_config_with_cmd(self, queue, tmp_path):
        """Job types can be dicts with cmd, cwd, env, defaults."""
        out = str(tmp_path / "out")
        req = make_request(job_type="rich", output_dir=out)
        queue.submit(req)
        job_types = {
            "rich": {
                "cmd": ["echo", "hello from {input_path}"],
            },
        }
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.DONE

    def test_cwd_is_respected(self, queue, tmp_path):
        """cwd field sets the working directory for the subprocess."""
        out = str(tmp_path / "out")
        work_dir = str(tmp_path / "workdir")
        os.makedirs(work_dir)
        req = make_request(job_type="cwd_test", output_dir=out)
        queue.submit(req)
        job_types = {
            "cwd_test": {
                "cmd": ["sh", "-c", "pwd > {output_dir}/cwd.txt"],
                "cwd": work_dir,
            },
        }
        queue.run_one(job_types)
        assert (Path(out) / "cwd.txt").read_text().strip() == work_dir

    def test_env_is_passed(self, queue, tmp_path):
        """env dict merges into subprocess environment."""
        out = str(tmp_path / "out")
        req = make_request(job_type="env_test", output_dir=out)
        queue.submit(req)
        job_types = {
            "env_test": {
                "cmd": ["sh", "-c", "echo $MY_TEST_VAR > {output_dir}/env.txt"],
                "env": {"MY_TEST_VAR": "greenroom_works"},
            },
        }
        queue.run_one(job_types)
        assert (Path(out) / "env.txt").read_text().strip() == "greenroom_works"

    def test_defaults_fill_missing_params(self, queue, tmp_path):
        """defaults provide fallback values for unspecified params."""
        out = str(tmp_path / "out")
        req = make_request(job_type="defaults_test", output_dir=out)
        # No seed param specified
        queue.submit(req)
        job_types = {
            "defaults_test": {
                "cmd": ["sh", "-c", "echo {seed} > {output_dir}/seed.txt"],
                "defaults": {"seed": "42"},
            },
        }
        queue.run_one(job_types)
        assert (Path(out) / "seed.txt").read_text().strip() == "42"

    def test_user_params_override_defaults(self, queue, tmp_path):
        """User-supplied params take precedence over defaults."""
        out = str(tmp_path / "out")
        req = make_request(job_type="defaults_test", output_dir=out, seed="99")
        queue.submit(req)
        job_types = {
            "defaults_test": {
                "cmd": ["sh", "-c", "echo {seed} > {output_dir}/seed.txt"],
                "defaults": {"seed": "42"},
            },
        }
        queue.run_one(job_types)
        assert (Path(out) / "seed.txt").read_text().strip() == "99"

    def test_cwd_with_env_and_defaults(self, queue, tmp_path):
        """All config fields work together."""
        out = str(tmp_path / "out")
        work_dir = str(tmp_path / "workdir")
        os.makedirs(work_dir)
        req = make_request(job_type="full", output_dir=out)
        queue.submit(req)
        job_types = {
            "full": {
                "cmd": ["sh", "-c", "echo $REPO:{seed}:$(pwd) > {output_dir}/combo.txt"],
                "cwd": work_dir,
                "env": {"REPO": "/dev/trellis2mlx"},
                "defaults": {"seed": "7"},
            },
        }
        queue.run_one(job_types)
        result = (Path(out) / "combo.txt").read_text().strip()
        assert result == f"/dev/trellis2mlx:7:{work_dir}"

    def test_bare_list_still_works(self, queue, echo_job_types):
        """Backwards compat: bare list job types still work."""
        req = make_request()
        queue.submit(req)
        queue.run_one(echo_job_types)
        assert queue.get_job(req.job_id).status == JobStatus.DONE


# --- Receipt route identity ---

class TestReceiptRouteIdentity:
    def test_receipt_records_effective_cwd(self, queue, tmp_path):
        out = str(tmp_path / "out")
        work_dir = str(tmp_path / "workdir")
        os.makedirs(work_dir)
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "hi"], "cwd": work_dir}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["effective_cwd"] == work_dir

    def test_receipt_records_effective_env(self, queue, tmp_path):
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "hi"], "env": {"FOO": "bar"}}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["effective_env"] == {"FOO": "bar"}

    def test_receipt_records_effective_defaults(self, queue, tmp_path):
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "{seed}"], "defaults": {"seed": "42"}}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["effective_defaults"] == {"seed": "42"}

    def test_receipt_records_ignored_params(self, queue, tmp_path):
        """Params submitted but not in template appear in receipt."""
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out, extra_thing="surprise")
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "{input_path}"]}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["ignored_params"] is not None
        assert "extra_thing" in receipt["ignored_params"]

    def test_receipt_no_ignored_when_all_consumed(self, queue, tmp_path):
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out, seed="7")
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "{seed}"]}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["ignored_params"] is None


# --- Configurable timeout ---

class TestConfigurableTimeout:
    def test_no_timeout_by_default(self, queue, tmp_path):
        """Rich config with no timeout field runs without time limit."""
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "ok"]}}
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.DONE
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["effective_timeout"] is None

    def test_explicit_timeout_recorded_in_receipt(self, queue, tmp_path):
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        job_types = {"t": {"cmd": ["echo", "ok"], "timeout": 3600}}
        queue.run_one(job_types)
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert receipt["effective_timeout"] == 3600

    def test_timeout_triggers_failure(self, queue, tmp_path):
        out = str(tmp_path / "out")
        req = make_request(job_type="t", output_dir=out)
        queue.submit(req)
        # 0.1s timeout on a 10s sleep
        job_types = {"t": {"cmd": ["sleep", "10"], "timeout": 0.1}}
        queue.run_one(job_types)
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.FAILED
        assert state.failure_phase == "timeout"

    def test_bare_list_has_no_timeout(self, queue):
        """Bare list job types run without timeout (no artificial limit)."""
        req = make_request()
        queue.submit(req)
        queue.run_one({"echo": ["echo", "hi"]})
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.DONE


# --- Pause/resume ---

class TestPauseResume:
    def test_pause_creates_marker(self, queue):
        queue.pause()
        assert queue.is_paused()
        assert queue.pause_path.exists()

    def test_resume_removes_marker(self, queue):
        queue.pause()
        queue.resume()
        assert not queue.is_paused()
        assert not queue.pause_path.exists()

    def test_resume_when_not_paused_is_noop(self, queue):
        queue.resume()  # should not raise
        assert not queue.is_paused()

    def test_pause_is_idempotent(self, queue):
        queue.pause()
        queue.pause()
        assert queue.is_paused()

    def test_run_one_skips_when_paused(self, queue, echo_job_types):
        """Paused worker must not pick up new jobs."""
        req = make_request()
        queue.submit(req)
        queue.pause()
        ran = queue.run_one(echo_job_types)
        assert ran is False
        state = queue.get_job(req.job_id)
        assert state.status == JobStatus.PENDING  # still pending, not consumed

    def test_run_one_resumes_after_unpause(self, queue, echo_job_types):
        req = make_request()
        queue.submit(req)
        queue.pause()
        assert queue.run_one(echo_job_types) is False
        queue.resume()
        assert queue.run_one(echo_job_types) is True
        assert queue.get_job(req.job_id).status == JobStatus.DONE

    def test_pause_does_not_affect_cancel(self, queue):
        """Cancel still works while paused."""
        req = make_request()
        queue.submit(req)
        queue.pause()
        assert queue.cancel(req.job_id) is True
        assert queue.get_job(req.job_id).status == JobStatus.CANCELLED

    def test_pause_does_not_affect_submit(self, queue):
        """Submit still works while paused."""
        queue.pause()
        req = make_request()
        job_dir = queue.submit(req)
        assert job_dir.exists()
        assert queue.get_job(req.job_id).status == JobStatus.PENDING

    def test_pause_after_lock_still_skips(self, queue, echo_job_types):
        """Pause between lock acquisition and job pickup still skips (post-flock recheck)."""
        req = make_request()
        queue.submit(req)
        # Pause after submit — run_one should see it even after acquiring the lock
        queue.pause()
        ran = queue.run_one(echo_job_types)
        assert ran is False
        assert queue.get_job(req.job_id).status == JobStatus.PENDING


# --- Durable output directory ---

class TestDurableOutputDir:
    def test_outputs_dir_created(self, queue):
        """Queue creates a durable outputs/ directory."""
        assert (queue.queue_dir / "outputs").is_dir()

    def test_default_output_dir_is_durable(self, queue):
        """When output_dir is omitted, submit auto-generates a durable path."""
        req = JobRequest(job_type="echo", input_path="/tmp/test.png")
        queue.submit(req)
        state = queue.get_job(req.job_id)
        assert state.output_dir.startswith(str(queue.queue_dir / "outputs"))
        assert req.job_id in state.output_dir
        assert Path(state.output_dir).parent == queue.queue_dir / "outputs"

    def test_explicit_output_dir_preserved(self, queue, tmp_path):
        """Explicit output_dir is not overwritten."""
        explicit = str(tmp_path / "my_output")
        req = JobRequest(job_type="echo", input_path="/tmp/test.png", output_dir=explicit)
        queue.submit(req)
        state = queue.get_job(req.job_id)
        assert state.output_dir == explicit

    def test_volatile_output_dir_warns_in_status(self, queue):
        """Output dir under /tmp or /private/tmp records a volatile_output warning."""
        req = JobRequest(job_type="echo", input_path="/tmp/test.png", output_dir="/tmp/ephemeral")
        queue.submit(req)
        status_file = queue.queue_dir / "pending" / req.job_id / "status.json"
        status = json.loads(status_file.read_text())
        assert "volatile_output" in (status.get("warnings") or [])

    def test_volatile_warning_in_receipt(self, queue):
        """Volatile warning propagates to the receipt after execution."""
        req = JobRequest(job_type="echo", input_path="/tmp/test.png", output_dir="/tmp/will-die")
        queue.submit(req)
        queue.run_one({"echo": ["echo", "hi"]})
        receipt = json.loads((queue.queue_dir / "done" / req.job_id / "receipt.json").read_text())
        assert "volatile_output" in (receipt.get("warnings") or [])

    def test_durable_output_dir_no_warning(self, queue, tmp_path):
        """Non-volatile paths produce no volatile_output warning."""
        safe = str(tmp_path / "safe_output")
        req = JobRequest(job_type="echo", input_path="/tmp/test.png", output_dir=safe)
        queue.submit(req)
        status_file = queue.queue_dir / "pending" / req.job_id / "status.json"
        status = json.loads(status_file.read_text())
        assert "volatile_output" not in (status.get("warnings") or [])

    def test_default_output_survives_in_queue_dir(self, queue):
        """Default output dir lives inside the queue dir, which is durable."""
        req = JobRequest(job_type="echo", input_path="/tmp/test.png")
        queue.submit(req)
        queue.run_one({"echo": {"cmd": ["sh", "-c", "echo data > {output_dir}/result.txt"]}})
        state = queue.get_job(req.job_id)
        assert (Path(state.output_dir) / "result.txt").exists()
        assert (Path(state.output_dir) / "result.txt").read_text().strip() == "data"


class TestMetadataSidecar:
    def test_metadata_written_on_success(self, queue, tmp_path):
        """Successful job writes metadata.json into output_dir."""
        out_dir = str(tmp_path / "output")
        req = make_request(
            job_type="write_output",
            input_path="/home/user/images/dragon.png",
            output_dir=out_dir,
            seed="42",
            resolution="512",
        )
        queue.submit(req)
        queue.run_one({"write_output": ["sh", "-c", "echo result > {output_dir}/result.txt"]})
        meta_path = Path(out_dir) / "metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["name"] == "dragon"
        assert meta["job_type"] == "write_output"
        assert meta["job_id"] == req.job_id
        assert meta["input_name"] == "dragon.png"
        assert meta["params"]["seed"] == "42"
        assert meta["params"]["resolution"] == "512"
        assert "result.txt" in meta["output_files"]
        assert "metadata.json" not in meta["output_files"]
        assert meta["created_at"] is not None
        assert meta["duration_s"] is not None

    def test_metadata_not_written_on_failure(self, queue, tmp_path):
        """Failed job does not write metadata.json."""
        out_dir = str(tmp_path / "output")
        os.makedirs(out_dir, exist_ok=True)
        req = make_request(job_type="failing", output_dir=out_dir)
        queue.submit(req)
        queue.run_one({"failing": ["false"]})
        assert not (Path(out_dir) / "metadata.json").exists()

    def test_metadata_uses_name_param(self, queue, tmp_path):
        """User-provided name param overrides input filename."""
        out_dir = str(tmp_path / "output")
        req = make_request(
            job_type="write_output",
            input_path="/tmp/IMG_0042.png",
            output_dir=out_dir,
            name="golden-goblet",
        )
        queue.submit(req)
        queue.run_one({"write_output": ["sh", "-c", "echo ok > {output_dir}/out.glb"]})
        meta = json.loads((Path(out_dir) / "metadata.json").read_text())
        assert meta["name"] == "golden-goblet"

    def test_metadata_includes_duration(self, queue, tmp_path):
        """Duration is computed from started_at and finished_at."""
        out_dir = str(tmp_path / "output")
        req = make_request(job_type="write_output", output_dir=out_dir)
        queue.submit(req)
        queue.run_one({"write_output": ["sh", "-c", "echo x > {output_dir}/x.txt"]})
        meta = json.loads((Path(out_dir) / "metadata.json").read_text())
        assert isinstance(meta["duration_s"], (int, float))
        assert meta["duration_s"] >= 0


# --- Terminal completion outbox ---

class TestTerminalCompletionOutbox:
    def _opted_request(self, **kwargs):
        request = make_request(**kwargs)
        request.completion_outbox = CompletionOutboxRequest(
            target_consumer="asset-consumer",
            target_consumer_id="consumer-asset",
            delivery_mode="checkpoint",
        )
        return request

    def _single_event(self, queue):
        event_paths = list(queue.completion_outbox_dir.glob("*.json"))
        assert len(event_paths) == 1
        return event_paths[0], json.loads(event_paths[0].read_text())

    def _assert_terminal_event(self, queue, request, terminal_status, failure_phase=None):
        terminal_dir = queue.queue_dir / terminal_status / request.job_id
        receipt_path = terminal_dir / "receipt.json"
        receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        event_path, event = self._single_event(queue)

        assert event_path.name == f"{request.job_id}-{receipt_digest}.json"
        assert event["event_kind"] == "gpu_greenroom.job_terminal"
        assert event["event_id"] == f"{request.job_id}:{receipt_digest}"
        assert event["job_id"] == request.job_id
        assert event["terminal_receipt_sha256"] == receipt_digest
        assert event["terminal_status"] == terminal_status
        assert event["failure_phase"] == failure_phase
        assert event["target_consumer"] == "asset-consumer"
        assert event["target_consumer_id"] == "consumer-asset"
        assert event["delivery_mode"] == "checkpoint"
        assert event["request_locator"] == str(terminal_dir / "request.json")
        assert event["receipt_locator"] == str(receipt_path)
        assert event["artifact_locator"] == request.output_dir
        return event_path, event

    def test_request_contract_tolerates_additive_fields(self):
        request = self._opted_request()
        payload = json.loads(request.to_json())
        payload["completion_outbox"]["consumer_schema"] = "future.v2"

        restored = JobRequest.from_json(json.dumps(payload))

        assert restored.completion_outbox == request.completion_outbox

    def test_success_emits_completion_with_terminal_locators(self, queue, echo_job_types):
        request = self._opted_request(input_path="/tmp/completion.png")
        queue.submit(request)

        assert queue.run_one(echo_job_types) is True

        _, event = self._assert_terminal_event(queue, request, "done")
        terminal_dir = queue.queue_dir / "done" / request.job_id
        assert event["log_locators"] == {
            "stdout": str(terminal_dir / "stdout.log"),
            "stderr": str(terminal_dir / "stderr.log"),
        }

    def test_execution_failure_emits_completion(self, queue, echo_job_types):
        request = self._opted_request(job_type="failing")
        queue.submit(request)

        queue.run_one(echo_job_types)

        self._assert_terminal_event(queue, request, "failed", "execution")

    def test_dispatch_failure_emits_completion(self, queue):
        request = self._opted_request(job_type="missing")
        queue.submit(request)

        queue.run_one({})

        _, event = self._assert_terminal_event(queue, request, "failed", "dispatch")
        assert event["log_locators"] == {"stdout": None, "stderr": None}

    def test_launch_failure_emits_completion(self, queue, monkeypatch):
        request = self._opted_request(job_type="broken-launch")
        queue.submit(request)

        def fail_launch(*args, **kwargs):
            raise OSError("deterministic launch failure")

        monkeypatch.setattr("gpu_queue.queue.subprocess.run", fail_launch)
        queue.run_one({"broken-launch": ["does-not-matter"]})

        self._assert_terminal_event(queue, request, "failed", "launch")

    def test_timeout_emits_completion(self, queue, monkeypatch):
        request = self._opted_request(job_type="timed-out")
        queue.submit(request)

        def time_out(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

        monkeypatch.setattr("gpu_queue.queue.subprocess.run", time_out)
        queue.run_one({"timed-out": {"cmd": ["does-not-matter"], "timeout": 1}})

        self._assert_terminal_event(queue, request, "failed", "timeout")

    def test_stale_recovery_emits_completion_once(self, queue):
        request = self._opted_request()
        job_dir = queue.submit(request)
        running_dir = queue.queue_dir / "running" / request.job_id
        job_dir.rename(running_dir)
        state = JobState.from_json((running_dir / "status.json").read_text())
        state.status = JobStatus.RUNNING
        state.started_at = time.time() - 100
        state.pid = 99999999
        (running_dir / "status.json").write_text(state.to_json())

        assert queue.recover_stale() == [request.job_id]
        assert queue.recover_stale() == []

        self._assert_terminal_event(queue, request, "failed", "stale_recovery")

    def test_pending_cancellation_emits_completion(self, queue):
        request = self._opted_request()
        queue.submit(request)

        assert queue.cancel(request.job_id) is True

        _, event = self._assert_terminal_event(queue, request, "cancelled")
        assert event["log_locators"] == {"stdout": None, "stderr": None}

    def test_duplicate_emit_keeps_one_identical_event(self, queue, echo_job_types):
        request = self._opted_request()
        queue.submit(request)
        queue.run_one(echo_job_types)
        terminal_dir = queue.queue_dir / "done" / request.job_id
        receipt_bytes = (terminal_dir / "receipt.json").read_bytes()
        event_path, _ = self._single_event(queue)
        original_bytes = event_path.read_bytes()
        original_mtime = event_path.stat().st_mtime_ns

        duplicate_path = queue._emit_terminal_completion_outbox(
            request,
            terminal_dir,
            receipt_bytes,
        )

        assert duplicate_path == event_path
        assert list(queue.completion_outbox_dir.glob("*.json")) == [event_path]
        assert event_path.read_bytes() == original_bytes
        assert event_path.stat().st_mtime_ns == original_mtime

    def test_duplicate_emit_fails_loud_for_conflicting_cached_event(self, queue, echo_job_types):
        request = self._opted_request()
        queue.submit(request)
        queue.run_one(echo_job_types)
        terminal_dir = queue.queue_dir / "done" / request.job_id
        receipt_bytes = (terminal_dir / "receipt.json").read_bytes()
        event_path, event = self._single_event(queue)
        event["target_consumer"] = "wrong-consumer"
        event_path.write_text(json.dumps(event, indent=2))

        with pytest.raises(RuntimeError, match="conflicting completion outbox event"):
            queue._emit_terminal_completion_outbox(request, terminal_dir, receipt_bytes)

    def test_terminal_move_crash_recovers_original_terminal_event(
        self, queue, echo_job_types, monkeypatch
    ):
        request = self._opted_request()
        queue.submit(request)
        original_move = queue._move_job

        def crash_before_terminal_move(job_dir, dest_status):
            if dest_status == "done":
                raise OSError("deterministic crash before terminal move")
            return original_move(job_dir, dest_status)

        monkeypatch.setattr(queue, "_move_job", crash_before_terminal_move)
        with pytest.raises(OSError, match="deterministic crash"):
            queue.run_one(echo_job_types)

        running_dir = queue.queue_dir / "running" / request.job_id
        state = JobState.from_json((running_dir / "status.json").read_text())
        state.pid = 99999999
        (running_dir / "status.json").write_text(state.to_json())
        monkeypatch.setattr(queue, "_move_job", original_move)

        assert queue.recover_stale() == [request.job_id]
        assert not running_dir.exists()
        assert (queue.queue_dir / "done" / request.job_id / "receipt.json").is_file()
        self._assert_terminal_event(queue, request, "done")

    def test_outbox_write_gap_keeps_terminal_receipt_and_reconciles(
        self, queue, echo_job_types, monkeypatch
    ):
        request = self._opted_request()
        queue.submit(request)
        original_emit = queue._emit_terminal_completion_outbox

        def fail_emit(*args, **kwargs):
            raise OSError("deterministic outbox write failure")

        monkeypatch.setattr(queue, "_emit_terminal_completion_outbox", fail_emit)

        assert queue.run_one(echo_job_types) is True
        terminal_dir = queue.queue_dir / "done" / request.job_id
        assert (terminal_dir / "receipt.json").is_file()
        assert (terminal_dir / "completion.json").is_file()
        assert list(queue.completion_outbox_dir.glob("*.json")) == []

        monkeypatch.setattr(queue, "_emit_terminal_completion_outbox", original_emit)
        reconciled = queue.reconcile_completion_outbox()

        assert len(reconciled) == 1
        self._assert_terminal_event(queue, request, "done")

    def test_outbox_visibility_happens_after_gpu_lock_release(
        self, queue, echo_job_types, monkeypatch
    ):
        request = self._opted_request()
        queue.submit(request)
        original_emit = queue._emit_terminal_completion_outbox
        observed_lock_released = []

        def emit_after_lock_release(*args, **kwargs):
            with open(queue.lock_path, "w") as probe_fd:
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                observed_lock_released.append(True)
                fcntl.flock(probe_fd, fcntl.LOCK_UN)
            return original_emit(*args, **kwargs)

        monkeypatch.setattr(
            queue,
            "_emit_terminal_completion_outbox",
            emit_after_lock_release,
        )

        assert queue.run_one(echo_job_types) is True
        assert observed_lock_released == [True]

    def test_zero_exit_failed_producer_report_cannot_emit_success(
        self, queue, echo_job_types, tmp_path
    ):
        producer_report = tmp_path / "producer-report.json"
        producer_report.write_text(json.dumps({
            "status": "failed",
            "failure_phase": "primary_validation",
            "effective_route": "wrong-route",
            "primary_output_validated": False,
        }))
        request = make_request()
        request.completion_outbox = CompletionOutboxRequest(
            target_consumer="asset-consumer",
            target_consumer_id="consumer-asset",
            delivery_mode="checkpoint",
            producer_report_locator=str(producer_report),
        )
        queue.submit(request)

        assert queue.run_one(echo_job_types) is True

        _, event = self._single_event(queue)
        assert event["observer_terminal_state"] == "done"
        assert event["observer_exit_code"] == 0
        assert event["terminal_class"] == "failed"
        assert event["claim_ceiling"] == "process_terminality_only"
        assert event["producer_report"]["locator"] == str(producer_report)
        assert event["producer_report"]["sha256"] == hashlib.sha256(
            producer_report.read_bytes()
        ).hexdigest()
        assert set(event["producer_report"]["failure_reasons"]) == {
            "producer_status_failed",
            "producer_failure_phase",
            "effective_route_mismatch",
            "primary_output_unvalidated",
        }

    def test_non_opted_jobs_do_not_emit_or_change_cancel_receipts(self, queue, echo_job_types):
        completed = make_request()
        cancelled = make_request()
        queue.submit(completed)
        queue.submit(cancelled)

        queue.run_one(echo_job_types)
        queue.cancel(cancelled.job_id)

        assert list(queue.completion_outbox_dir.glob("*.json")) == []
        assert not (queue.queue_dir / "cancelled" / cancelled.job_id / "receipt.json").exists()
