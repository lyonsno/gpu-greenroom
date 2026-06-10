"""Filesystem-backed GPU job queue with flock serialization."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .models import JobRequest, JobState, JobStatus


class GPUQueue:
    """Filesystem-backed job queue.

    Directory layout:
        queue_dir/
            pending/     - submitted jobs awaiting execution
            running/     - at most one job being executed
            done/        - completed jobs
            failed/      - failed jobs
            cancelled/   - cancelled jobs
    """

    def __init__(self, queue_dir: str | Path):
        self.queue_dir = Path(queue_dir)
        for sub in ("pending", "running", "done", "failed", "cancelled"):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)

    @property
    def lock_path(self) -> Path:
        return self.queue_dir / "gpu.lock"

    def submit(self, request: JobRequest) -> Path:
        """Submit a job. Returns the job directory path."""
        job_dir = self.queue_dir / "pending" / request.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Write request
        (job_dir / "request.json").write_text(request.to_json())

        # Write initial status
        state = JobState(
            job_id=request.job_id,
            status=JobStatus.PENDING,
            job_type=request.job_type,
            input_path=request.input_path,
            output_dir=request.output_dir,
            params=request.params,
            submitted_at=request.submitted_at,
        )
        (job_dir / "status.json").write_text(state.to_json())

        return job_dir

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job. Returns True if cancelled, False if not found/not pending."""
        pending_dir = self.queue_dir / "pending" / job_id
        if not pending_dir.exists():
            return False

        state = JobState.from_json((pending_dir / "status.json").read_text())
        state.status = JobStatus.CANCELLED
        state.finished_at = time.time()
        (pending_dir / "status.json").write_text(state.to_json())

        dest = self.queue_dir / "cancelled" / job_id
        shutil.move(str(pending_dir), str(dest))
        return True

    def list_jobs(self, status: JobStatus | None = None) -> list[JobState]:
        """List jobs, optionally filtered by status."""
        results = []
        dirs = ["pending", "running", "done", "failed", "cancelled"]
        if status:
            dirs = [status.value]
        for sub in dirs:
            sub_dir = self.queue_dir / sub
            if not sub_dir.exists():
                continue
            for job_dir in sorted(sub_dir.iterdir()):
                status_file = job_dir / "status.json"
                if status_file.exists():
                    results.append(JobState.from_json(status_file.read_text()))
        return results

    def get_job(self, job_id: str) -> JobState | None:
        """Get a specific job's state."""
        for sub in ("pending", "running", "done", "failed", "cancelled"):
            status_file = self.queue_dir / sub / job_id / "status.json"
            if status_file.exists():
                return JobState.from_json(status_file.read_text())
        return None

    def _next_pending(self) -> Path | None:
        """Get the oldest pending job directory."""
        pending = self.queue_dir / "pending"
        jobs = []
        for job_dir in pending.iterdir():
            status_file = job_dir / "status.json"
            if status_file.exists():
                state = JobState.from_json(status_file.read_text())
                jobs.append((state.submitted_at, job_dir))
        if not jobs:
            return None
        jobs.sort(key=lambda x: x[0])
        return jobs[0][1]

    def _move_job(self, job_dir: Path, dest_status: str) -> Path:
        """Move a job directory to a new status folder."""
        dest = self.queue_dir / dest_status / job_dir.name
        shutil.move(str(job_dir), str(dest))
        return dest

    def run_one(self, job_types: dict[str, list[str]]) -> bool:
        """Pick and run the next pending job under flock.

        job_types: mapping of job_type name -> command template list.
            Template strings may contain {input_path}, {output_dir}, and
            any key from params.

        Returns True if a job was run, False if queue was empty.
        """
        lock_fd = open(self.lock_path, "w")
        try:
            # Non-blocking lock attempt
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_fd.close()
            return False

        try:
            job_dir = self._next_pending()
            if job_dir is None:
                return False

            request = JobRequest.from_json((job_dir / "request.json").read_text())

            # Move to running
            job_dir = self._move_job(job_dir, "running")

            # Update status
            state = JobState.from_json((job_dir / "status.json").read_text())
            state.status = JobStatus.RUNNING
            state.started_at = time.time()
            state.pid = os.getpid()

            # Resolve command template
            cmd_template = job_types.get(request.job_type)
            if cmd_template is None:
                state.status = JobStatus.FAILED
                state.finished_at = time.time()
                state.failure_phase = "dispatch"
                state.error_message = f"Unknown job type: {request.job_type}"
                state.exit_code = -1
                (job_dir / "status.json").write_text(state.to_json())
                self._move_job(job_dir, "failed")
                return True

            subs = {
                "input_path": request.input_path,
                "output_dir": request.output_dir,
                **request.params,
            }
            cmd = [part.format(**subs) for part in cmd_template]
            state.effective_route = " ".join(cmd)
            (job_dir / "status.json").write_text(state.to_json())

            # Ensure output directory exists
            os.makedirs(request.output_dir, exist_ok=True)

            # Execute
            stdout_path = job_dir / "stdout.log"
            stderr_path = job_dir / "stderr.log"

            try:
                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    proc = subprocess.run(
                        cmd,
                        stdout=out_f,
                        stderr=err_f,
                        timeout=7200,  # 2 hour max
                    )
                state.exit_code = proc.returncode
                if proc.returncode == 0:
                    state.status = JobStatus.DONE
                    dest_status = "done"
                else:
                    state.status = JobStatus.FAILED
                    state.failure_phase = "execution"
                    state.error_message = f"Process exited with code {proc.returncode}"
                    dest_status = "failed"
            except subprocess.TimeoutExpired:
                state.status = JobStatus.FAILED
                state.failure_phase = "timeout"
                state.error_message = "Job exceeded 2 hour timeout"
                state.exit_code = -1
                dest_status = "failed"
            except Exception as e:
                state.status = JobStatus.FAILED
                state.failure_phase = "launch"
                state.error_message = str(e)
                state.exit_code = -1
                dest_status = "failed"

            state.finished_at = time.time()
            (job_dir / "status.json").write_text(state.to_json())

            # Write receipt
            receipt = {
                "job_id": state.job_id,
                "job_type": state.job_type,
                "status": state.status.value,
                "input_path": state.input_path,
                "output_dir": state.output_dir,
                "effective_route": state.effective_route,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "exit_code": state.exit_code,
                "failure_phase": state.failure_phase,
                "error_message": state.error_message,
            }
            (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

            self._move_job(job_dir, dest_status)
            return True

        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    def recover_stale(self) -> list[str]:
        """Check for stale running jobs (process no longer alive) and move to failed.

        Returns list of recovered job IDs.
        """
        recovered = []
        running_dir = self.queue_dir / "running"
        if not running_dir.exists():
            return recovered

        for job_dir in running_dir.iterdir():
            status_file = job_dir / "status.json"
            if not status_file.exists():
                continue
            state = JobState.from_json(status_file.read_text())
            if state.pid is not None:
                try:
                    os.kill(state.pid, 0)  # check if process is alive
                except ProcessLookupError:
                    # Process is dead — stale job
                    state.status = JobStatus.FAILED
                    state.finished_at = time.time()
                    state.failure_phase = "stale_recovery"
                    state.error_message = f"Process {state.pid} no longer alive; recovered by stale detection"
                    state.exit_code = -1
                    (status_file).write_text(state.to_json())

                    receipt = {
                        "job_id": state.job_id,
                        "job_type": state.job_type,
                        "status": "failed",
                        "failure_phase": "stale_recovery",
                        "error_message": state.error_message,
                        "started_at": state.started_at,
                        "finished_at": state.finished_at,
                    }
                    (job_dir / "receipt.json").write_text(json.dumps(receipt, indent=2))

                    self._move_job(job_dir, "failed")
                    recovered.append(state.job_id)
        return recovered
