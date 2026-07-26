"""Aggregate read and execution-start controls over registered Greenroom queues."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .models import JobStatus
from .queue import GPUQueue


class QueueControlError(RuntimeError):
    """Aggregate control failed after producing structured mutation accounting."""

    def __init__(self, report: dict):
        super().__init__(report.get("error_message", "aggregate queue control failed"))
        self.report = report


class QueueRegistry:
    """One-time adapter registry; native queue directories remain state authority."""

    SCHEMA = "gpu-greenroom.queue-registry.v1"

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.receipts_dir = self.path.parent / "queue-control-receipts"

    @contextmanager
    def _lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()

    def _load_unlocked(self) -> dict:
        if not self.path.exists():
            return {"schema": self.SCHEMA, "queues": []}
        payload = json.loads(self.path.read_text())
        if payload.get("schema") != self.SCHEMA:
            raise ValueError(f"unsupported queue registry schema: {payload.get('schema')!r}")
        if not isinstance(payload.get("queues"), list):
            raise ValueError("queue registry queues must be a list")
        return payload

    def _write_atomic(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temp, path)

    def register(
        self,
        *,
        name: str,
        queue_dir: str | Path,
        contention_class: str,
        adapter: str = "greenroom-v1",
    ) -> dict:
        queue_path = Path(queue_dir).expanduser().resolve()
        GPUQueue(queue_path)
        entry = {
            "name": name,
            "queue_dir": str(queue_path),
            "contention_class": contention_class,
            "adapter": adapter,
        }
        with self._lock():
            payload = self._load_unlocked()
            queues = [
                row for row in payload["queues"]
                if row.get("name") != name
            ]
            queues.append(entry)
            payload["queues"] = sorted(queues, key=lambda row: row["name"])
            self._write_atomic(self.path, payload)
        return entry

    def entries(self) -> list[dict]:
        with self._lock():
            return list(self._load_unlocked()["queues"])

    def status(self, contention_class: str | None = None) -> list[dict]:
        entries = self.entries()
        return self._status_entries(entries, contention_class)

    def _status_entries(
        self,
        entries: list[dict],
        contention_class: str | None = None,
    ) -> list[dict]:
        rows = []
        for entry in entries:
            if contention_class and entry["contention_class"] != contention_class:
                continue
            queue_path = Path(entry["queue_dir"])
            if not queue_path.is_dir():
                rows.append({
                    **entry,
                    "source": str(queue_path),
                    "available": False,
                    "error": "queue_dir_missing",
                    "paused": None,
                    "pending": None,
                    "running": None,
                    "running_jobs": [],
                })
                continue
            queue = GPUQueue(queue_path)
            pending = queue.list_jobs(JobStatus.PENDING)
            running = queue.list_jobs(JobStatus.RUNNING)
            rows.append({
                **entry,
                "source": str(queue_path),
                "available": True,
                "error": None,
                "paused": queue.is_paused(),
                "pending": len(pending),
                "running": len(running),
                "running_jobs": [
                    {
                        "job_id": state.job_id,
                        "job_type": state.job_type,
                        "started_at": state.started_at,
                        "effective_route": state.effective_route,
                    }
                    for state in running
                ],
            })
        return rows

    def _set_paused(self, contention_class: str, paused: bool) -> dict:
        action = "pause" if paused else "resume"
        receipt_name = f"{time.time_ns()}-{action}-{uuid.uuid4().hex[:8]}.json"
        receipt_path = self.receipts_dir / receipt_name
        with self._lock():
            entries = self._load_unlocked()["queues"]
            selected = [
                entry for entry in entries
                if entry["contention_class"] == contention_class
            ]
            if not selected:
                raise KeyError(
                    f"no registered queues for contention class {contention_class}"
                )
            missing = [
                Path(entry["queue_dir"])
                for entry in selected
                if not Path(entry["queue_dir"]).is_dir()
            ]
            if missing:
                raise FileNotFoundError(
                    "registered queue is missing: " + ", ".join(map(str, missing))
                )

            mutation_rows = [
                {
                    "name": entry["name"],
                    "queue_dir": entry["queue_dir"],
                    "attempted": False,
                    "mutation": "not-attempted",
                    "previous_paused": (
                        Path(entry["queue_dir"]) / "paused"
                    ).exists(),
                    "observed_paused": (
                        Path(entry["queue_dir"]) / "paused"
                    ).exists(),
                    "error": None,
                }
                for entry in selected
            ]
            mutation_error = None
            for entry, row in zip(selected, mutation_rows):
                queue = GPUQueue(entry["queue_dir"])
                row["attempted"] = True
                try:
                    queue.pause() if paused else queue.resume()
                    row["mutation"] = "succeeded"
                except OSError as error:
                    row["mutation"] = "failed"
                    row["error"] = str(error)
                    mutation_error = error
                finally:
                    row["observed_paused"] = queue.is_paused()
                if mutation_error is not None:
                    break

            report = {
                "schema": "gpu-greenroom.queue-control-receipt.v1",
                "status": "failed" if mutation_error else "succeeded",
                "failure_phase": (
                    "native-marker-mutation" if mutation_error else None
                ),
                "action": action,
                "contention_class": contention_class,
                "observed_at": time.time(),
                "registry_path": str(self.path),
                "receipt_path": str(receipt_path),
                "receipt_persisted": True,
                "rollback_attempted": False,
                "error_message": str(mutation_error) if mutation_error else None,
                "queues": self._status_entries(selected),
                "mutations": mutation_rows,
            }
            try:
                self._write_atomic(receipt_path, report)
            except OSError as receipt_error:
                failure_report = {
                    **report,
                    "status": "failed",
                    "failure_phase": "receipt-write",
                    "primary_failure_phase": report["failure_phase"],
                    "error_message": str(receipt_error),
                    "failed_receipt_path": str(receipt_path),
                    "receipt_path": None,
                    "receipt_persisted": False,
                }
                raise QueueControlError(failure_report) from receipt_error

            if mutation_error is not None:
                raise QueueControlError(report) from mutation_error
            return report

    def pause(self, contention_class: str) -> dict:
        return self._set_paused(contention_class, True)

    def resume(self, contention_class: str) -> dict:
        return self._set_paused(contention_class, False)
