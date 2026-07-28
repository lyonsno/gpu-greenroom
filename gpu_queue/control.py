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
from .queue import GPUQueue, PauseStateError


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
                    "pause_state": None,
                })
                continue
            try:
                queue = GPUQueue(queue_path)
            except OSError as error:
                rows.append({
                    **entry,
                    "source": str(queue_path),
                    "available": False,
                    "error": f"queue_initialization_failed: {error}",
                    "paused": (queue_path / "paused").exists(),
                    "pending": None,
                    "running": None,
                    "running_jobs": [],
                    "pause_state": None,
                })
                continue
            try:
                pending = queue.list_jobs(JobStatus.PENDING)
                running = queue.list_jobs(JobStatus.RUNNING)
            except Exception as error:
                rows.append({
                    **entry,
                    "source": str(queue_path),
                    "available": True,
                    "error": f"queue_status_failed: {error}",
                    "paused": queue.is_paused(),
                    "pending": None,
                    "running": None,
                    "running_jobs": [],
                    "pause_state": None,
                })
                continue
            pause_state = None
            pause_state_error = None
            try:
                pause_state = queue.pause_state()
            except Exception as error:
                pause_state_error = f"pause_state_invalid: {error}"
            rows.append({
                **entry,
                "source": str(queue_path),
                "available": True,
                "error": pause_state_error,
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
                "pause_state": pause_state,
            })
        return rows

    def _set_paused(
        self,
        contention_class: str,
        paused: bool,
        *,
        owner: str,
        epoch: str | None,
    ) -> dict:
        action = "pause" if paused else "resume"
        if not owner.strip():
            raise ValueError("pause control owner must not be empty")
        if not paused and not epoch:
            raise ValueError("resume requires the exact pause epoch")
        epoch = epoch or uuid.uuid4().hex
        requested_at = time.time()
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
                    "acknowledgement": None,
                }
                for entry in selected
            ]
            mutation_error = None
            failure_phase = None
            for entry, row in zip(selected, mutation_rows):
                row["attempted"] = True
                try:
                    queue = GPUQueue(entry["queue_dir"])
                    acknowledgement = (
                        queue.pause(
                            owner=owner,
                            epoch=epoch,
                            contention_class=contention_class,
                            requested_at=requested_at,
                        )
                        if paused
                        else queue.resume(owner=owner, epoch=epoch)
                    )
                    row["acknowledgement"] = {
                        "name": entry["name"],
                        "adapter": entry["adapter"],
                        **acknowledgement,
                    }
                    row["mutation"] = "succeeded"
                except (OSError, PauseStateError, ValueError) as error:
                    row["mutation"] = "failed"
                    row["error"] = str(error)
                    mutation_error = error
                    failure_phase = (
                        "pause-epoch-mismatch"
                        if isinstance(error, PauseStateError)
                        else "native-marker-mutation"
                    )
                finally:
                    row["observed_paused"] = (
                        Path(entry["queue_dir"]) / "paused"
                    ).exists()
                if mutation_error is not None:
                    break

            request_queues = [
                {
                    "name": entry["name"],
                    "queue_dir": str(Path(entry["queue_dir"]).resolve()),
                    "adapter": entry["adapter"],
                }
                for entry in selected
            ]
            acknowledgements = []
            for entry, row in zip(selected, mutation_rows):
                acknowledgement = row["acknowledgement"]
                if acknowledgement is None:
                    acknowledgement = {
                        "name": entry["name"],
                        "adapter": entry["adapter"],
                        "schema": "gpu-greenroom.pause-acknowledgement.v1",
                        "action": action,
                        "owner": owner,
                        "epoch": epoch,
                        "queue_dir": str(Path(entry["queue_dir"]).resolve()),
                        "acknowledged_at": None,
                        "effective_paused": row["observed_paused"],
                        "idempotent": False,
                        "error": row["error"],
                    }
                acknowledgements.append(acknowledgement)
            fully_effective = (
                mutation_error is None
                and all(row["observed_paused"] is paused for row in mutation_rows)
            )
            report = {
                "schema": "gpu-greenroom.queue-control-receipt.v1",
                "status": "failed" if mutation_error else "succeeded",
                "failure_phase": failure_phase,
                "action": action,
                "contention_class": contention_class,
                "requested_at": requested_at,
                "observed_at": time.time(),
                "registry_path": str(self.path),
                "receipt_path": str(receipt_path),
                "receipt_persisted": True,
                "rollback_attempted": False,
                "error_message": str(mutation_error) if mutation_error else None,
                "request": {
                    "action": action,
                    "owner": owner,
                    "epoch": epoch,
                    "contention_class": contention_class,
                    "queues": request_queues,
                },
                "effective": {
                    "fully_effective": fully_effective,
                    "acknowledgements": acknowledgements,
                },
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

    def pause(
        self,
        contention_class: str,
        *,
        owner: str = "unspecified",
        epoch: str | None = None,
    ) -> dict:
        return self._set_paused(
            contention_class,
            True,
            owner=owner,
            epoch=epoch,
        )

    def resume(
        self,
        contention_class: str,
        *,
        owner: str = "unspecified",
        epoch: str | None = None,
    ) -> dict:
        return self._set_paused(
            contention_class,
            False,
            owner=owner,
            epoch=epoch,
        )
