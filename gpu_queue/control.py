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
        rows = []
        for entry in self.entries():
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
        selected = [
            entry for entry in self.entries()
            if entry["contention_class"] == contention_class
        ]
        if not selected:
            raise KeyError(f"no registered queues for contention class {contention_class}")
        missing = [
            Path(entry["queue_dir"])
            for entry in selected
            if not Path(entry["queue_dir"]).is_dir()
        ]
        if missing:
            raise FileNotFoundError(
                "registered queue is missing: " + ", ".join(map(str, missing))
            )
        for entry in selected:
            queue = GPUQueue(entry["queue_dir"])
            queue.pause() if paused else queue.resume()

        action = "pause" if paused else "resume"
        receipt = {
            "schema": "gpu-greenroom.queue-control-receipt.v1",
            "action": action,
            "contention_class": contention_class,
            "observed_at": time.time(),
            "registry_path": str(self.path),
            "queues": self.status(contention_class),
        }
        receipt_name = f"{time.time_ns()}-{action}-{contention_class}.json"
        receipt_path = self.receipts_dir / receipt_name
        receipt["receipt_path"] = str(receipt_path)
        self._write_atomic(receipt_path, receipt)
        return receipt

    def pause(self, contention_class: str) -> dict:
        return self._set_paused(contention_class, True)

    def resume(self, contention_class: str) -> dict:
        return self._set_paused(contention_class, False)
