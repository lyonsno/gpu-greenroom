"""Job request and status models for the GPU queue."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CHECKPOINT_PAUSED = "checkpoint_paused"
    CANCELLED = "cancelled"


@dataclass
class JobRequest:
    job_type: str
    input_path: str
    output_dir: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> JobRequest:
        d = json.loads(text)
        return cls(**_known_fields(cls, d))


@dataclass
class JobState:
    job_id: str
    status: JobStatus
    job_type: str
    input_path: str
    output_dir: str
    params: dict[str, Any] = field(default_factory=dict)
    submitted_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    failure_phase: str | None = None
    error_message: str | None = None
    effective_route: str | None = None
    pid: int | None = None
    worker_pid: int | None = None
    child_pid: int | None = None
    process_group_id: int | None = None
    checkpoint_dir: str | None = None
    checkpoint_stop_file: str | None = None
    checkpoint_yield: dict[str, Any] | None = None
    checkpoint_pause_request: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> JobState:
        d = json.loads(text)
        d["status"] = JobStatus(d["status"])
        if d.get("warnings") is None:
            d["warnings"] = []
        return cls(**_known_fields(cls, d))


def _known_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted JSON forward-compatible with additive fields."""
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}
