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
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> JobState:
        d = _normalize_job_state_json(json.loads(text))
        d["status"] = JobStatus(d["status"])
        if d.get("warnings") is None:
            d["warnings"] = []
        return cls(**_known_fields(cls, d))


def _known_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted JSON forward-compatible with additive fields."""
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


def _normalize_job_state_json(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize persisted status rows across Greenroom schema revisions."""
    normalized = dict(values)

    aliases = {
        "job_id": ("jobId",),
        "job_type": ("jobType",),
        "input_path": ("inputPath",),
        "output_dir": ("outputDir", "bundleRoot"),
        "submitted_at": ("submittedAt",),
        "started_at": ("startedAt",),
        "finished_at": ("finishedAt",),
        "exit_code": ("exitCode",),
        "failure_phase": ("failurePhase",),
        "error_message": ("errorMessage",),
        "effective_route": ("effectiveRoute",),
    }

    for canonical, legacy_keys in aliases.items():
        if canonical in normalized:
            continue
        for legacy_key in legacy_keys:
            if legacy_key in normalized:
                normalized[canonical] = normalized[legacy_key]
                break

    # Older provider-route status rows did not include all queue-native fields.
    # Listing and status inspection should degrade, not crash, during incidents.
    normalized.setdefault("input_path", "")
    normalized.setdefault("output_dir", "")
    normalized.setdefault("params", {})
    normalized.setdefault("submitted_at", 0.0)

    for numeric_key in ("submitted_at", "started_at", "finished_at"):
        if isinstance(normalized.get(numeric_key), str):
            normalized.pop(numeric_key)

    return normalized
