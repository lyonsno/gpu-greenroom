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


class LeaseStatus(str, Enum):
    ACTIVE = "active"
    HANDOFF = "handoff"
    RELEASED = "released"
    OWNERSHIP_UNKNOWN = "ownership_unknown"


class BumpStatus(str, Enum):
    PENDING = "pending"
    GRANT_PENDING_CHECKPOINT = "grant_pending_checkpoint"
    GRANTED = "granted"
    DECLINED = "declined"
    CLOSED = "closed"


@dataclass(frozen=True)
class CompletionOutboxRequest:
    """Opt-in routing contract for a producer-local terminal event."""

    target_consumer: str
    target_consumer_id: str
    delivery_mode: str
    notify_on: str = "always"
    producer_report_locator: str | None = None
    evidence_manifest_locator: str | None = None

    def __post_init__(self) -> None:
        if not self.target_consumer:
            raise ValueError("completion outbox target_consumer must not be empty")
        if not self.target_consumer_id:
            raise ValueError("completion outbox target_consumer_id must not be empty")
        if self.delivery_mode not in {"checkpoint", "immediate", "passive"}:
            raise ValueError(
                "completion outbox delivery_mode must be checkpoint, immediate, or passive"
            )
        if self.notify_on not in {"always", "failure"}:
            raise ValueError("completion outbox notify_on must be always or failure")
        if (
            self.producer_report_locator is not None
            and not Path(self.producer_report_locator).expanduser().is_absolute()
        ):
            raise ValueError("completion outbox producer_report_locator must be absolute")
        if (
            self.evidence_manifest_locator is not None
            and not Path(self.evidence_manifest_locator).expanduser().is_absolute()
        ):
            raise ValueError("completion outbox evidence_manifest_locator must be absolute")


@dataclass
class JobRequest:
    job_type: str
    input_path: str
    output_dir: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted_at: float = field(default_factory=time.time)
    completion_outbox: CompletionOutboxRequest | None = None
    output_dir_auto_assigned: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> JobRequest:
        d = json.loads(text)
        if isinstance(d.get("completion_outbox"), dict):
            d["completion_outbox"] = CompletionOutboxRequest(
                **_known_fields(CompletionOutboxRequest, d["completion_outbox"])
            )
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
        d = json.loads(text)
        d["status"] = JobStatus(d["status"])
        if d.get("warnings") is None:
            d["warnings"] = []
        return cls(**_known_fields(cls, d))


@dataclass
class ExternalLease:
    owner: str
    agent_id: str
    repo_root: str
    effective_route: str
    backend: str
    device: str
    profile: str
    supports_checkpoints: bool
    interruptible: bool
    lease_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    lifecycle_state: LeaseStatus = LeaseStatus.ACTIVE
    pid: int | None = None
    process_group: int | None = None
    claimed_at: float = field(default_factory=time.time)
    renewed_at: float = field(default_factory=time.time)
    ttl_seconds: float = 300.0
    handoff_bump_id: str | None = None
    released_at: float | None = None
    released_by: str | None = None
    release_reason: str | None = None
    unknown_at: float | None = None
    unknown_reason: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["lifecycle_state"] = self.lifecycle_state.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ExternalLease:
        d = json.loads(text)
        d["lifecycle_state"] = LeaseStatus(d["lifecycle_state"])
        return cls(**_known_fields(cls, d))


@dataclass
class BumpRequest:
    requester: str
    agent_id: str
    repo_root: str
    intended_route: str
    workload_class: str
    memory_pressure: str
    estimated_occupancy: str
    full_quiescence_required: bool
    reason: str
    callback_address: str
    bump_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: BumpStatus = BumpStatus.PENDING
    requested_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    estimated_occupancy_authority: str = "diagnostic_only"
    holder_lease_id: str | None = None
    granted_by: str | None = None
    granted_at: float | None = None
    checkpoint: str | None = None
    quiescence_confirmed: bool = False
    declined_by: str | None = None
    declined_at: float | None = None
    decline_reason: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> BumpRequest:
        d = json.loads(text)
        d["status"] = BumpStatus(d["status"])
        return cls(**_known_fields(cls, d))


def _known_fields(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    """Keep persisted JSON forward-compatible with additive fields."""
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in names}
