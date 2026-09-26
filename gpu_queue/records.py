"""One reader for the records the queue and the collector write.

The worker writes ``request.json``, ``status.json`` and ``receipt.json`` under one of the
state directories; ``GPUQueue.cancel`` stamps ``finished_at`` on a job that never started;
the collector re-lists every job that ever wrote under a directory each time it collects
that directory, so one job accrues several receipts. Every reader of those files goes
through this module so no reader re-derives the shape on its own. Torn files are reported,
never trusted; a job id found in two state directories resolves to the copy furthest along.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

STATE_DIRS = ("pending", "running", "done", "failed", "cancelled")   # the JobStatus values, in lifecycle order
TERMINAL = ("done", "failed", "cancelled")
_PRECEDENCE = {"pending": 1, "running": 2, "done": 3, "failed": 3, "cancelled": 3}
RECEIPT_SCHEMA = "gpu-greenroom.gc-receipt.v1"


def read_json(path: Path) -> tuple[dict | None, bool]:
    """(document, unreadable). Absent file: (None, False). Present but not a JSON object: (None, True)."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None, False
    except OSError:
        return None, True
    try:
        doc = json.loads(text)
    except ValueError:
        return None, True
    return (doc, False) if isinstance(doc, dict) else (None, True)


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


@dataclass
class JobRecord:
    job_id: str
    state_dir: str
    request: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    receipt: dict = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)    # record files that exist but did not parse
    duplicates: list[str] = field(default_factory=list)    # other state dirs holding this job id, lost to precedence

    @property
    def terminal(self) -> bool:
        return self.state_dir in TERMINAL

    @property
    def readable(self) -> bool:
        return not self.unreadable

    @property
    def status(self) -> str:
        return self.state_dir if self.terminal else (self.state.get("status") or self.state_dir)

    @property
    def started_at(self):
        return _num(self.receipt.get("started_at") or self.state.get("started_at"))

    @property
    def finished_at(self):
        return _num(self.receipt.get("finished_at") or self.state.get("finished_at"))

    @property
    def never_started(self) -> bool:
        """The shape GPUQueue.cancel writes: cancelled, finished_at stamped, started_at never set."""
        return self.status == "cancelled" and self.started_at is None

    @property
    def job_type(self):
        return self.receipt.get("job_type") or self.state.get("job_type") or self.request.get("job_type")

    @property
    def agent_id(self):
        return self.request.get("agent_id")

    @property
    def output_dir(self):
        return self.receipt.get("output_dir") or self.state.get("output_dir") or self.request.get("output_dir")

    @property
    def input_path(self):
        return self.receipt.get("input_path") or self.state.get("input_path") or self.request.get("input_path")


def load_job_records(queue_dir: Path) -> dict[str, JobRecord]:
    """Every job the queue has a record for, keyed by job id, resolved deterministically across state dirs."""
    queue_dir = Path(queue_dir)
    out: dict[str, JobRecord] = {}
    for state in STATE_DIRS:
        state_dir = queue_dir / state
        if not state_dir.is_dir():
            continue
        for job_dir in sorted(state_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            rec = JobRecord(job_dir.name, state)
            for name, attr in (("request.json", "request"), ("status.json", "state"), ("receipt.json", "receipt")):
                doc, torn = read_json(job_dir / name)
                if torn:
                    rec.unreadable.append(name)
                elif doc is not None:
                    setattr(rec, attr, doc)
            prev = out.get(rec.job_id)
            if prev is None:
                out[rec.job_id] = rec
            elif _PRECEDENCE[state] > _PRECEDENCE[prev.state_dir]:
                rec.duplicates = sorted(prev.duplicates + [prev.state_dir])
                out[rec.job_id] = rec
            else:
                prev.duplicates = sorted(prev.duplicates + [state])
    return out


@dataclass
class GcReceipt:
    epoch: str
    name: str
    path: str | None
    job_ids: list[str]
    deleted_at: float | None
    partial: bool
    owner: str | None
    artifact_manifest: list[dict]
    input_artifacts: list[dict]
    artifact_manifest_by_job: dict | None
    input_artifacts_by_job: dict | None
    source: str

    @property
    def confirmed(self) -> bool:
        return self.deleted_at is not None

    @property
    def deletion(self) -> str:
        return "confirmed" if self.confirmed else ("partial" if self.partial else "unconfirmed")

    @property
    def has_maps(self) -> bool:
        return self.artifact_manifest_by_job is not None or self.input_artifacts_by_job is not None


def load_gc_receipts(queue_dir: Path) -> tuple[list[GcReceipt], list[str]]:
    """(receipts in a fixed order: by deletion time, then epoch, then name; paths of torn receipt files)."""
    receipts_dir = Path(queue_dir) / "gc-receipts"
    receipts: list[GcReceipt] = []
    unreadable: list[str] = []
    if not receipts_dir.is_dir():
        return receipts, unreadable
    for path in receipts_dir.glob("*/*.json"):
        if path.name.startswith("_"):
            continue
        doc, torn = read_json(path)
        if torn:
            unreadable.append(str(path))
            continue
        if doc is None or doc.get("schema") != RECEIPT_SCHEMA:
            continue
        receipts.append(GcReceipt(
            epoch=str(doc.get("epoch")), name=str(doc.get("name")), path=doc.get("path") if isinstance(doc.get("path"), str) else None,
            job_ids=[j for j in (doc.get("job_ids") or []) if isinstance(j, str)],
            deleted_at=_num(doc.get("deleted_at")), partial=bool(doc.get("partial")),
            owner=doc.get("owner") if isinstance(doc.get("owner"), str) else None,
            artifact_manifest=[m for m in (doc.get("artifact_manifest") or []) if isinstance(m, dict) and m.get("sha256")],
            input_artifacts=[i for i in (doc.get("input_artifacts") or []) if isinstance(i, dict) and i.get("path")],
            artifact_manifest_by_job=doc.get("artifact_manifest_by_job") if isinstance(doc.get("artifact_manifest_by_job"), dict) else None,
            input_artifacts_by_job=doc.get("input_artifacts_by_job") if isinstance(doc.get("input_artifacts_by_job"), dict) else None,
            source=str(path),
        ))
    receipts.sort(key=lambda r: (r.deleted_at if r.deleted_at is not None else float("inf"), r.epoch, r.name))
    return receipts, sorted(unreadable)


@dataclass
class Deletion:
    deleted_at: float | None = None      # the earliest confirmed deletion of this job's bytes
    epoch: str | None = None             # the epoch of that deletion (or the first epoch that listed the job, if none confirmed)
    deletion: str = "unconfirmed"
    epochs: list[str] = field(default_factory=list)   # every epoch whose receipt lists the job


def deletions_by_job(receipts: list[GcReceipt]) -> dict[str, Deletion]:
    """Merge receipts per job: the earliest confirmed deletion governs; a later partial or unconfirmed receipt never downgrades it."""
    out: dict[str, Deletion] = {}
    for r in sorted(receipts, key=lambda r: (r.deleted_at if r.deleted_at is not None else float("inf"), r.epoch, r.name)):
        for job_id in r.job_ids:
            d = out.setdefault(job_id, Deletion())
            if r.epoch not in d.epochs:
                d.epochs.append(r.epoch)
            if r.confirmed and (d.deleted_at is None or r.deleted_at < d.deleted_at):
                d.deleted_at, d.epoch, d.deletion = r.deleted_at, r.epoch, "confirmed"
            elif not r.confirmed and d.deletion != "confirmed":
                if r.partial:
                    d.deletion = "partial"
                d.epoch = d.epoch or r.epoch
    return out
