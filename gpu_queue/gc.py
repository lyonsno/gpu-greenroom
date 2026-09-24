"""Retention for the queue's own outputs/: classify, pin by declaration, collect in two phases.

Policy (operator-approved 2026-09-24):

- Every output has a class and an owner. Class comes from ``output_class`` on
  the job type in job_types.json or on a structured command request; owner is
  the job's ``agent_id``. Missing class is ``unclassified``: reported, never
  collected.
- TTL by class: intermediate 30 days, witness 60 days, final 180 days.
- Pins are declared with ``gpu-greenroom retain``; nothing infers a pin.
- Collection is two-phase. ``gc --dry-run`` writes an epoch-bound candidate
  list and an apply-not-before time. ``gc --apply --epoch`` deletes only that
  list, after the grace window, re-checking pins and active jobs per row.
- Every deletion writes a receipt, with the job's artifact-manifest digests
  when they exist, before the bytes go. Provenance outlives the bytes.
- Only direct children of the queue's own outputs/ are ever removed.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from pathlib import Path

CANDIDATES_SCHEMA = "gpu-greenroom.gc-candidates.v1"
RECEIPT_SCHEMA = "gpu-greenroom.gc-receipt.v1"
APPLY_SCHEMA = "gpu-greenroom.gc-apply.v1"
PINS_SCHEMA = "gpu-greenroom.retention-pins.v1"
CLASSES = ("final", "witness", "intermediate")  # longest TTL first: the conservative choice on conflict
DEFAULT_TTL_DAYS = {"intermediate": 30.0, "witness": 60.0, "final": 180.0}
DAY = 86400.0


class GCRefused(RuntimeError):
    """The apply request could not be honored; nothing was deleted."""


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


class RetentionPins:
    """Declared retention pins in retention/pins.json."""

    def __init__(self, queue_dir):
        self.queue_dir = Path(queue_dir)
        self.path = self.queue_dir / "retention" / "pins.json"

    def load(self) -> dict:
        doc = _load_json(self.path) if self.path.exists() else None
        if not isinstance(doc, dict) or doc.get("schema") != PINS_SCHEMA or not isinstance(doc.get("pins"), dict):
            return {"schema": PINS_SCHEMA, "pins": {}}
        return doc

    def entries(self) -> dict:
        return self.load()["pins"]

    def pin(self, name: str, *, owner: str, reason: str, until: float | None = None) -> dict:
        if not isinstance(name, str) or not name.strip() or "/" in name:
            raise ValueError("pin name must be a top-level outputs/ entry name")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("pin owner is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("pin reason is required")
        doc = self.load()
        entry = {"owner": owner, "reason": reason, "pinned_at": time.time(), "until": until}
        doc["pins"][name] = entry
        _write_json_atomic(self.path, doc)
        return entry

    def unpin(self, name: str) -> bool:
        doc = self.load()
        if name not in doc["pins"]:
            return False
        del doc["pins"][name]
        _write_json_atomic(self.path, doc)
        return True

    def is_pinned(self, name: str, now: float) -> bool:
        entry = self.entries().get(name)
        if not entry:
            return False
        until = entry.get("until")
        return until is None or float(until) > now


def _top_level_name(output_dir: str | None, outputs: Path) -> str | None:
    if not output_dir:
        return None
    try:
        rel = Path(output_dir).resolve().relative_to(outputs)
    except (ValueError, OSError):
        return None
    return rel.parts[0] if rel.parts else None


def _job_records(queue_dir: Path) -> dict[str, list[dict]]:
    """Every terminal job that wrote under outputs/, keyed by top-level entry name."""
    outputs = (queue_dir / "outputs").resolve()
    records: dict[str, list[dict]] = {}
    for status in ("done", "failed", "cancelled"):
        status_dir = queue_dir / status
        if not status_dir.is_dir():
            continue
        for job_dir in status_dir.iterdir():
            request = _load_json(job_dir / "request.json") or {}
            state = _load_json(job_dir / "status.json") or {}
            receipt = _load_json(job_dir / "receipt.json") or {}
            output_dir = state.get("output_dir") or request.get("output_dir") or receipt.get("output_dir")
            name = _top_level_name(output_dir, outputs)
            if name is None:
                continue
            records.setdefault(name, []).append({
                "job_id": job_dir.name,
                "status": status,
                "job_type": state.get("job_type") or request.get("job_type"),
                "agent_id": request.get("agent_id"),
                "finished_at": state.get("finished_at") or receipt.get("finished_at"),
                "declared_class": request.get("output_class"),
                "artifact_manifest": receipt.get("artifact_manifest"),
                "input_artifact": receipt.get("input_artifact"),
            })
    return records


def _active_refs(queue_dir: Path) -> set[str]:
    """Top-level entries referenced by a pending or running job."""
    outputs = (queue_dir / "outputs").resolve()
    names: set[str] = set()
    for status in ("pending", "running"):
        status_dir = queue_dir / status
        if not status_dir.is_dir():
            continue
        for job_dir in status_dir.iterdir():
            for record_name in ("request.json", "status.json"):
                doc = _load_json(job_dir / record_name) or {}
                name = _top_level_name(doc.get("output_dir"), outputs)
                if name:
                    names.add(name)
    return names


def resolve_class(records: list[dict], job_types: dict) -> tuple[str, str | None]:
    """Class for an entry: the longest-TTL class any of its jobs declared, else unclassified."""
    found: dict[str, str] = {}
    for record in records:
        declared = record.get("declared_class")
        if declared in CLASSES:
            found.setdefault(declared, "request")
            continue
        config = job_types.get(record.get("job_type") or "")
        if isinstance(config, dict) and config.get("output_class") in CLASSES:
            found.setdefault(config["output_class"], "job_type")
    for cls in CLASSES:
        if cls in found:
            return cls, found[cls]
    return "unclassified", None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for file in files:
            try:
                total += os.lstat(os.path.join(root, file)).st_size
            except OSError:
                continue
    return total


def scan(queue_dir, job_types: dict, *, now: float, ttl_days: dict | None = None, compute_size: bool = True) -> list[dict]:
    """One row per top-level directory in outputs/, with class, owner, age, and candidacy."""
    queue_dir = Path(queue_dir).resolve()
    outputs = queue_dir / "outputs"
    ttl = {**DEFAULT_TTL_DAYS, **(ttl_days or {})}
    records = _job_records(queue_dir)
    active = _active_refs(queue_dir)
    pins = RetentionPins(queue_dir)
    rows = []
    if not outputs.is_dir():
        return rows
    for entry in sorted(outputs.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        recs = records.get(entry.name, [])
        output_class, class_source = resolve_class(recs, job_types)
        finished = [r["finished_at"] for r in recs if isinstance(r.get("finished_at"), (int, float))]
        if finished:
            age_days = (now - max(finished)) / DAY
            age_source = "job_finished_at"
        else:
            age_days = (now - entry.stat().st_mtime) / DAY
            age_source = "mtime"
        owners = sorted({r["agent_id"] for r in recs if isinstance(r.get("agent_id"), str) and r["agent_id"].strip()})
        owner = owners[0] if len(owners) == 1 else (",".join(owners) if owners else None)
        pinned = pins.is_pinned(entry.name, now)
        is_active = entry.name in active
        limit = ttl.get(output_class)
        candidate = False
        if is_active:
            reason = "active"
        elif pinned:
            reason = "pinned"
        elif output_class == "unclassified" or limit is None:
            reason = "unclassified"
        elif age_days <= limit:
            reason = "within_ttl"
        else:
            candidate = True
            reason = "past_ttl"
        rows.append({
            "name": entry.name,
            "path": str(entry),
            "output_class": output_class,
            "class_source": class_source,
            "owner": owner,
            "job_ids": [r["job_id"] for r in recs],
            "age_days": round(age_days, 2),
            "age_source": age_source,
            "ttl_days": limit,
            "pinned": pinned,
            "active": is_active,
            "size_bytes": _dir_size(entry) if compute_size else None,
            "candidate": candidate,
            "reason": reason,
        })
    return rows


def write_candidates(queue_dir, rows: list[dict], *, now: float, grace_hours: float = 72.0, authority: str | None = None) -> dict:
    """Write the epoch-bound candidate list that a later apply must name exactly."""
    queue_dir = Path(queue_dir).resolve()
    epoch = uuid.uuid4().hex[:12]

    def total(predicate):
        return sum((r.get("size_bytes") or 0) for r in rows if predicate(r))

    doc = {
        "schema": CANDIDATES_SCHEMA,
        "epoch": epoch,
        "queue_dir": str(queue_dir),
        "created_at": now,
        "grace_hours": grace_hours,
        "apply_not_before": now + grace_hours * 3600.0,
        "authority": authority,
        "totals": {
            "entry_count": len(rows),
            "total_bytes": total(lambda r: True),
            "candidate_count": sum(r["candidate"] for r in rows),
            "candidate_bytes": total(lambda r: r["candidate"]),
            "unclassified_count": sum(r["reason"] == "unclassified" for r in rows),
            "unclassified_bytes": total(lambda r: r["reason"] == "unclassified"),
            "pinned_count": sum(r["pinned"] for r in rows),
            "active_count": sum(r["active"] for r in rows),
        },
        "rows": rows,
    }
    _write_json_atomic(queue_dir / "gc-candidates.json", doc)
    return doc


def apply(queue_dir, *, epoch: str, owner: str, now: float) -> dict:
    """Delete exactly the candidate list for ``epoch`` after its grace window, with receipts."""
    queue_dir = Path(queue_dir).resolve()
    outputs = queue_dir / "outputs"
    if not isinstance(owner, str) or not owner.strip():
        raise GCRefused("apply requires --owner")
    candidates_path = queue_dir / "gc-candidates.json"
    doc = _load_json(candidates_path) if candidates_path.exists() else None
    if not isinstance(doc, dict) or doc.get("schema") != CANDIDATES_SCHEMA:
        raise GCRefused("no candidate list; run gc --dry-run first")
    if doc.get("epoch") != epoch:
        raise GCRefused(f"epoch mismatch: candidate list is {doc.get('epoch')}, apply requested {epoch}")
    if now < float(doc.get("apply_not_before") or 0):
        remaining = (float(doc["apply_not_before"]) - now) / 3600.0
        raise GCRefused(f"grace window has not elapsed: {remaining:.1f} h remaining")
    candidates = [r for r in doc.get("rows", []) if r.get("candidate")]
    for row in candidates:
        target = Path(row.get("path", ""))
        if not row.get("name") or target.name != row["name"] or target.resolve().parent != outputs.resolve():
            raise GCRefused(f"candidate path outside outputs/: {row.get('path')}")

    receipts_dir = queue_dir / "gc-receipts" / epoch
    pins = RetentionPins(queue_dir)
    active = _active_refs(queue_dir)
    records = _job_records(queue_dir)
    deleted = 0
    freed = 0
    holds = []
    receipts = []
    for row in candidates:
        name = row["name"]
        target = outputs / name
        if target.is_symlink() or not target.is_dir():
            holds.append({"name": name, "reason": "missing"})
            continue
        if pins.is_pinned(name, now):
            holds.append({"name": name, "reason": "pinned"})
            continue
        if name in active:
            holds.append({"name": name, "reason": "active"})
            continue
        recs = records.get(name, [])
        manifests = [m for r in recs for m in (r.get("artifact_manifest") or [])]
        inputs = [r["input_artifact"] for r in recs if r.get("input_artifact")]
        size = _dir_size(target)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "epoch": epoch,
            "name": name,
            "path": str(target),
            "output_class": row.get("output_class"),
            "class_source": row.get("class_source"),
            "owner": row.get("owner"),
            "applied_by": owner,
            "authority": doc.get("authority"),
            "size_bytes": size,
            "age_days": row.get("age_days"),
            "ttl_days": row.get("ttl_days"),
            "job_ids": [r["job_id"] for r in recs],
            "artifact_manifest": manifests or None,
            "input_artifacts": inputs or None,
            "written_at": time.time(),
            "deleted_at": None,
        }
        receipt_path = receipts_dir / f"{name}.json"
        _write_json_atomic(receipt_path, receipt)   # the receipt exists before the bytes go
        shutil.rmtree(target)
        receipt["deleted_at"] = time.time()
        _write_json_atomic(receipt_path, receipt)
        deleted += 1
        freed += size
        receipts.append(str(receipt_path))
    summary = {
        "schema": APPLY_SCHEMA,
        "epoch": epoch,
        "applied_by": owner,
        "applied_at": time.time(),
        "deleted": deleted,
        "held": len(holds),
        "holds": holds,
        "freed_bytes": freed,
        "receipts": receipts,
    }
    _write_json_atomic(receipts_dir / "_summary.json", summary)
    return summary
