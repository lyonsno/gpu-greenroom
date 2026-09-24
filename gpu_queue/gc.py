"""Retention for the queue's own outputs/: classify, pin by declaration, collect in two phases.

Policy (operator-approved 2026-09-24):

- Every output has a class and an owner. Class comes from ``output_class`` on
  the job type in job_types.json or on a structured command request; owner is
  the job's ``agent_id``. An entry that any unclassified job wrote into is
  ``unclassified``.
- TTL by class: intermediate 30 days, witness 60 days, final 180 days.
- Unclassified entries are reported for one cycle, then treated as
  intermediate: once a dry-run has listed an entry as unclassified and the
  grace window has passed, later scans classify it ``intermediate`` with
  ``class_source`` ``graduated``.
- Pins are declared with ``gpu-greenroom retain``; nothing infers a pin. An
  unreadable pins file fails closed: nothing is a candidate and apply refuses.
- Collection is two-phase. ``gc --dry-run`` writes an epoch-bound candidate
  list with a per-row snapshot, an apply-not-before time, and one notice per
  owner. ``gc --apply --epoch`` deletes only that list, once, after the grace
  window, holding any row whose entry changed, was pinned, or is referenced
  by a pending or running job (as output, input, or argv) at execution time.
- Every deletion writes a receipt, with the job's artifact-manifest digests
  when they exist, before the bytes go. A failed removal leaves a partial
  receipt and the run still writes its summary.
- Only direct children of the queue's own outputs/ are ever removed.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

CANDIDATES_SCHEMA = "gpu-greenroom.gc-candidates.v1"
RECEIPT_SCHEMA = "gpu-greenroom.gc-receipt.v1"
APPLY_SCHEMA = "gpu-greenroom.gc-apply.v1"
NOTICE_SCHEMA = "gpu-greenroom.gc-notice.v1"
PINS_SCHEMA = "gpu-greenroom.retention-pins.v1"
CLASSES = ("final", "witness", "intermediate")  # longest TTL first: the conservative choice on conflict
DEFAULT_TTL_DAYS = {"intermediate": 30.0, "witness": 60.0, "final": 180.0}
DEFAULT_GRACE_HOURS = 72.0
MIN_GRACE_HOURS = 72.0   # the approved window; there is no shorter grace
DAY = 86400.0
NOT_RECORDED = "not-recorded"


class GCRefused(RuntimeError):
    """The apply request could not be honored; nothing was deleted."""


class PinsUnreadable(RuntimeError):
    """retention/pins.json exists but cannot be trusted; retention fails closed."""


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


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class RetentionPins:
    """Declared retention pins in retention/pins.json, edited under a file lock."""

    def __init__(self, queue_dir):
        self.queue_dir = Path(queue_dir)
        self.path = self.queue_dir / "retention" / "pins.json"
        self.lock_path = self.queue_dir / "retention" / "pins.lock"

    @contextmanager
    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self) -> dict:
        if not self.path.exists():
            return {"schema": PINS_SCHEMA, "pins": {}}
        doc = _load_json(self.path)
        if not isinstance(doc, dict) or doc.get("schema") != PINS_SCHEMA or not isinstance(doc.get("pins"), dict):
            raise PinsUnreadable(f"retention pins file is unreadable or has an unknown schema: {self.path}")
        for name, entry in doc["pins"].items():
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise PinsUnreadable(f"retention pin entry for {name!r} is not an object: {self.path}")
        return doc

    def entries(self) -> dict:
        return self.load()["pins"]

    def pin(self, name: str, *, owner: str, reason: str, until: float | None = None) -> dict:
        if not isinstance(name, str) or not name.strip() or "/" in name or name in (".", ".."):
            raise ValueError("pin name must be a top-level outputs/ entry name")
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("pin owner is required")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("pin reason is required")
        if until is not None and not _is_number(until):
            raise ValueError("pin until must be a timestamp")
        with self._locked():
            doc = self.load()
            entry = {"owner": owner, "reason": reason, "pinned_at": time.time(), "until": until}
            doc["pins"][name] = entry
            _write_json_atomic(self.path, doc)
        return entry

    def unpin(self, name: str) -> bool:
        with self._locked():
            doc = self.load()
            if name not in doc["pins"]:
                return False
            del doc["pins"][name]
            _write_json_atomic(self.path, doc)
        return True

    def is_pinned(self, name: str, now: float) -> bool:
        """A pin holds until its ``until`` passes; an ``until`` nobody can read holds forever."""
        entry = self.entries().get(name)
        if not entry:
            return False
        until = _parse_until(entry.get("until"))
        if until is _UNPARSEABLE:
            return True   # fail closed: a malformed deadline never unpins
        return until is None or until > now


_UNPARSEABLE = object()


def _parse_until(value):
    """None (no deadline), a timestamp, or _UNPARSEABLE."""
    if value is None:
        return None
    if _is_number(value) and math.isfinite(value):
        return float(value)
    if isinstance(value, str) and value.strip():
        from datetime import datetime, timezone
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return _UNPARSEABLE
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    return _UNPARSEABLE


def _top_level_name(candidate: str | None, outputs: Path) -> str | None:
    """Top-level entry name for an absolute path under outputs/, else None.

    Relative paths are ignored rather than resolved against the process's
    working directory, so a record can never attach to an entry by accident.
    """
    if not isinstance(candidate, str) or not candidate or not os.path.isabs(candidate):
        return None
    try:
        rel = Path(candidate).resolve().relative_to(outputs)
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
            finished = state.get("finished_at")
            if not _is_number(finished):
                finished = receipt.get("finished_at") if _is_number(receipt.get("finished_at")) else None
            records.setdefault(name, []).append({
                "job_id": job_dir.name,
                "status": status,
                "job_type": state.get("job_type") or request.get("job_type"),
                "agent_id": request.get("agent_id"),
                "finished_at": finished,
                "declared_class": request.get("output_class"),
                "artifact_manifest": receipt.get("artifact_manifest"),
                "input_artifact": receipt.get("input_artifact"),
            })
    return records


def _strings_in(value):
    """Every string anywhere inside a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings_in(item)


def _names_in_string(text: str, outputs: Path, outputs_prefix: str) -> set[str]:
    """Entry names a string refers to: as a path (symlinks resolved), an option value, or a substring."""
    names: set[str] = set()
    pieces = [text] + [piece for piece in text.split("=") if piece]
    for piece in pieces:
        name = _top_level_name(piece, outputs)
        if name:
            names.add(name)
    if outputs_prefix in text:
        tail = text.split(outputs_prefix, 1)[1]
        name = tail.split(os.sep, 1)[0]
        if name:
            names.add(name)
    return names


def _active_refs(queue_dir: Path) -> set[str]:
    """Top-level entries any string in a pending or running job's records refers to.

    Covers output_dir, input_path, command_cwd, params values, and argv,
    with symlinked paths resolved to their real location.
    """
    outputs = (queue_dir / "outputs").resolve()
    outputs_prefix = str(outputs) + os.sep
    names: set[str] = set()
    for status in ("pending", "running"):
        status_dir = queue_dir / status
        if not status_dir.is_dir():
            continue
        for job_dir in status_dir.iterdir():
            for record_name in ("request.json", "status.json"):
                doc = _load_json(job_dir / record_name)
                for text in _strings_in(doc):
                    names |= _names_in_string(text, outputs, outputs_prefix)
    return names


def resolve_class(records: list[dict], job_types: dict) -> tuple[str, str | None]:
    """Class for an entry.

    Every job that wrote into the entry must carry a class; the longest-TTL
    class wins when they differ. Any unclassified job makes the entry
    unclassified (``class_source`` ``mixed`` when others were classified).
    """
    found: dict[str, str] = {}
    unclassified_jobs = 0
    for record in records:
        declared = record.get("declared_class")
        if declared in CLASSES:
            found.setdefault(declared, "request")
            continue
        config = job_types.get(record.get("job_type") or "")
        if isinstance(config, dict) and config.get("output_class") in CLASSES:
            found.setdefault(config["output_class"], "job_type")
            continue
        unclassified_jobs += 1
    if unclassified_jobs:
        return "unclassified", ("mixed" if found else None)
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


def _history_dir(queue_dir: Path) -> Path:
    return queue_dir / "gc-history"


def reported_unclassified(queue_dir, *, now: float) -> set[str]:
    """Entries a dry-run listed as unclassified in a cycle whose own apply deadline has passed."""
    queue_dir = Path(queue_dir).resolve()
    names: set[str] = set()
    history = _history_dir(queue_dir)
    if not history.is_dir():
        return names
    for path in history.glob("*.json"):
        doc = _load_json(path)
        if not isinstance(doc, dict) or doc.get("schema") != CANDIDATES_SCHEMA:
            continue
        deadline = doc.get("apply_not_before")
        if not _is_number(deadline):
            created, grace = doc.get("created_at"), doc.get("grace_hours")
            if not (_is_number(created) and _is_number(grace)):
                continue
            deadline = created + grace * 3600.0
        if deadline > now:
            continue
        for row in doc.get("rows", []):
            if row.get("reason") == "unclassified" and row.get("name"):
                names.add(row["name"])
    return names


def scan(queue_dir, job_types: dict, *, now: float, ttl_days: dict | None = None, compute_size: bool = True) -> list[dict]:
    """One row per top-level directory in outputs/, with class, owner, age, snapshot, and candidacy."""
    queue_dir = Path(queue_dir).resolve()
    outputs = queue_dir / "outputs"
    ttl = {**DEFAULT_TTL_DAYS, **(ttl_days or {})}
    records = _job_records(queue_dir)
    active = _active_refs(queue_dir)
    graduated = reported_unclassified(queue_dir, now=now)
    pins = RetentionPins(queue_dir)
    try:
        pins.load()
        pins_readable = True
    except PinsUnreadable:
        pins_readable = False
    rows = []
    if not outputs.is_dir():
        return rows
    for entry in sorted(outputs.iterdir()):
        if entry.is_symlink() or not entry.is_dir():
            continue
        recs = records.get(entry.name, [])
        output_class, class_source = resolve_class(recs, job_types)
        if output_class == "unclassified" and class_source is None and entry.name in graduated:
            output_class, class_source = "intermediate", "graduated"   # never for mixed entries: a declared class must be seen
        finished = [r["finished_at"] for r in recs if _is_number(r.get("finished_at"))]
        newest_finished = max(finished) if finished else None
        dir_mtime = entry.stat().st_mtime
        if newest_finished is not None:
            age_days = (now - newest_finished) / DAY
            age_source = "job_finished_at"
        else:
            age_days = (now - dir_mtime) / DAY
            age_source = "mtime"
        owners = sorted({r["agent_id"] for r in recs if isinstance(r.get("agent_id"), str) and r["agent_id"].strip()})
        owner = owners[0] if len(owners) == 1 else (",".join(owners) if owners else None)
        pinned = pins.is_pinned(entry.name, now) if pins_readable else None
        is_active = entry.name in active
        limit = ttl.get(output_class)
        candidate = False
        if not pins_readable:
            reason = "pins_unreadable"
        elif is_active:
            reason = "active"
        elif pinned:
            reason = "pinned"
        elif output_class == "unclassified" or limit is None:
            reason = "unclassified"
        elif age_days <= limit:
            reason = "within_ttl"
        else:
            candidate = True
            reason = "graduated" if class_source == "graduated" else "past_ttl"
        rows.append({
            "name": entry.name,
            "path": str(entry),
            "output_class": output_class,
            "class_source": class_source,
            "owner": owner,
            "job_ids": sorted(r["job_id"] for r in recs),
            "age_days": round(age_days, 2),
            "age_source": age_source,
            "ttl_days": limit,
            "pinned": pinned,
            "active": is_active,
            "size_bytes": _dir_size(entry) if compute_size else None,
            "candidate": candidate,
            "reason": reason,
            "snapshot": {"newest_finished_at": newest_finished, "dir_mtime": dir_mtime, "job_ids": sorted(r["job_id"] for r in recs)},
        })
    return rows


def write_candidates(queue_dir, rows: list[dict], *, now: float, grace_hours: float = DEFAULT_GRACE_HOURS,
                     authority: str | None = None) -> dict:
    """Write the epoch-bound candidate list, its history copy, and one notice per owner."""
    if not _is_number(grace_hours) or not math.isfinite(grace_hours) or grace_hours < MIN_GRACE_HOURS:
        raise ValueError(f"grace window must be a finite number of hours, at least {MIN_GRACE_HOURS:g} (the approved window)")
    queue_dir = Path(queue_dir).resolve()
    epoch = uuid.uuid4().hex[:12]

    def total(predicate):
        return sum((r.get("size_bytes") or 0) for r in rows if predicate(r))

    doc = {
        "schema": CANDIDATES_SCHEMA,
        "epoch": epoch,
        "status": "open",
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
            "graduated_count": sum(r["candidate"] and r["reason"] == "graduated" for r in rows),
            "graduated_bytes": total(lambda r: r["candidate"] and r["reason"] == "graduated"),
            "unclassified_count": sum(r["reason"] == "unclassified" for r in rows),
            "unclassified_bytes": total(lambda r: r["reason"] == "unclassified"),
            "pinned_count": sum(bool(r["pinned"]) for r in rows),
            "active_count": sum(r["active"] for r in rows),
            "pins_readable": all(r["reason"] != "pins_unreadable" for r in rows),
        },
        "rows": rows,
    }
    _write_json_atomic(queue_dir / "gc-candidates.json", doc)
    _write_json_atomic(_history_dir(queue_dir) / f"{epoch}.json", doc)
    by_owner: dict[str, list[dict]] = {}
    for row in rows:
        if row["candidate"]:
            by_owner.setdefault(row["owner"] or NOT_RECORDED, []).append(row)
    for owner, owner_rows in by_owner.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in owner) or NOT_RECORDED
        _write_json_atomic(queue_dir / "gc-notices" / epoch / f"{safe}.json", {
            "schema": NOTICE_SCHEMA,
            "epoch": epoch,
            "owner": owner,
            "created_at": now,
            "apply_not_before": doc["apply_not_before"],
            "candidate_bytes": sum((r.get("size_bytes") or 0) for r in owner_rows),
            "candidates": [{"name": r["name"], "size_bytes": r["size_bytes"], "output_class": r["output_class"],
                            "age_days": r["age_days"], "reason": r["reason"]} for r in owner_rows],
            "how_to_keep": "gpu-greenroom retain <name> --owner <you> --reason <why> before apply_not_before",
        })
    return doc


def _entry_changed(row: dict, current: dict | None, created_at: float) -> bool:
    snap = row.get("snapshot") or {}
    if current is None:
        return True
    if sorted(current["job_ids"]) != sorted(snap.get("job_ids") or []):
        return True
    newest_then = snap.get("newest_finished_at")
    newest_now = current["newest_finished_at"]
    if (newest_now is not None) != (newest_then is not None) or (newest_now is not None and newest_now > newest_then):
        return True
    return current["dir_mtime"] > created_at


def apply(queue_dir, *, epoch: str, owner: str, now: float) -> dict:
    """Delete exactly the candidate list for ``epoch``, once, after its grace window, with receipts."""
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
    receipts_dir = queue_dir / "gc-receipts" / epoch
    if doc.get("status") == "applied" or (receipts_dir / "_summary.json").exists():
        raise GCRefused(f"epoch {epoch} was already applied; run a new gc --dry-run")
    if not _is_number(doc.get("apply_not_before")) or not _is_number(doc.get("created_at")):
        raise GCRefused("candidate list is malformed: apply_not_before or created_at is not a number")
    if now < float(doc["apply_not_before"]):
        remaining = (float(doc["apply_not_before"]) - now) / 3600.0
        raise GCRefused(f"grace window has not elapsed: {remaining:.1f} h remaining")
    pins = RetentionPins(queue_dir)
    try:
        pins.load()
    except PinsUnreadable as exc:
        raise GCRefused(f"retention pins are unreadable; refusing to collect: {exc}") from exc
    candidates = [r for r in doc.get("rows", []) if r.get("candidate")]
    for row in candidates:
        target = Path(row.get("path", ""))
        if not row.get("name") or target.name != row["name"] or target.resolve().parent != outputs.resolve():
            raise GCRefused(f"candidate path outside outputs/: {row.get('path')}")

    created_at = float(doc["created_at"])
    records = _job_records(queue_dir)
    deleted = 0
    freed = 0
    holds: list[dict] = []
    receipts: list[str] = []
    summary = {
        "schema": APPLY_SCHEMA, "epoch": epoch, "applied_by": owner, "applied_at": None,
        "deleted": 0, "held": 0, "holds": holds, "freed_bytes": 0, "receipts": receipts, "completed": False,
    }
    try:
        for row in candidates:
            name = row["name"]
            target = outputs / name
            if target.is_symlink() or not target.is_dir():
                holds.append({"name": name, "reason": "missing"})
                continue
            recs = records.get(name, [])
            finished = [r["finished_at"] for r in recs if _is_number(r.get("finished_at"))]
            current = {"job_ids": [r["job_id"] for r in recs], "newest_finished_at": max(finished) if finished else None,
                       "dir_mtime": target.stat().st_mtime}
            if _entry_changed(row, current, created_at):
                holds.append({"name": name, "reason": "changed_since_dry_run"})
                continue
            try:
                if pins.is_pinned(name, now):
                    holds.append({"name": name, "reason": "pinned"})
                    continue
            except PinsUnreadable:
                holds.append({"name": name, "reason": "pins_unreadable"})
                continue
            if name in _active_refs(queue_dir):   # re-read per row: a job may have been submitted mid-run
                holds.append({"name": name, "reason": "active"})
                continue
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
                "reason": row.get("reason"),
                "owner": row.get("owner"),
                "applied_by": owner,
                "authority": doc.get("authority"),
                "size_bytes": size,
                "age_days": row.get("age_days"),
                "ttl_days": row.get("ttl_days"),
                "job_ids": [r["job_id"] for r in recs],
                "snapshot": row.get("snapshot"),
                "artifact_manifest": manifests or None,
                "input_artifacts": inputs or None,
                "written_at": time.time(),
                "deleted_at": None,
                "partial": False,
            }
            receipt_path = receipts_dir / f"{name}.json"
            _write_json_atomic(receipt_path, receipt)   # the receipt exists before the bytes go
            try:
                shutil.rmtree(target)
            except OSError as exc:
                receipt["partial"] = True
                receipt["error"] = str(exc)[:300]
                _write_json_atomic(receipt_path, receipt)
                holds.append({"name": name, "reason": "delete_failed", "error": str(exc)[:300]})
                continue
            receipt["deleted_at"] = time.time()
            _write_json_atomic(receipt_path, receipt)
            deleted += 1
            freed += size
            receipts.append(str(receipt_path))
        summary["completed"] = True
    finally:
        summary.update({"applied_at": time.time(), "deleted": deleted, "held": len(holds), "freed_bytes": freed})
        _write_json_atomic(receipts_dir / "_summary.json", summary)
        doc["status"] = "applied"
        doc["applied_at"] = summary["applied_at"]
        doc["applied_by"] = owner
        _write_json_atomic(_history_dir(queue_dir) / f"{epoch}.json", doc)
        current = _load_json(candidates_path) if candidates_path.exists() else None
        if isinstance(current, dict) and current.get("epoch") == epoch:
            _write_json_atomic(candidates_path, doc)   # a newer dry-run written meanwhile is left alone
    return summary
