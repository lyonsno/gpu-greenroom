"""Lineage over receipts.

Every terminal job leaves a receipt with its input (path and digest), its
effective route and worker identity, and the digests of its outputs. A
directory collected by gc leaves a gc receipt that keeps the job ids and the
artifact digests. Lineage walks those records: producers are found by digest
first and by path second, consumers the same way in reverse. Bytes may be
gone; the graph is not.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = "gpu-greenroom.lineage.v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")


class LineageNotFound(LookupError):
    """No receipt matches the subject."""


def _load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _norm(path: str | None) -> str | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        return str(Path(path).resolve())
    except OSError:
        return os.path.abspath(path)


class Records:
    """Every job the queue knows about, from job records and from gc receipts."""

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.jobs: dict[str, dict] = {}
        self._load_jobs()
        self._load_gc_receipts()
        self._index()

    def _load_jobs(self) -> None:
        for status in ("done", "failed", "cancelled", "running", "pending"):
            status_dir = self.queue_dir / status
            if not status_dir.is_dir():
                continue
            for job_dir in status_dir.iterdir():
                if not job_dir.is_dir():
                    continue
                request = _load_json(job_dir / "request.json") or {}
                state = _load_json(job_dir / "status.json") or {}
                receipt = _load_json(job_dir / "receipt.json") or {}
                job_id = job_dir.name
                worker = receipt.get("worker") or {}
                source = worker.get("source") or {}
                output_dir = _norm(receipt.get("output_dir") or state.get("output_dir") or request.get("output_dir"))
                artifacts = []
                for item in receipt.get("artifact_manifest") or []:
                    if isinstance(item, dict) and item.get("sha256"):
                        artifacts.append({"path": item.get("path"), "sha256": item["sha256"], "size_bytes": item.get("size_bytes")})
                input_path = _norm(receipt.get("input_path") or state.get("input_path") or request.get("input_path"))
                input_artifact = receipt.get("input_artifact") or None
                self.jobs[job_id] = {
                    "job_id": job_id,
                    "status": status if status in ("done", "failed", "cancelled") else (state.get("status") or status),
                    "job_type": receipt.get("job_type") or state.get("job_type") or request.get("job_type"),
                    "agent_id": request.get("agent_id"),
                    "requested_route": receipt.get("requested_route") or request.get("route_identity"),
                    "effective_route": receipt.get("effective_route") or state.get("effective_route"),
                    "effective_argv": receipt.get("effective_argv"),
                    "worker_commit": source.get("commit"),
                    "worker_source_root": source.get("root"),
                    "started_at": receipt.get("started_at") or state.get("started_at"),
                    "finished_at": receipt.get("finished_at") or state.get("finished_at"),
                    "exit_code": receipt.get("exit_code", state.get("exit_code")),
                    "failure_phase": receipt.get("failure_phase") or state.get("failure_phase"),
                    "output_dir": output_dir,
                    "artifacts": artifacts,
                    "inputs": [{"path": input_path, "sha256": (input_artifact or {}).get("sha256"), "producer": None}] if input_path else [],
                    "record_source": "job-record",
                    "deleted": False,
                    "deleted_by_epoch": None,
                }

    def _load_gc_receipts(self) -> None:
        receipts_dir = self.queue_dir / "gc-receipts"
        if not receipts_dir.is_dir():
            return
        for path in receipts_dir.glob("*/*.json"):
            if path.name.startswith("_"):
                continue
            doc = _load_json(path)
            if not isinstance(doc, dict) or doc.get("schema") != "gpu-greenroom.gc-receipt.v1" or not doc.get("deleted_at"):
                continue
            manifest = [{"path": m.get("path"), "sha256": m["sha256"], "size_bytes": m.get("size_bytes")}
                        for m in (doc.get("artifact_manifest") or []) if isinstance(m, dict) and m.get("sha256")]
            for job_id in doc.get("job_ids") or []:
                job = self.jobs.get(job_id)
                if job is None:
                    job = self.jobs[job_id] = {
                        "job_id": job_id, "status": "unknown", "job_type": None, "agent_id": doc.get("owner"),
                        "requested_route": None, "effective_route": None, "effective_argv": None, "worker_commit": None,
                        "worker_source_root": None, "started_at": None, "finished_at": None, "exit_code": None,
                        "failure_phase": None, "output_dir": _norm(doc.get("path")), "artifacts": [], "inputs": [],
                        "record_source": "gc-receipt", "deleted": False, "deleted_by_epoch": None,
                    }
                    for item in (doc.get("input_artifacts") or []):
                        if isinstance(item, dict) and item.get("path"):
                            job["inputs"].append({"path": _norm(item["path"]), "sha256": item.get("sha256"), "producer": None})
                job["deleted"] = True
                job["deleted_by_epoch"] = doc.get("epoch")
                known = {a["sha256"] for a in job["artifacts"]}
                job["artifacts"].extend(m for m in manifest if m["sha256"] not in known)

    def _index(self) -> None:
        self.by_digest: dict[str, list[str]] = {}
        self.by_artifact_path: dict[str, str] = {}
        self.by_output_dir: dict[str, list[str]] = {}
        for job_id, job in self.jobs.items():
            for artifact in job["artifacts"]:
                self.by_digest.setdefault(artifact["sha256"], []).append(job_id)
                if job["output_dir"] and artifact.get("path"):
                    self.by_artifact_path[os.path.join(job["output_dir"], artifact["path"])] = job_id
            if job["output_dir"]:
                self.by_output_dir.setdefault(job["output_dir"], []).append(job_id)

    def producers_of(self, path: str | None, digest: str | None) -> tuple[list[str], str | None]:
        """Jobs that produced this input, and how they were matched."""
        if digest and digest in self.by_digest:
            return sorted(self.by_digest[digest]), f"sha256:{digest}"
        if path:
            if path in self.by_artifact_path:
                return [self.by_artifact_path[path]], f"path:{path}"
            for output_dir, ids in self.by_output_dir.items():
                if path == output_dir or path.startswith(output_dir + os.sep):
                    return sorted(ids), f"path:{path}"
        return [], None

    def consumers_of(self, job_id: str) -> list[tuple[str, str]]:
        job = self.jobs[job_id]
        digests = {a["sha256"] for a in job["artifacts"]}
        found: dict[str, str] = {}
        for other_id, other in self.jobs.items():
            if other_id == job_id:
                continue
            for inp in other["inputs"]:
                if inp.get("sha256") and inp["sha256"] in digests:
                    found.setdefault(other_id, f"sha256:{inp['sha256']}")
                elif inp.get("path") and job["output_dir"] and (inp["path"] == job["output_dir"] or inp["path"].startswith(job["output_dir"] + os.sep)):
                    found.setdefault(other_id, f"path:{inp['path']}")
        return sorted(found.items(), key=lambda kv: (self.jobs[kv[0]].get("finished_at") or 0, kv[0]))


def resolve_subject(queue_dir, subject: str, records: Records | None = None) -> dict:
    """The job a subject names: a job id, a sha256 digest, or a path (artifact path or file content)."""
    queue_dir = Path(queue_dir).resolve()
    records = records or Records(queue_dir)
    if subject in records.jobs:
        return {"job_id": subject, "matched_by": "job_id"}
    if _HEX64.match(subject):
        ids = records.by_digest.get(subject)
        if ids:
            return {"job_id": sorted(ids)[0], "matched_by": "artifact_digest"}
        raise LineageNotFound(f"no receipt records an artifact with digest {subject}")
    norm = _norm(subject)
    if norm:
        if norm in records.by_artifact_path:
            return {"job_id": records.by_artifact_path[norm], "matched_by": "artifact_path"}
        for output_dir, ids in records.by_output_dir.items():
            if norm == output_dir or norm.startswith(output_dir + os.sep):
                return {"job_id": sorted(ids)[0], "matched_by": "output_dir"}
        if Path(norm).is_file():
            digest = _sha256(Path(norm))
            if digest and digest in records.by_digest:
                return {"job_id": sorted(records.by_digest[digest])[0], "matched_by": "content_digest"}
    raise LineageNotFound(f"no receipt matches {subject!r} as a job id, artifact digest, or artifact path")


def lineage(queue_dir, subject: str) -> dict:
    """Ancestors (nearest first) and descendants (nearest first) of the subject job."""
    queue_dir = Path(queue_dir).resolve()
    records = Records(queue_dir)
    match = resolve_subject(queue_dir, subject, records)
    root_id = match["job_id"]
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = {root_id}

    def node(job_id: str, relation: str, depth: int) -> dict:
        job = dict(records.jobs[job_id])
        job["inputs"] = [dict(i) for i in job["inputs"]]
        job["relation"] = relation
        job["depth"] = depth
        return job

    subject_node = node(root_id, "subject", 0)

    # ancestors: breadth-first through inputs
    frontier = [(root_id, subject_node, 0)]
    while frontier:
        job_id, current, depth = frontier.pop(0)
        for inp in current["inputs"]:
            producers, via = records.producers_of(inp.get("path"), inp.get("sha256"))
            inp["producer"] = producers[0] if producers else None
            for producer in producers:
                edges.append({"from": producer, "to": job_id, "via": via})
                if producer not in seen:
                    seen.add(producer)
                    n = node(producer, "ancestor", depth + 1)
                    nodes.append(n)
                    frontier.append((producer, n, depth + 1))

    # descendants: breadth-first through outputs
    frontier = [(root_id, 0)]
    while frontier:
        job_id, depth = frontier.pop(0)
        for consumer, via in records.consumers_of(job_id):
            edges.append({"from": job_id, "to": consumer, "via": via})
            if consumer not in seen:
                seen.add(consumer)
                n = node(consumer, "descendant", depth + 1)
                for inp in n["inputs"]:
                    producers, _ = records.producers_of(inp.get("path"), inp.get("sha256"))
                    inp["producer"] = producers[0] if producers else None
                nodes.append(n)
                frontier.append((consumer, depth + 1))

    nodes.sort(key=lambda n: (0 if n["relation"] == "ancestor" else 1, n["depth"]))
    unique_edges = []
    seen_edges = set()
    for e in edges:
        key = (e["from"], e["to"], e["via"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)
    return {
        "schema": SCHEMA,
        "queue_dir": str(queue_dir),
        "subject": subject_node,
        "subject_matched_by": match["matched_by"],
        "nodes": nodes,
        "edges": unique_edges,
    }


def render_text(graph: dict) -> str:
    import time as _time

    def when(ts):
        return _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ts)) if isinstance(ts, (int, float)) else "?"

    def line(n: dict) -> str:
        flags = []
        if n.get("deleted"):
            flags.append(f"deleted by gc epoch {n.get('deleted_by_epoch')}")
        if n.get("record_source") == "gc-receipt":
            flags.append("record from gc receipt only")
        head = f"{n['job_id']}  {n.get('job_type') or '?':14s} {n.get('status') or '?':8s} {when(n.get('finished_at'))}  owner {n.get('agent_id') or 'not recorded'}"
        route = n.get("effective_route") or n.get("requested_route") or ""
        commit = n.get("worker_commit")
        parts = [head]
        if route:
            parts.append(f"    route: {route[:160]}")
        if commit:
            parts.append(f"    worker commit: {commit}")
        for a in n.get("artifacts") or []:
            parts.append(f"    artifact: {a.get('path')}  sha256:{a['sha256']}")
        if flags:
            parts.append("    " + "; ".join(flags))
        return "\n".join(parts)

    out = [f"subject ({graph['subject_matched_by']}):", line(graph["subject"])]
    ancestors = [n for n in graph["nodes"] if n["relation"] == "ancestor"]
    descendants = [n for n in graph["nodes"] if n["relation"] == "descendant"]
    edge_map = {(e["from"], e["to"]): e["via"] for e in graph["edges"]}
    if ancestors:
        out.append("\nancestors (nearest first):")
        for n in ancestors:
            out.append(("  " * n["depth"]) + line(n).replace("\n", "\n" + "  " * n["depth"]))
    if descendants:
        out.append("\ndescendants (nearest first):")
        for n in descendants:
            out.append(("  " * n["depth"]) + line(n).replace("\n", "\n" + "  " * n["depth"]))
    if graph["edges"]:
        out.append("\nedges:")
        for e in graph["edges"]:
            out.append(f"  {e['from']} -> {e['to']}  via {e['via']}")
    return "\n".join(out)
