"""Lineage over receipts.

Lineage reconstructs producer and consumer relationships from what the
queue recorded, and only from that. Every terminal job record carries one
``input_path`` and, when the job type configured ``artifact_manifest``,
the digests of its outputs; an input digest exists only when the job type
configured ``source_attestation``. A directory retention collected leaves a
gc receipt that keeps the job ids and digests, per job when the receipt
recorded them.

Edges are labeled by their evidence: ``artifact-path`` (the input is a
recorded artifact of the producer), ``output-dir`` (the input lies inside
the producer's recorded output directory), ``sha256`` (the input's
recorded digest equals a recorded artifact digest), or combinations. A
producer that finished after its consumer started is rejected; a recorded
input digest that contradicts the producer's recorded digest for the same
path rejects that producer; when more than one producer remains, the
input is marked ambiguous and every accepted candidate is shown. Relative
recorded paths are never resolved. Deleting bytes does not delete what the
receipts recorded, and lineage shows exactly what they recorded, no more.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

SCHEMA = "gpu-greenroom.lineage.v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
BASIS_ORDER = ("artifact-path", "output-dir", "sha256")
BASIS_RANK = {"artifact-path": 2.0, "output-dir": 1.0, "sha256": 0.5}
LIMITS = [
    "Each job record carries one input_path; inputs passed through argv or params are not seen.",
    "Input digests exist only for job types that configured source_attestation; other edges rest on recorded paths.",
    "Artifact digests exist only for job types that configured artifact_manifest.",
    "A gc receipt shared by several jobs attributes artifacts and inputs to the directory, not to a job, unless it recorded per-job manifests.",
    "Relative recorded paths are shown but never matched.",
]


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


def _abs(path: str | None) -> tuple[str | None, bool]:
    """(normalized absolute path or the raw value, relative?) — relative paths are never resolved."""
    if not isinstance(path, str) or not path:
        return None, False
    if not os.path.isabs(path):
        return path, True
    try:
        return str(Path(path).resolve()), False
    except OSError:
        return os.path.normpath(path), False


def _num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class Records:
    """Every job the queue knows about, from job records and from gc receipts."""

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.jobs: dict[str, dict] = {}
        self._load_jobs()
        self._load_gc_receipts()
        self._index()

    # ---- loading ------------------------------------------------------
    def _blank(self, job_id: str, **over) -> dict:
        base = {
            "job_id": job_id, "status": "unknown", "job_type": None, "agent_id": None, "requested_route": None,
            "effective_route": None, "effective_argv": None, "worker_commit": None, "worker_source_root": None,
            "started_at": None, "finished_at": None, "exit_code": None, "failure_phase": None,
            "output_dir": None, "output_dir_relative": False, "artifacts": [], "inputs": [], "inputs_attribution": "job",
            "record_source": "job-record", "deleted": False, "deletion": None, "deleted_by_epoch": None,
        }
        base.update(over)
        return base

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
                worker = receipt.get("worker") or {}
                commit = worker.get("commit") or ((worker.get("source") or {}).get("commit"))   # flat is what the worker writes
                source_root = worker.get("source_root") or ((worker.get("source") or {}).get("root"))
                output_dir, out_rel = _abs(receipt.get("output_dir") or state.get("output_dir") or request.get("output_dir"))
                input_path, in_rel = _abs(receipt.get("input_path") or state.get("input_path") or request.get("input_path"))
                input_artifact = receipt.get("input_artifact") if isinstance(receipt.get("input_artifact"), dict) else None
                artifacts = [{"path": m.get("path"), "sha256": m["sha256"], "size_bytes": m.get("size_bytes"), "attribution": "job"}
                             for m in (receipt.get("artifact_manifest") or []) if isinstance(m, dict) and m.get("sha256")]
                inputs = []
                if input_path:
                    inputs.append({"path": input_path, "relative": in_rel, "sha256": input_artifact.get("sha256") if input_artifact else None,
                                   "digest_recorded": bool(input_artifact and input_artifact.get("sha256")), "basis": "job",
                                   "producer": None, "producer_basis": None, "candidates": [], "ambiguous": False})
                self.jobs[job_dir.name] = self._blank(
                    job_dir.name,
                    status=status if status in ("done", "failed", "cancelled") else (state.get("status") or status),
                    job_type=receipt.get("job_type") or state.get("job_type") or request.get("job_type"),
                    agent_id=request.get("agent_id"),
                    requested_route=receipt.get("requested_route") or request.get("route_identity"),
                    effective_route=receipt.get("effective_route") or state.get("effective_route"),
                    effective_argv=receipt.get("effective_argv"), worker_commit=commit, worker_source_root=source_root,
                    started_at=_num(receipt.get("started_at") or state.get("started_at")),
                    finished_at=_num(receipt.get("finished_at") or state.get("finished_at")),
                    exit_code=receipt.get("exit_code", state.get("exit_code")),
                    failure_phase=receipt.get("failure_phase") or state.get("failure_phase"),
                    output_dir=output_dir, output_dir_relative=out_rel, artifacts=artifacts, inputs=inputs,
                )

    def _load_gc_receipts(self) -> None:
        receipts_dir = self.queue_dir / "gc-receipts"
        if not receipts_dir.is_dir():
            return
        for path in receipts_dir.glob("*/*.json"):
            if path.name.startswith("_"):
                continue
            doc = _load_json(path)
            if not isinstance(doc, dict) or doc.get("schema") != "gpu-greenroom.gc-receipt.v1":
                continue
            deletion = "confirmed" if doc.get("deleted_at") else ("partial" if doc.get("partial") else "unconfirmed")
            job_ids = [j for j in (doc.get("job_ids") or []) if isinstance(j, str)]
            by_job = doc.get("artifact_manifest_by_job") if isinstance(doc.get("artifact_manifest_by_job"), dict) else {}
            inputs_by_job = doc.get("input_artifacts_by_job") if isinstance(doc.get("input_artifacts_by_job"), dict) else {}
            flat = [m for m in (doc.get("artifact_manifest") or []) if isinstance(m, dict) and m.get("sha256")]
            flat_inputs = [i for i in (doc.get("input_artifacts") or []) if isinstance(i, dict) and i.get("path")]
            shared = len(job_ids) > 1
            dir_path, dir_rel = _abs(doc.get("path"))
            for job_id in job_ids:
                job = self.jobs.get(job_id)
                if job is None:
                    job = self.jobs[job_id] = self._blank(job_id, agent_id=doc.get("owner"), output_dir=dir_path, output_dir_relative=dir_rel,
                                                          record_source="gc-receipt")
                job["deletion"] = deletion
                job["deleted"] = deletion == "confirmed"
                job["deleted_by_epoch"] = doc.get("epoch")
                known = {a["sha256"] for a in job["artifacts"]}
                if job_id in by_job or not shared:
                    mine = by_job.get(job_id) if job_id in by_job else flat
                    for m in mine or []:
                        if isinstance(m, dict) and m.get("sha256") and m["sha256"] not in known:
                            job["artifacts"].append({"path": m.get("path"), "sha256": m["sha256"], "size_bytes": m.get("size_bytes"), "attribution": "job"})
                            known.add(m["sha256"])
                else:
                    for m in flat:
                        if m["sha256"] not in known:
                            job["artifacts"].append({"path": m.get("path"), "sha256": m["sha256"], "size_bytes": m.get("size_bytes"), "attribution": "gc-receipt-shared"})
                            known.add(m["sha256"])
                if job["record_source"] == "gc-receipt" and not job["inputs"]:
                    if job_id in inputs_by_job and isinstance(inputs_by_job[job_id], dict):
                        i = inputs_by_job[job_id]
                        p, rel = _abs(i.get("path"))
                        job["inputs"].append({"path": p, "relative": rel, "sha256": i.get("sha256"), "digest_recorded": bool(i.get("sha256")),
                                              "basis": "job", "producer": None, "producer_basis": None, "candidates": [], "ambiguous": False})
                    elif not shared:
                        for i in flat_inputs:
                            p, rel = _abs(i.get("path"))
                            job["inputs"].append({"path": p, "relative": rel, "sha256": i.get("sha256"), "digest_recorded": bool(i.get("sha256")),
                                                  "basis": "job", "producer": None, "producer_basis": None, "candidates": [], "ambiguous": False})
                    else:
                        job["inputs_attribution"] = f"shared-across-{len(job_ids)}-jobs"
                        for i in flat_inputs:
                            p, rel = _abs(i.get("path"))
                            job["inputs"].append({"path": p, "relative": rel, "sha256": i.get("sha256"), "digest_recorded": bool(i.get("sha256")),
                                                  "basis": "gc-receipt-shared", "producer": None, "producer_basis": None, "candidates": [], "ambiguous": False})

    def _index(self) -> None:
        self.by_digest: dict[str, set[str]] = {}
        self.by_artifact_path: dict[str, list[str]] = {}
        self.by_output_dir: dict[str, list[str]] = {}
        for job_id, job in self.jobs.items():
            for artifact in job["artifacts"]:
                if artifact["attribution"] != "job":
                    continue   # directory-level evidence never enters the per-job digest index
                self.by_digest.setdefault(artifact["sha256"], set()).add(job_id)
                if job["output_dir"] and not job["output_dir_relative"] and artifact.get("path"):
                    self.by_artifact_path.setdefault(os.path.join(job["output_dir"], artifact["path"]), []).append(job_id)
            if job["output_dir"] and not job["output_dir_relative"]:
                self.by_output_dir.setdefault(job["output_dir"], []).append(job_id)

    # ---- matching -----------------------------------------------------
    def _artifact_digest_at(self, job_id: str, path: str) -> str | None:
        job = self.jobs[job_id]
        if not job["output_dir"]:
            return None
        for a in job["artifacts"]:
            if a.get("path") and os.path.join(job["output_dir"], a["path"]) == path:
                return a["sha256"]
        return None

    def candidates_for(self, inp: dict, consumer_id: str) -> list[dict]:
        """Every producer the evidence could support for one input, accepted or rejected with a reason."""
        if inp.get("relative") or inp.get("basis") == "gc-receipt-shared" or not inp.get("path"):
            return []
        path, digest = inp["path"], inp.get("sha256") if inp.get("digest_recorded") else None
        found: dict[str, set[str]] = {}
        if digest:
            for j in self.by_digest.get(digest, ()):
                found.setdefault(j, set()).add("sha256")
        for j in self.by_artifact_path.get(path, []):
            found.setdefault(j, set()).add("artifact-path")
        best_dir = None
        for out_dir in self.by_output_dir:
            if path == out_dir or path.startswith(out_dir + os.sep):
                if best_dir is None or len(out_dir) > len(best_dir):
                    best_dir = out_dir
        if best_dir:
            for j in self.by_output_dir[best_dir]:
                found.setdefault(j, set()).add("output-dir")
        consumer = self.jobs[consumer_id]
        out = []
        for j, bases in found.items():
            if j == consumer_id:
                continue
            if "artifact-path" in bases:
                bases.discard("output-dir")   # an exact artifact match subsumes containment in the same directory
            producer = self.jobs[j]
            reason, accepted = None, True
            recorded_there = self._artifact_digest_at(j, path) if "artifact-path" in bases else None
            if digest and recorded_there and recorded_there != digest:
                accepted, reason = False, "digest contradicts the producer's recorded digest for this path"
            pf, cs = producer.get("finished_at"), consumer.get("started_at")
            if accepted and pf is not None and cs is not None and pf > cs:
                accepted, reason = False, "producer finished after the consumer started"
            basis = "+".join(b for b in BASIS_ORDER if b in bases)
            score = sum(BASIS_RANK[b] for b in bases)
            via = f"sha256:{digest}" if "sha256" in bases else (f"artifact-path:{path}" if "artifact-path" in bases else f"output-dir:{best_dir}")
            out.append({"job_id": j, "basis": basis, "via": via, "accepted": accepted, "reason": reason, "score": score,
                        "finished_at": pf})
        out.sort(key=lambda c: (not c["accepted"], -c["score"], -(c["finished_at"] or 0), c["job_id"]))
        return out

    def attach_producers(self, job_id: str) -> None:
        for inp in self.jobs[job_id]["inputs"]:
            cands = self.candidates_for(inp, job_id)
            accepted = [c for c in cands if c["accepted"]]
            inp["candidates"] = [{k: v for k, v in c.items() if k != "score"} for c in cands]
            inp["ambiguous"] = len(accepted) > 1
            inp["producer"] = accepted[0]["job_id"] if accepted else None
            inp["producer_basis"] = accepted[0]["basis"] if accepted else None

    def consumers_of(self, job_id: str) -> list[dict]:
        """Jobs whose inputs accept this job as a producer."""
        found = []
        for other_id, other in self.jobs.items():
            if other_id == job_id:
                continue
            for inp in other["inputs"]:
                for c in self.candidates_for(inp, other_id):
                    if c["job_id"] == job_id and c["accepted"]:
                        accepted = [x for x in self.candidates_for(inp, other_id) if x["accepted"]]
                        found.append({"job_id": other_id, "basis": c["basis"], "via": c["via"], "ambiguous": len(accepted) > 1})
        found.sort(key=lambda kv: (self.jobs[kv["job_id"]].get("finished_at") or 0, kv["job_id"]))
        return found


def _primary(records: Records, ids) -> str:
    ids = sorted(set(ids))
    return max(ids, key=lambda j: (records.jobs[j].get("finished_at") or 0, j))


def resolve_subject(queue_dir, subject: str, records: Records | None = None) -> dict:
    """The job a subject names: a job id, a sha256 digest, or a path (artifact path, output dir, or file content)."""
    queue_dir = Path(queue_dir).resolve()
    records = records or Records(queue_dir)
    if subject in records.jobs:
        return {"job_id": subject, "matched_by": "job_id", "ambiguous": False, "candidates": [subject]}

    def answer(ids, how):
        ids = sorted(set(ids))
        return {"job_id": _primary(records, ids), "matched_by": how, "ambiguous": len(ids) > 1, "candidates": ids}

    if _HEX64.match(subject):
        ids = records.by_digest.get(subject)
        if ids:
            return answer(ids, "artifact_digest")
        raise LineageNotFound(f"no receipt records an artifact with digest {subject}")
    norm, _rel = _abs(subject if os.path.isabs(subject) else os.path.abspath(subject))
    if norm:
        if norm in records.by_artifact_path:
            return answer(records.by_artifact_path[norm], "artifact_path")
        best = None
        for out_dir in records.by_output_dir:
            if norm == out_dir or norm.startswith(out_dir + os.sep):
                if best is None or len(out_dir) > len(best):
                    best = out_dir
        if best:
            return answer(records.by_output_dir[best], "output_dir")
        if Path(norm).is_file():
            digest = _sha256(Path(norm))
            if digest and digest in records.by_digest:
                return answer(records.by_digest[digest], "content_digest")
    raise LineageNotFound(f"no receipt matches {subject!r} as a job id, artifact digest, or artifact path")


def lineage(queue_dir, subject: str) -> dict:
    """Ancestors (nearest first) and descendants (nearest first) of the subject job."""
    queue_dir = Path(queue_dir).resolve()
    records = Records(queue_dir)
    match = resolve_subject(queue_dir, subject, records)
    root_id = match["job_id"]
    for job_id in records.jobs:
        records.attach_producers(job_id)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = {root_id}

    def node(job_id: str, relation: str, depth: int) -> dict:
        job = json.loads(json.dumps(records.jobs[job_id]))
        job["relation"] = relation
        job["depth"] = depth
        return job

    subject_node = node(root_id, "subject", 0)
    frontier = [(root_id, 0)]
    while frontier:
        job_id, depth = frontier.pop(0)
        for inp in records.jobs[job_id]["inputs"]:
            accepted = [c for c in inp["candidates"] if c["accepted"]]
            for c in accepted:
                edges.append({"from": c["job_id"], "to": job_id, "via": c["via"], "basis": c["basis"], "ambiguous": len(accepted) > 1})
                if c["job_id"] not in seen:
                    seen.add(c["job_id"])
                    nodes.append(node(c["job_id"], "ancestor", depth + 1))
                    frontier.append((c["job_id"], depth + 1))
    frontier = [(root_id, 0)]
    while frontier:
        job_id, depth = frontier.pop(0)
        for c in records.consumers_of(job_id):
            edges.append({"from": job_id, "to": c["job_id"], "via": c["via"], "basis": c["basis"], "ambiguous": c["ambiguous"]})
            if c["job_id"] not in seen:
                seen.add(c["job_id"])
                nodes.append(node(c["job_id"], "descendant", depth + 1))
                frontier.append((c["job_id"], depth + 1))
    nodes.sort(key=lambda n: (0 if n["relation"] == "ancestor" else 1, n["depth"]))
    unique, seen_edges = [], set()
    for e in edges:
        key = (e["from"], e["to"], e["via"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique.append(e)
    return {
        "schema": SCHEMA, "queue_dir": str(queue_dir), "subject": subject_node, "subject_matched_by": match["matched_by"],
        "subject_ambiguous": match["ambiguous"], "subject_candidates": match["candidates"],
        "nodes": nodes, "edges": unique, "limits": list(LIMITS),
    }


def render_text(graph: dict) -> str:
    import time as _time

    def when(ts):
        return _time.strftime("%Y-%m-%d %H:%M", _time.localtime(ts)) if isinstance(ts, (int, float)) else "?"

    def lines(n: dict) -> list[str]:
        out = [f"{n['job_id']}  {n.get('job_type') or '?':14s} {n.get('status') or '?':8s} {when(n.get('finished_at'))}  owner {n.get('agent_id') or 'not recorded'}"]
        route = n.get("effective_route") or n.get("requested_route") or ""
        if route:
            out.append(f"    route: {route[:160]}{'…' if len(route) > 160 else ''}")
        out.append(f"    worker commit: {n.get('worker_commit') or 'not recorded'}")
        for a in n.get("artifacts") or []:
            tag = "" if a.get("attribution") == "job" else f"  ({a.get('attribution')})"
            out.append(f"    artifact: {a.get('path')}  sha256:{a['sha256']}{tag}")
        for i in n.get("inputs") or []:
            digest = f"sha256:{i['sha256']}" if i.get("digest_recorded") else "no digest recorded"
            if i.get("relative"):
                prod = "relative path, not matched"
            elif i.get("basis") == "gc-receipt-shared":
                prod = "attributed to the directory by a shared gc receipt, not matched"
            elif i.get("producer"):
                prod = f"<- {i['producer']} ({i['producer_basis']})" + (f", ambiguous among {sum(c['accepted'] for c in i['candidates'])}" if i.get("ambiguous") else "")
            else:
                rejected = [c for c in i.get("candidates") or [] if not c["accepted"]]
                prod = "external / not found" + (f"; rejected {len(rejected)}: " + "; ".join(f"{c['job_id']} ({c['reason']})" for c in rejected) if rejected else "")
            out.append(f"    input: {i.get('path')}  [{digest}]  {prod}")
        if n.get("deletion") == "confirmed":
            out.append(f"    deleted by gc epoch {n.get('deleted_by_epoch')}")
        elif n.get("deletion"):
            out.append(f"    deletion {n['deletion']} (gc epoch {n.get('deleted_by_epoch')})")
        if n.get("record_source") == "gc-receipt":
            out.append("    record from gc receipt only" + (f"; inputs {n['inputs_attribution']}" if n.get("inputs_attribution") != "job" else ""))
        return out

    def block(n: dict) -> str:
        pad = "  " * n.get("depth", 0)
        return "\n".join(pad + l for l in lines(n))

    out = [f"subject ({graph['subject_matched_by']}{', ambiguous among ' + str(len(graph['subject_candidates'])) if graph.get('subject_ambiguous') else ''}):", block(graph["subject"])]
    ancestors = [n for n in graph["nodes"] if n["relation"] == "ancestor"]
    descendants = [n for n in graph["nodes"] if n["relation"] == "descendant"]
    if ancestors:
        out.append("\nancestors (nearest first):")
        out.extend(block(n) for n in ancestors)
    if descendants:
        out.append("\ndescendants (nearest first):")
        out.extend(block(n) for n in descendants)
    if graph["edges"]:
        out.append("\nedges:")
        for e in graph["edges"]:
            out.append(f"  {e['from']} -> {e['to']}  via {e['via']}  [{e['basis']}{'; ambiguous' if e.get('ambiguous') else ''}]")
    out.append("\nlimits:")
    out.extend(f"  - {l}" for l in graph.get("limits", []))
    return "\n".join(out)
