"""Fail-first contract for `gpu-greenroom lineage`.

Lineage walks receipts, not files. Every job's receipt records its input
(path and digest), its effective route and worker identity, and the digests
of its outputs. Producers are found by digest first and by path second;
consumers the same way in reverse. A directory collected by gc still appears
in lineage because the gc receipt kept the job ids and the artifact digests.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from gpu_queue import lineage as lin

NOW = 1_800_000_000.0


@pytest.fixture
def queue_dir(tmp_path):
    q = tmp_path / "queue"
    for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
        (q / sub).mkdir(parents=True)
    return q


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def job(queue_dir: Path, *, job_type: str, agent: str, input_path: str, input_bytes: bytes | None, outputs: dict[str, bytes],
        name: str, finished: float, status: str = "done", route: str = "python gen.py", commit: str = "abc123") -> str:
    """A terminal job with an artifact manifest, written the way run_one writes receipts."""
    job_id = uuid.uuid4().hex[:12]
    out = queue_dir / "outputs" / name
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel, data in outputs.items():
        (out / rel).write_bytes(data)
        manifest.append({"path": rel, "sha256": sha(data), "size_bytes": len(data)})
    jd = queue_dir / status / job_id
    jd.mkdir(parents=True)
    (jd / "request.json").write_text(json.dumps({"job_type": job_type, "input_path": input_path, "output_dir": str(out),
                                                 "params": {}, "agent_id": agent, "job_id": job_id, "submitted_at": finished - 100,
                                                 "route_identity": route}))
    (jd / "status.json").write_text(json.dumps({"job_id": job_id, "status": status, "job_type": job_type, "input_path": input_path,
                                                "output_dir": str(out), "submitted_at": finished - 100, "started_at": finished - 50,
                                                "finished_at": finished, "effective_route": route}))
    (jd / "receipt.json").write_text(json.dumps({
        "job_id": job_id, "job_type": job_type, "status": status, "input_path": input_path, "output_dir": str(out),
        "effective_route": route, "effective_argv": route.split(), "requested_route": route, "started_at": finished - 50,
        "finished_at": finished, "exit_code": 0 if status == "done" else 1,
        "worker": {"pid": 1, "capabilities": [], "source": {"root": "/src", "commit": commit, "dirty": False}},
        "input_artifact": {"path": input_path, "sha256": sha(input_bytes), "size_bytes": len(input_bytes)} if input_bytes is not None else None,
        "artifact_manifest": manifest or None,
    }))
    return job_id


def chain(queue_dir: Path):
    """plate.png -> A (mesh) -> B (render from A's mesh) -> C (edit from B's render); D is unrelated."""
    plate = queue_dir / "inputs" / "plate.png"
    plate.parent.mkdir()
    plate.write_bytes(b"PLATE")
    a = job(queue_dir, job_type="mesh", agent="lane-a", input_path=str(plate), input_bytes=b"PLATE",
            outputs={"mesh.glb": b"MESH-A"}, name="skull-mesh", finished=NOW - 3000)
    mesh_path = str(queue_dir / "outputs" / "skull-mesh" / "mesh.glb")
    b = job(queue_dir, job_type="render", agent="lane-b", input_path=mesh_path, input_bytes=b"MESH-A",
            outputs={"front.png": b"PNG-B", "side.png": b"PNG-B2"}, name="skull-render", finished=NOW - 2000)
    front = str(queue_dir / "outputs" / "skull-render" / "front.png")
    c = job(queue_dir, job_type="edit", agent="lane-c", input_path=front, input_bytes=b"PNG-B",
            outputs={"edit.png": b"PNG-C"}, name="skull-edit", finished=NOW - 1000)
    d = job(queue_dir, job_type="mesh", agent="lane-d", input_path="/elsewhere/x.png", input_bytes=b"X",
            outputs={"mesh.glb": b"MESH-D"}, name="other-mesh", finished=NOW - 500)
    return {"plate": plate, "a": a, "b": b, "c": c, "d": d, "mesh_path": mesh_path, "front": front}


class TestResolveSubject:
    def test_by_job_id_path_and_digest_resolve_to_the_same_job(self, queue_dir):
        ids = chain(queue_dir)
        by_id = lin.resolve_subject(queue_dir, ids["b"])
        by_path = lin.resolve_subject(queue_dir, ids["front"])
        by_digest = lin.resolve_subject(queue_dir, sha(b"PNG-B"))
        assert by_id["job_id"] == by_path["job_id"] == by_digest["job_id"] == ids["b"]
        assert by_path["matched_by"] == "artifact_path" and by_digest["matched_by"] == "artifact_digest"

    def test_unknown_subject_is_a_structured_error(self, queue_dir):
        chain(queue_dir)
        with pytest.raises(lin.LineageNotFound, match="no receipt"):
            lin.resolve_subject(queue_dir, "deadbeef0000")


class TestGraph:
    def test_ancestors_follow_digests_and_stop_at_external_inputs(self, queue_dir):
        ids = chain(queue_dir)
        g = lin.lineage(queue_dir, ids["c"])
        assert g["subject"]["job_id"] == ids["c"]
        ancestors = [n["job_id"] for n in g["nodes"] if n["relation"] == "ancestor"]
        assert ancestors == [ids["b"], ids["a"]]              # nearest first
        assert ids["d"] not in [n["job_id"] for n in g["nodes"]]
        edges = {(e["from"], e["to"], e["via"]) for e in g["edges"]}
        assert (ids["a"], ids["b"], "sha256:" + sha(b"MESH-A")) in edges
        assert (ids["b"], ids["c"], "sha256:" + sha(b"PNG-B")) in edges
        root = [n for n in g["nodes"] if n["job_id"] == ids["a"]][0]
        assert root["inputs"][0]["path"].endswith("plate.png") and root["inputs"][0]["producer"] is None

    def test_descendants_follow_outputs_forward(self, queue_dir):
        ids = chain(queue_dir)
        g = lin.lineage(queue_dir, ids["a"])
        descendants = [n["job_id"] for n in g["nodes"] if n["relation"] == "descendant"]
        assert descendants == [ids["b"], ids["c"]]

    def test_nodes_carry_route_worker_and_artifacts(self, queue_dir):
        ids = chain(queue_dir)
        g = lin.lineage(queue_dir, ids["b"])
        node = g["subject"]
        assert node["job_type"] == "render" and node["agent_id"] == "lane-b" and node["status"] == "done"
        assert node["effective_route"] == "python gen.py" and node["worker_commit"] == "abc123"
        assert sorted(a["path"] for a in node["artifacts"]) == ["front.png", "side.png"]
        assert node["artifacts"][0]["sha256"]

    def test_deleted_producer_still_appears_through_its_gc_receipt(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        # gc collected A's directory and left a receipt carrying the manifest
        epoch = "e1"
        rdir = queue_dir / "gc-receipts" / epoch
        rdir.mkdir(parents=True)
        (rdir / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": epoch, "name": "skull-mesh",
            "path": str(queue_dir / "outputs" / "skull-mesh"), "job_ids": [ids["a"]],
            "artifact_manifest": [{"path": "mesh.glb", "sha256": sha(b"MESH-A"), "size_bytes": 6}],
            "deleted_at": NOW - 10, "applied_by": "ops",
        }))
        shutil.rmtree(queue_dir / "outputs" / "skull-mesh")
        shutil.rmtree(queue_dir / "done" / ids["a"])      # even the job record is gone

        g = lin.lineage(queue_dir, ids["c"])

        a = [n for n in g["nodes"] if n["job_id"] == ids["a"]]
        assert a and a[0]["deleted"] is True and a[0]["deleted_by_epoch"] == epoch
        assert a[0]["artifacts"][0]["sha256"] == sha(b"MESH-A")
        assert a[0]["record_source"] == "gc-receipt"

    def test_path_subject_hashes_the_file_when_it_exists(self, queue_dir):
        ids = chain(queue_dir)
        # a copy of B's render elsewhere still resolves by content
        copy = queue_dir / "copy.png"
        copy.write_bytes(b"PNG-B")
        g = lin.lineage(queue_dir, str(copy))
        assert g["subject"]["job_id"] == ids["b"] and g["subject_matched_by"] == "content_digest"


class TestCLI:
    def run(self, queue_dir, *args):
        p = subprocess.run([sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), "lineage", *args], capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr

    def test_json_and_text_renderings(self, queue_dir):
        ids = chain(queue_dir)
        rc, out, err = self.run(queue_dir, ids["c"], "--json")
        assert rc == 0, err
        doc = json.loads(out)
        assert doc["schema"] == "gpu-greenroom.lineage.v1" and doc["subject"]["job_id"] == ids["c"]
        rc, out, err = self.run(queue_dir, ids["c"])
        assert rc == 0 and ids["a"] in out and ids["b"] in out and "sha256:" in out
        assert "edit" in out and "render" in out and "mesh" in out

    def test_unknown_subject_exits_one_with_structured_stderr(self, queue_dir):
        chain(queue_dir)
        rc, out, err = self.run(queue_dir, "nope")
        assert rc == 1 and "no receipt" in err and "Traceback" not in err
