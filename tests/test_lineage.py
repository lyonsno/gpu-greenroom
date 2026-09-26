"""Fail-first contract for `gpu-greenroom lineage`.

Lineage walks receipts, not files. A job record carries one input path, a
digest only when its job type attested the input, and artifact digests only
when it configured a manifest. Producers are matched by recorded artifact
path, then output-directory containment, then digest, with time order and
recorded-digest contradictions rejecting candidates and ambiguity shown. A
directory collected by gc still appears through the gc receipt's job ids
and digests.
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
from tests.realqueue import RealQueue

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
        "worker": {"pid": 1, "capabilities": [], "commit": commit, "source_root": "/src", "git_dirty": False},
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


class TestMatchingIsTruthful:
    """Findings from the wrong-object assessment: a digest match must not override a recorded path,
    a producer must have finished before its consumer started, ambiguity must be visible,
    gc-only reconstruction must not invent per-job inputs, and a reused output directory
    must not merge old and new producers."""

    def test_digest_collision_with_a_later_job_does_not_displace_the_path_producer(self, queue_dir):
        ids = chain(queue_dir)
        # an unrelated job finishes AFTER C started, with an artifact whose bytes equal C's input
        late = job(queue_dir, job_type="edit", agent="lane-z", input_path="/elsewhere/z.png", input_bytes=b"Z",
                   outputs={"copy.png": b"PNG-B"}, name="late-copy", finished=NOW - 100)
        g = lin.lineage(queue_dir, ids["c"])
        subject_input = g["subject"]["inputs"][0]
        assert subject_input["producer"] == ids["b"]
        assert subject_input["producer_basis"] == "artifact-path+sha256"
        rejected = {c["job_id"]: c for c in subject_input["candidates"]}
        assert rejected[late]["accepted"] is False and "after" in rejected[late]["reason"]
        assert late not in [n["job_id"] for n in g["nodes"] if n["relation"] == "ancestor"]

    def test_equal_bytes_from_two_valid_producers_is_marked_ambiguous_not_guessed(self, queue_dir):
        ids = chain(queue_dir)
        # a second legitimate producer of identical bytes that finished before C started, with no path relation
        twin = job(queue_dir, job_type="render", agent="lane-t", input_path="/elsewhere/t.png", input_bytes=b"T",
                   outputs={"front.png": b"PNG-B"}, name="twin-render", finished=NOW - 1500)
        g = lin.lineage(queue_dir, ids["c"])
        subject_input = g["subject"]["inputs"][0]
        assert subject_input["producer"] == ids["b"]          # the recorded path wins
        assert subject_input["ambiguous"] is True
        assert {c["job_id"] for c in subject_input["candidates"] if c["accepted"]} == {ids["b"], twin}
        edges = {(e["from"], e["to"]): e for e in g["edges"]}
        assert edges[(ids["b"], ids["c"])]["basis"] == "artifact-path+sha256"
        assert edges[(twin, ids["c"])]["basis"] == "sha256" and edges[(twin, ids["c"])]["ambiguous"] is True

    def test_gc_only_reconstruction_does_not_invent_per_job_inputs(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        # two jobs shared one output directory; gc kept a flattened list of both inputs
        second = job(queue_dir, job_type="mesh", agent="lane-a", input_path="/other/plate2.png", input_bytes=b"PLATE2",
                     outputs={"mesh2.glb": b"MESH-A2"}, name="skull-mesh", finished=NOW - 2900)
        rdir = queue_dir / "gc-receipts" / "e2"
        rdir.mkdir(parents=True)
        (rdir / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e2", "name": "skull-mesh", "path": str(queue_dir / "outputs" / "skull-mesh"),
            "job_ids": [ids["a"], second], "deleted_at": NOW - 10, "applied_by": "ops",
            "artifact_manifest": [{"path": "mesh.glb", "sha256": sha(b"MESH-A"), "size_bytes": 6}, {"path": "mesh2.glb", "sha256": sha(b"MESH-A2"), "size_bytes": 7}],
            "input_artifacts": [{"path": str(ids["plate"]), "sha256": sha(b"PLATE"), "size_bytes": 5}, {"path": "/other/plate2.png", "sha256": sha(b"PLATE2"), "size_bytes": 6}],
        }))
        shutil.rmtree(queue_dir / "outputs" / "skull-mesh")
        shutil.rmtree(queue_dir / "done" / ids["a"])
        shutil.rmtree(queue_dir / "done" / second)
        g = lin.lineage(queue_dir, ids["b"])
        a = [n for n in g["nodes"] if n["job_id"] == ids["a"]][0]
        assert a["record_source"] == "gc-receipt"
        assert all(i["basis"] == "gc-receipt-shared" for i in a["inputs"])   # attributed to the directory, not to this job
        assert a["inputs_attribution"] == "shared-across-2-jobs"

    def test_reused_output_directory_after_gc_does_not_merge_producers(self, queue_dir):
        ids = chain(queue_dir)
        # a later job reuses the name skull-render after B; C consumed B's bytes before that job existed
        later = job(queue_dir, job_type="render", agent="lane-n", input_path="/elsewhere/n.png", input_bytes=b"N",
                    outputs={"front.png": b"PNG-NEW"}, name="skull-render", finished=NOW - 50)
        g = lin.lineage(queue_dir, ids["c"])
        ancestors = [n["job_id"] for n in g["nodes"] if n["relation"] == "ancestor"]
        assert ids["b"] in ancestors and later not in ancestors
        g2 = lin.lineage(queue_dir, later)
        assert ids["c"] not in [n["job_id"] for n in g2["nodes"]]   # C did not consume the new job's output

    def test_coverage_limits_are_stated_in_the_graph(self, queue_dir):
        ids = chain(queue_dir)
        g = lin.lineage(queue_dir, ids["c"])
        assert "one input_path" in " ".join(g["limits"]) and "argv" in " ".join(g["limits"])


class TestRealWorkerReceipts:
    """Receipts written by the real worker, not by hand."""

    def _run(self, queue_dir, job_types, request):
        from gpu_queue.queue import GPUQueue
        q = GPUQueue(queue_dir)
        q.submit(request)
        assert q.run_one(job_types) is True

    def test_chain_through_the_real_worker_shows_worker_commit_and_labels_edges_truthfully(self, queue_dir, tmp_path):
        from gpu_queue.models import JobRequest
        from gpu_queue.queue import worker_identity
        writer = ("import pathlib, sys; out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
                  "(out / 'mesh.glb').write_bytes(b'MESH')")
        reader = ("import pathlib, sys; out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); "
                  "(out / 'view.png').write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-VIEW')")
        job_types = {
            "mesh": {"cmd": [sys.executable, "-c", writer, "{output_dir}"], "artifact_manifest": ["mesh.glb"]},
            "render": {"cmd": [sys.executable, "-c", reader, "{input_path}", "{output_dir}"], "artifact_manifest": ["view.png"]},
        }
        mesh_out = queue_dir / "outputs" / "real-mesh"
        a = JobRequest(job_type="mesh", input_path=str(tmp_path / "plate.png"), output_dir=str(mesh_out), agent_id="lane-a")
        (tmp_path / "plate.png").write_bytes(b"PLATE")
        self._run(queue_dir, job_types, a)
        b = JobRequest(job_type="render", input_path=str(mesh_out / "mesh.glb"), output_dir=str(queue_dir / "outputs" / "real-render"), agent_id="lane-b")
        self._run(queue_dir, job_types, b)

        g = lin.lineage(queue_dir, b.job_id)

        assert g["subject"]["worker_commit"] == worker_identity()["commit"]
        assert g["subject"]["worker_commit"], "the worker records its commit; lineage must show it"
        anc = [n for n in g["nodes"] if n["relation"] == "ancestor"]
        assert [n["job_id"] for n in anc] == [a.job_id]
        edge = [e for e in g["edges"] if e["from"] == a.job_id and e["to"] == b.job_id][0]
        assert edge["basis"] == "artifact-path"          # no attestation was configured, so no input digest exists
        assert edge["via"].startswith("artifact-path:")
        assert g["subject"]["inputs"][0]["sha256"] is None and g["subject"]["inputs"][0]["digest_recorded"] is False
        text = lin.render_text(g)
        assert "worker commit" in text and "no digest recorded" in text and "artifact-path" in text

    def test_contradicting_recorded_digest_draws_no_edge(self, queue_dir):
        ids = chain(queue_dir)
        # tamper C's recorded input digest so it no longer matches what B recorded for that path
        jd = queue_dir / "done" / ids["c"]
        r = json.loads((jd / "receipt.json").read_text())
        r["input_artifact"]["sha256"] = "ff" * 32
        (jd / "receipt.json").write_text(json.dumps(r))
        g = lin.lineage(queue_dir, ids["c"])
        assert [n["job_id"] for n in g["nodes"] if n["relation"] == "ancestor"] == []
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] is None and any(c["reason"].startswith("digest contradicts") for c in inp["candidates"])

    def test_relative_recorded_paths_are_marked_and_never_resolved(self, queue_dir, monkeypatch):
        ids = chain(queue_dir)
        jd = queue_dir / "done" / ids["c"]
        for name in ("receipt.json", "status.json", "request.json"):
            doc = json.loads((jd / name).read_text())
            doc["input_path"] = "front.png"
            (jd / name).write_text(json.dumps(doc))
        monkeypatch.chdir(queue_dir / "outputs" / "skull-render")   # the file exists here by name
        g = lin.lineage(queue_dir, ids["c"])
        inp = g["subject"]["inputs"][0]
        assert inp["relative"] is True and inp["producer"] is None and inp["candidates"] == []

    def test_unconfirmed_deletion_is_shown_as_unconfirmed(self, queue_dir):
        ids = chain(queue_dir)
        rdir = queue_dir / "gc-receipts" / "e3"
        rdir.mkdir(parents=True)
        (rdir / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e3", "name": "skull-mesh", "path": str(queue_dir / "outputs" / "skull-mesh"),
            "job_ids": [ids["a"]], "artifact_manifest": [{"path": "mesh.glb", "sha256": sha(b"MESH-A"), "size_bytes": 6}],
            "deleted_at": None, "partial": True, "error": "simulated", "applied_by": "ops",
        }))
        g = lin.lineage(queue_dir, ids["b"])
        a = [n for n in g["nodes"] if n["job_id"] == ids["a"]][0]
        assert a["deleted"] is False and a["deletion"] == "partial" and a["deleted_by_epoch"] == "e3"
        assert "deletion partial" in lin.render_text(g)

    def test_ambiguous_subject_resolution_is_reported(self, queue_dir):
        ids = chain(queue_dir)
        twin = job(queue_dir, job_type="render", agent="lane-t", input_path="/elsewhere/t.png", input_bytes=b"T",
                   outputs={"front.png": b"PNG-B"}, name="twin-render", finished=NOW - 1500)
        m = lin.resolve_subject(queue_dir, sha(b"PNG-B"))
        assert m["ambiguous"] is True and set(m["candidates"]) == {ids["b"], twin}
        assert m["job_id"] == twin          # the most recently finished candidate is primary (twin finished after B)


class TestRevisionTwo:
    def test_a_queued_job_into_a_reused_directory_is_never_a_producer(self, queue_dir):
        ids = chain(queue_dir)
        # the ordinary rerun pattern: a new job is queued into B's directory but has not run
        pj = queue_dir / "pending" / "p-rerun"
        pj.mkdir()
        (pj / "request.json").write_text(json.dumps({"job_type": "render", "input_path": "/x.png", "output_dir": str(queue_dir / "outputs" / "skull-render"), "params": {}, "job_id": "p-rerun"}))
        (pj / "status.json").write_text(json.dumps({"job_id": "p-rerun", "status": "pending", "job_type": "render", "output_dir": str(queue_dir / "outputs" / "skull-render")}))
        g = lin.lineage(queue_dir, ids["c"])
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] == ids["b"] and inp["ambiguous"] is False
        rejected = {c["job_id"]: c["reason"] for c in inp["candidates"] if not c["accepted"]}
        assert "p-rerun" in rejected and "not finished" in rejected["p-rerun"]
        assert "p-rerun" not in [n["job_id"] for n in g["nodes"]]
        assert [n["job_id"] for n in lin.lineage(queue_dir, "p-rerun")["nodes"] if n["relation"] == "descendant"] == []

    def test_consumer_search_reuses_attached_producers(self, queue_dir, monkeypatch):
        ids = chain(queue_dir)
        calls = {"n": 0}
        real = lin.Records.candidates_for
        def counting(self, inp, consumer_id):
            calls["n"] += 1
            return real(self, inp, consumer_id)
        monkeypatch.setattr(lin.Records, "candidates_for", counting)
        lin.lineage(queue_dir, ids["a"])
        assert calls["n"] == 4   # exactly once per input across all jobs (the chain has 4 inputs); a doubled pass would be 8

    def test_gc_receipt_with_per_job_maps_is_authoritative_for_jobs_missing_from_them(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        plain = job(queue_dir, job_type="mesh", agent="lane-a", input_path="/other/p.png", input_bytes=None, outputs={}, name="skull-mesh", finished=NOW - 2900)
        rdir = queue_dir / "gc-receipts" / "e4"
        rdir.mkdir(parents=True)
        (rdir / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e4", "name": "skull-mesh", "path": str(queue_dir / "outputs" / "skull-mesh"),
            "job_ids": [ids["a"], plain], "deleted_at": NOW - 10, "applied_by": "ops",
            "artifact_manifest": [{"path": "mesh.glb", "sha256": sha(b"MESH-A"), "size_bytes": 6}],
            "artifact_manifest_by_job": {ids["a"]: [{"path": "mesh.glb", "sha256": sha(b"MESH-A"), "size_bytes": 6}]},
            "input_artifacts_by_job": {},
        }))
        shutil.rmtree(queue_dir / "outputs" / "skull-mesh")
        g = lin.lineage(queue_dir, ids["b"])
        node = {n["job_id"]: n for n in g["nodes"] + [g["subject"]]}
        # the plain job produced nothing according to the map; it must not be credited with A's mesh
        assert all(a["attribution"] == "job" for a in node[ids["a"]]["artifacts"])
        assert plain not in node or node[plain]["artifacts"] == []

    def test_digest_known_only_at_directory_level_says_so(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        second = job(queue_dir, job_type="mesh", agent="lane-a", input_path="/o/2.png", input_bytes=None, outputs={"m2.glb": b"M2"}, name="skull-mesh", finished=NOW - 2900)
        rdir = queue_dir / "gc-receipts" / "e5"
        rdir.mkdir(parents=True)
        (rdir / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e5", "name": "skull-mesh", "path": str(queue_dir / "outputs" / "skull-mesh"),
            "job_ids": [ids["a"], second], "deleted_at": NOW - 10, "applied_by": "ops",
            "artifact_manifest": [{"path": "only-in-receipt.bin", "sha256": "ee" * 32, "size_bytes": 1}],
        }))
        shutil.rmtree(queue_dir / "done" / ids["a"]); shutil.rmtree(queue_dir / "done" / second)
        with pytest.raises(lin.LineageNotFound, match="directory level.*e5"):
            lin.resolve_subject(queue_dir, "ee" * 32)


class TestRevisionThree:
    """Record shapes the real writers produce, which the revision-two rules assumed away."""

    def _job_types(self):
        writer = ("import pathlib, sys; out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
                  "(out / 'mesh.glb').write_bytes(b'MESH')")
        reader = ("import pathlib, sys; out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); "
                  "(out / 'view.png').write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-VIEW')")
        return {
            "mesh": {"cmd": [sys.executable, "-c", writer, "{output_dir}"], "artifact_manifest": ["mesh.glb"], "output_class": "intermediate"},
            "render": {"cmd": [sys.executable, "-c", reader, "{input_path}", "{output_dir}"], "artifact_manifest": ["view.png"]},
        }

    def test_a_job_cancelled_by_the_real_queue_before_it_started_is_never_a_producer(self, queue_dir, tmp_path):
        from gpu_queue.models import JobRequest
        from gpu_queue.queue import GPUQueue
        jt = self._job_types(); q = GPUQueue(queue_dir)
        m = queue_dir / "outputs" / "m"
        (tmp_path / "plate.png").write_bytes(b"PLATE")
        a = JobRequest(job_type="mesh", input_path=str(tmp_path / "plate.png"), output_dir=str(m), agent_id="lane-a")
        q.submit(a); assert q.run_one(jt) is True
        x = JobRequest(job_type="mesh", input_path=str(tmp_path / "plate.png"), output_dir=str(m), agent_id="lane-x")
        q.submit(x); assert q.cancel(x.job_id)
        st = json.loads((queue_dir / "cancelled" / x.job_id / "status.json").read_text())
        assert st["started_at"] is None and st["finished_at"] is not None   # the shape GPUQueue.cancel really writes
        b = JobRequest(job_type="render", input_path=str(m / "mesh.glb"), output_dir=str(queue_dir / "outputs" / "r"), agent_id="lane-b")
        q.submit(b); assert q.run_one(jt) is True

        g = lin.lineage(queue_dir, b.job_id)
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] == a.job_id and inp["ambiguous"] is False
        xc = [c for c in inp["candidates"] if c["job_id"] == x.job_id][0]
        assert xc["accepted"] is False and "cancelled before it started" in xc["reason"]
        assert [n for n in lin.lineage(queue_dir, x.job_id)["nodes"] if n["relation"] == "descendant"] == []

    def test_a_producer_whose_bytes_gc_deleted_before_the_consumer_started_is_rejected(self, queue_dir, tmp_path):
        from gpu_queue import gc as gc_mod
        from gpu_queue.models import JobRequest
        from gpu_queue.queue import GPUQueue
        jt = self._job_types(); q = GPUQueue(queue_dir)
        m = queue_dir / "outputs" / "m"
        (tmp_path / "plate.png").write_bytes(b"PLATE")
        a = JobRequest(job_type="mesh", input_path=str(tmp_path / "plate.png"), output_dir=str(m), agent_id="lane-a")
        q.submit(a); assert q.run_one(jt) is True
        t0 = time.time()
        rows = gc_mod.scan(queue_dir, jt, now=t0 + 60, ttl_days={"intermediate": 0.0})
        assert {r["name"]: r for r in rows}["m"]["candidate"]
        cand = gc_mod.write_candidates(queue_dir, rows, now=t0 + 60)
        summary = gc_mod.apply(queue_dir, epoch=cand["epoch"], owner="ops", now=t0 + 60 + 73 * 3600, job_types=jt)
        assert summary["holds"] == [] and not m.exists()          # the receipt is the real collector's, bytes gone
        receipt = json.loads((queue_dir / "gc-receipts" / cand["epoch"] / "m.json").read_text())
        deleted_at = receipt["deleted_at"]                        # the collector stamps the wall clock, not the policy clock
        assert deleted_at is not None
        # the rerun writes the same bytes, so digests alone cannot tell A from A2: only the deletion time can
        a2 = job(queue_dir, job_type="mesh", agent="lane-a2", input_path=str(tmp_path / "plate.png"), input_bytes=b"PLATE",
                 outputs={"mesh.glb": b"MESH"}, name="m", finished=deleted_at + 7200)
        c = job(queue_dir, job_type="render", agent="lane-c", input_path=str(m / "mesh.glb"), input_bytes=b"MESH",
                outputs={"view.png": b"V"}, name="r", finished=deleted_at + 7300)

        g = lin.lineage(queue_dir, c)
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] == a2 and inp["ambiguous"] is False
        ac = [cnd for cnd in inp["candidates"] if cnd["job_id"] == a.job_id][0]
        assert ac["accepted"] is False and "deleted" in ac["reason"] and "before the consumer started" in ac["reason"]
        assert all(n["job_id"] != a.job_id for n in g["nodes"])          # a rejected candidate is not an ancestor
        assert lin.lineage(queue_dir, a.job_id)["subject"]["deleted_at"] == deleted_at


class TestSharedRecords:
    """Lineage reads records through the shared loader: deterministic deletions, untrusted torn records, resolved duplicates."""

    @pytest.mark.parametrize("last", ["first-collection", "second-collection"])
    def test_a_directory_collected_twice_gives_one_answer_whichever_receipt_loads_last(self, queue_dir, tmp_path, monkeypatch, last):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        m = queue_dir / "outputs" / "m"
        a = rq.run("mesh", tmp_path / "in.png", m, agent="lane-a")
        r1 = rq.collect("m")
        a2 = rq.run("mesh", tmp_path / "in.png", m, agent="lane-a2")          # same bytes: only deletion time can separate A from A2
        c = rq.run("render", m / "mesh.glb", queue_dir / "outputs" / "r", agent="lane-c")
        r2 = rq.collect("m")                                                    # re-lists A with a later deleted_at
        last_epoch = r1["epoch"] if last == "first-collection" else r2["epoch"]
        real_glob = Path.glob
        def ordered(self, pattern):
            found = list(real_glob(self, pattern))
            return iter(sorted(found, key=lambda p: (last_epoch in str(p), str(p))))   # the chosen receipt loads last, by identity not by chance
        monkeypatch.setattr(Path, "glob", ordered)

        g = lin.lineage(queue_dir, c)
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] == a2 and inp["ambiguous"] is False
        ac = [cnd for cnd in inp["candidates"] if cnd["job_id"] == a][0]
        assert ac["accepted"] is False and "deleted" in ac["reason"] and r1["epoch"] in ac["reason"]
        na = lin.lineage(queue_dir, a)["subject"]
        assert na["deleted_by_epoch"] == r1["epoch"] and na["deleted_at"] == r1["deleted_at"]
        assert sorted(na["deleted_by_epochs"]) == sorted([r1["epoch"], r2["epoch"]])

    def test_a_deleted_node_renders_its_deletion_once(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        a = rq.run("mesh", tmp_path / "in.png", queue_dir / "outputs" / "m", agent="lane-a")
        rq.collect("m")
        lines = [l for l in lin.render_text(lin.lineage(queue_dir, a)).splitlines() if l.strip().startswith(("deleted by", "deletion "))]
        assert len(lines) == 1 and "deleted by gc epoch" in lines[0]

    def test_a_partial_deletion_still_renders_when_the_record_is_also_torn_across_dirs(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        (queue_dir / "gc-receipts" / "e1").mkdir(parents=True)
        (queue_dir / "gc-receipts" / "e1" / "skull-mesh.json").write_text(json.dumps({
            "schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e1", "name": "skull-mesh", "path": str(queue_dir / "outputs" / "skull-mesh"),
            "job_ids": [ids["a"]], "deleted_at": None, "partial": True}))
        shutil.copytree(queue_dir / "done" / ids["a"], queue_dir / "running" / ids["a"])
        text = lin.render_text(lin.lineage(queue_dir, ids["a"]))
        assert "deletion partial (gc epoch e1)" in text and "also found under running/" in text

    def test_a_torn_status_file_the_real_writer_can_leave_is_flagged_and_not_credited(self, queue_dir):
        ids = chain(queue_dir)
        (queue_dir / "done" / ids["a"] / "status.json").write_bytes(b'{"status": "done", "fini')   # status.json is the file written non-atomically
        inp = lin.lineage(queue_dir, ids["b"])["subject"]["inputs"][0]
        ac = [cnd for cnd in inp["candidates"] if cnd["job_id"] == ids["a"]][0]
        assert ac["accepted"] is False and "status.json" in ac["reason"]

    def test_a_torn_file_that_is_not_utf8_is_unreadable_not_a_crash(self, queue_dir):
        ids = chain(queue_dir)
        (queue_dir / "done" / ids["a"] / "receipt.json").write_bytes(b'{"x": "\xe2\x82')
        assert lin.lineage(queue_dir, ids["a"])["subject"]["record_unreadable"] == ["receipt.json"]

    def test_a_producer_with_a_torn_record_file_is_not_credited(self, queue_dir):
        ids = chain(queue_dir)
        (queue_dir / "done" / ids["a"] / "receipt.json").write_text("{")     # torn write: exists, does not parse
        g = lin.lineage(queue_dir, ids["b"])
        inp = g["subject"]["inputs"][0]
        assert inp["producer"] is None
        ac = [cnd for cnd in inp["candidates"] if cnd["job_id"] == ids["a"]][0]
        assert ac["accepted"] is False and "unreadable" in ac["reason"] and "receipt.json" in ac["reason"]
        assert lin.lineage(queue_dir, ids["a"])["subject"]["record_unreadable"] == ["receipt.json"]
        assert "unreadable" in lin.render_text(lin.lineage(queue_dir, ids["a"]))

    def test_a_job_torn_between_two_state_dirs_is_read_from_the_furthest_copy(self, queue_dir):
        import shutil
        ids = chain(queue_dir)
        shutil.copytree(queue_dir / "done" / ids["a"], queue_dir / "running" / ids["a"])   # a move that never finished
        st = json.loads((queue_dir / "running" / ids["a"] / "status.json").read_text()); st["status"] = "running"; st["finished_at"] = None
        (queue_dir / "running" / ids["a"] / "status.json").write_text(json.dumps(st))
        g = lin.lineage(queue_dir, ids["b"])
        assert g["subject"]["inputs"][0]["producer"] == ids["a"]
        na = lin.lineage(queue_dir, ids["a"])["subject"]
        assert na["status"] == "done" and na["duplicate_records"] == ["running"]

    def test_a_job_the_real_worker_records_as_failed_is_shown_as_failed_and_stays_eligible(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        f = rq.run_failing(tmp_path / "in.png", queue_dir / "outputs" / "broken", agent="lane-f")
        n = lin.lineage(queue_dir, f)["subject"]
        assert n["status"] == "failed" and n["exit_code"] == 3 and n["never_started"] is False and n["record_unreadable"] == []
        assert "failed" in lin.render_text(lin.lineage(queue_dir, f))

    def test_a_producer_the_worker_nested_inside_a_husk_is_still_found(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        from gpu_queue.models import JobRequest
        m = queue_dir / "outputs" / "m"
        a = JobRequest(job_type="mesh", input_path=str(tmp_path / "in.png"), output_dir=str(m), agent_id="lane-a")
        (queue_dir / "done" / a.job_id).mkdir()
        rq.q.submit(a); assert rq.q.run_one(rq.job_types) is True
        assert (queue_dir / "done" / a.job_id / a.job_id / "status.json").is_file()
        b = rq.run("render", m / "mesh.glb", queue_dir / "outputs" / "r", agent="lane-b")
        g = lin.lineage(queue_dir, b)
        assert g["subject"]["inputs"][0]["producer"] == a.job_id
        na = lin.lineage(queue_dir, a.job_id)["subject"]
        assert na["status"] == "done" and na["nested_record"] is True
        assert "nested" in lin.render_text(lin.lineage(queue_dir, a.job_id))
