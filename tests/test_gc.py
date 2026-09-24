"""Fail-first contract for `gpu-greenroom gc` and `retain`.

Retention policy (operator-approved 2026-09-24): every output has a class and
an owner; TTL by class (intermediate 30 d, witness 60 d, final 180 d);
unclassified is reported, never collected; pins are declared, never inferred;
collection is two-phase (dry-run writes an epoch-bound candidate list, apply
deletes only that list after a grace window, re-checking every row); every
deletion writes a receipt before the bytes go; nothing outside the queue's own
outputs/ is ever touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from gpu_queue import gc as gc_mod

DAY = 86400.0
NOW = 1_800_000_000.0  # fixed "now" so ages are deterministic


@pytest.fixture
def queue_dir(tmp_path):
    q = tmp_path / "queue"
    for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
        (q / sub).mkdir(parents=True)
    (q / "job_types.json").write_text(json.dumps({
        "trace": {"cmd": ["true"], "output_class": "intermediate"},
        "render": {"cmd": ["true"], "output_class": "witness"},
        "mesh": {"cmd": ["true"], "output_class": "final"},
        "legacy": {"cmd": ["true"]},
    }))
    return q


def make_output(queue_dir: Path, name: str, *, job_type: str, agent: str | None, finished_days_ago: float,
                status: str = "done", size: int = 4096, manifest: bool = False, now: float = NOW) -> Path:
    out = queue_dir / "outputs" / name
    out.mkdir(parents=True)
    (out / "blob.bin").write_bytes(b"x" * size)
    job_id = uuid.uuid4().hex[:12]
    jd = queue_dir / status / job_id
    jd.mkdir(parents=True)
    finished = now - finished_days_ago * DAY
    (jd / "request.json").write_text(json.dumps({
        "job_type": job_type, "input_path": "/in.png", "output_dir": str(out),
        "params": {}, "agent_id": agent, "job_id": job_id, "submitted_at": finished - 60,
    }))
    (jd / "status.json").write_text(json.dumps({
        "job_id": job_id, "status": status, "job_type": job_type, "input_path": "/in.png",
        "output_dir": str(out), "submitted_at": finished - 60, "started_at": finished - 30, "finished_at": finished,
    }))
    receipt = {"job_id": job_id, "job_type": job_type, "status": status, "output_dir": str(out),
               "finished_at": finished, "artifact_manifest": None, "input_artifact": None}
    if manifest:
        receipt["artifact_manifest"] = [{"path": "blob.bin", "sha256": "ab" * 32, "size_bytes": size}]
    (jd / "receipt.json").write_text(json.dumps(receipt))
    return out


def load_job_types(queue_dir):
    return json.loads((queue_dir / "job_types.json").read_text())


class TestScan:
    def test_ttl_by_class_and_unclassified_is_never_a_candidate(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        make_output(queue_dir, "young-trace", job_type="trace", agent="lane-a", finished_days_ago=10)
        make_output(queue_dir, "old-render", job_type="render", agent="lane-b", finished_days_ago=70)
        make_output(queue_dir, "mid-render", job_type="render", agent="lane-b", finished_days_ago=45)
        make_output(queue_dir, "old-mesh", job_type="mesh", agent="lane-c", finished_days_ago=200)
        make_output(queue_dir, "kept-mesh", job_type="mesh", agent="lane-c", finished_days_ago=100)
        make_output(queue_dir, "old-legacy", job_type="legacy", agent=None, finished_days_ago=400)

        rows = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=NOW)}

        assert rows["old-trace"]["candidate"] and rows["old-trace"]["output_class"] == "intermediate"
        assert not rows["young-trace"]["candidate"]
        assert rows["old-render"]["candidate"] and rows["old-render"]["output_class"] == "witness"
        assert not rows["mid-render"]["candidate"]
        assert rows["old-mesh"]["candidate"] and rows["old-mesh"]["output_class"] == "final"
        assert not rows["kept-mesh"]["candidate"]
        assert rows["old-legacy"]["output_class"] == "unclassified"
        assert not rows["old-legacy"]["candidate"]
        assert rows["old-legacy"]["reason"] == "unclassified"
        assert rows["old-trace"]["owner"] == "lane-a"
        assert rows["old-legacy"]["owner"] is None
        assert rows["old-trace"]["size_bytes"] >= 4096

    def test_pin_blocks_candidacy_and_is_declared_not_inferred(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        gc_mod.RetentionPins(queue_dir).pin("old-trace", owner="lane-a", reason="raw checkpoint custody")

        row = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=NOW)}["old-trace"]

        assert row["pinned"] is True
        assert not row["candidate"]
        assert row["reason"] == "pinned"
        pins = json.loads((queue_dir / "retention" / "pins.json").read_text())
        assert pins["pins"]["old-trace"]["reason"] == "raw checkpoint custody"

    def test_expired_pin_no_longer_protects(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        gc_mod.RetentionPins(queue_dir).pin("old-trace", owner="lane-a", reason="until review", until=NOW - 1)

        row = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=NOW)}["old-trace"]

        assert row["pinned"] is False
        assert row["candidate"]

    def test_output_referenced_by_pending_or_running_job_is_not_a_candidate(self, queue_dir):
        out = make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        pj = queue_dir / "pending" / "p1"
        pj.mkdir()
        (pj / "request.json").write_text(json.dumps({"job_type": "trace", "input_path": "/in.png", "output_dir": str(out / "again"), "params": {}}))

        row = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=NOW)}["old-trace"]

        assert row["active"] is True
        assert not row["candidate"]

    def test_dir_without_any_job_record_is_unclassified_with_mtime_age(self, queue_dir):
        stray = queue_dir / "outputs" / "stray"
        stray.mkdir()
        (stray / "x").write_text("x")

        row = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=time.time())}["stray"]

        assert row["output_class"] == "unclassified"
        assert row["job_ids"] == []
        assert row["age_days"] is not None and row["age_days"] < 1
        assert not row["candidate"]


class TestTwoPhaseApply:
    def _candidates(self, queue_dir, now=NOW, grace_hours=72.0):
        rows = gc_mod.scan(queue_dir, load_job_types(queue_dir), now=now)
        return gc_mod.write_candidates(queue_dir, rows, now=now, grace_hours=grace_hours)

    def test_apply_refuses_unknown_epoch_and_deletes_nothing(self, queue_dir):
        out = make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        self._candidates(queue_dir)

        with pytest.raises(gc_mod.GCRefused, match="epoch"):
            gc_mod.apply(queue_dir, epoch="not-the-epoch", owner="ops", now=NOW + 100 * DAY)
        assert out.exists()

    def test_apply_refuses_before_grace_elapses(self, queue_dir):
        out = make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        cand = self._candidates(queue_dir, grace_hours=72.0)

        with pytest.raises(gc_mod.GCRefused, match="grace"):
            gc_mod.apply(queue_dir, epoch=cand["epoch"], owner="ops", now=NOW + 71 * 3600)
        assert out.exists()

    def test_apply_deletes_only_listed_rows_with_receipt_before_bytes(self, queue_dir):
        old = make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45, manifest=True)
        young = make_output(queue_dir, "young-trace", job_type="trace", agent="lane-a", finished_days_ago=1)
        cand = self._candidates(queue_dir)
        # something that became a candidate only after the list was written must not be collected by this epoch
        later = make_output(queue_dir, "later-trace", job_type="trace", agent="lane-a", finished_days_ago=45)

        summary = gc_mod.apply(queue_dir, epoch=cand["epoch"], owner="ops", now=NOW + 4 * DAY)

        assert not old.exists()
        assert young.exists() and later.exists()
        assert summary["deleted"] == 1 and summary["held"] == 0
        receipt = json.loads((queue_dir / "gc-receipts" / cand["epoch"] / "old-trace.json").read_text())
        assert receipt["schema"] == "gpu-greenroom.gc-receipt.v1"
        assert receipt["output_class"] == "intermediate"
        assert receipt["owner"] == "lane-a"
        assert receipt["applied_by"] == "ops"
        assert receipt["size_bytes"] >= 4096
        assert receipt["job_ids"]
        assert receipt["artifact_manifest"][0]["sha256"] == "ab" * 32
        assert receipt["deleted_at"] >= receipt["written_at"]

    def test_apply_rechecks_pins_and_active_jobs_at_execution_time(self, queue_dir):
        pinned = make_output(queue_dir, "pinned-later", job_type="trace", agent="lane-a", finished_days_ago=45)
        active = make_output(queue_dir, "active-later", job_type="trace", agent="lane-a", finished_days_ago=45)
        cand = self._candidates(queue_dir)
        assert {r["name"] for r in cand["rows"] if r["candidate"]} == {"pinned-later", "active-later"}
        gc_mod.RetentionPins(queue_dir).pin("pinned-later", owner="lane-a", reason="objection during grace")
        pj = queue_dir / "pending" / "p2"
        pj.mkdir()
        (pj / "request.json").write_text(json.dumps({"job_type": "trace", "input_path": "/in.png", "output_dir": str(active), "params": {}}))

        summary = gc_mod.apply(queue_dir, epoch=cand["epoch"], owner="ops", now=NOW + 4 * DAY)

        assert pinned.exists() and active.exists()
        assert summary["deleted"] == 0 and summary["held"] == 2
        reasons = {h["name"]: h["reason"] for h in summary["holds"]}
        assert reasons["pinned-later"] == "pinned"
        assert reasons["active-later"] == "active"

    def test_apply_never_touches_a_path_outside_outputs(self, queue_dir, tmp_path):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        cand = self._candidates(queue_dir)
        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "precious").write_text("do not delete")
        path = queue_dir / "gc-candidates.json"
        doc = json.loads(path.read_text())
        doc["rows"][0]["path"] = str(victim)
        path.write_text(json.dumps(doc))

        with pytest.raises(gc_mod.GCRefused, match="outside"):
            gc_mod.apply(queue_dir, epoch=cand["epoch"], owner="ops", now=NOW + 4 * DAY)
        assert (victim / "precious").exists()

    def test_candidates_file_carries_epoch_totals_and_apply_deadline(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45)
        make_output(queue_dir, "old-legacy", job_type="legacy", agent=None, finished_days_ago=400)

        cand = self._candidates(queue_dir, grace_hours=72.0)

        doc = json.loads((queue_dir / "gc-candidates.json").read_text())
        assert doc["schema"] == "gpu-greenroom.gc-candidates.v1"
        assert doc["epoch"] == cand["epoch"]
        assert doc["apply_not_before"] == pytest.approx(NOW + 72 * 3600)
        assert doc["totals"]["candidate_count"] == 1
        assert doc["totals"]["unclassified_count"] == 1
        assert doc["totals"]["candidate_bytes"] >= 4096


def run_cli(*args, queue_dir):
    proc = subprocess.run([sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


class TestCLI:
    def test_submit_command_declares_output_class_and_scan_uses_it(self, queue_dir, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        rc, out, err = run_cli("submit-command", "--agent-id", "lane-x", "--repo-root", str(repo), "--cwd", str(repo),
                               "--route-identity", "demo", "--output-dir", str(queue_dir / "outputs" / "declared"),
                               "--output-class", "intermediate", "--", "true", queue_dir=queue_dir)
        assert rc == 0, err
        response = json.loads(out)
        assert response["output_class"] == "intermediate"
        request = json.loads((queue_dir / "pending" / response["job_id"] / "request.json").read_text())
        assert request["output_class"] == "intermediate"
        # move the pending record to done with an old finish so scan sees a terminal job
        job_dir = queue_dir / "pending" / response["job_id"]
        (queue_dir / "outputs" / "declared").mkdir(parents=True)
        status = json.loads((job_dir / "status.json").read_text())
        status.update({"status": "done", "finished_at": NOW - 45 * DAY})
        (job_dir / "status.json").write_text(json.dumps(status))
        job_dir.rename(queue_dir / "done" / response["job_id"])

        row = {r["name"]: r for r in gc_mod.scan(queue_dir, load_job_types(queue_dir), now=NOW)}["declared"]

        assert row["output_class"] == "intermediate" and row["class_source"] == "request"
        assert row["owner"] == "lane-x"
        assert row["candidate"]

    def test_submit_command_rejects_unknown_output_class(self, queue_dir, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        rc, out, err = run_cli("submit-command", "--agent-id", "lane-x", "--repo-root", str(repo), "--cwd", str(repo),
                               "--route-identity", "demo", "--output-class", "forever", "--", "true", queue_dir=queue_dir)
        assert rc == 2
        assert not any((queue_dir / "pending").iterdir())

    def test_gc_dry_run_writes_candidates_and_apply_requires_epoch(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45, now=time.time())
        rc, out, err = run_cli("gc", "--dry-run", "--json", queue_dir=queue_dir)
        assert rc == 0, err
        doc = json.loads(out)
        assert doc["schema"] == "gpu-greenroom.gc-candidates.v1"
        assert [c["name"] for c in doc["candidates"]] == ["old-trace"]
        assert (queue_dir / "gc-candidates.json").exists()

        rc, out, err = run_cli("gc", "--apply", "--owner", "ops", queue_dir=queue_dir)
        assert rc == 2
        rc, out, err = run_cli("gc", "--apply", "--owner", "ops", "--epoch", "nope", queue_dir=queue_dir)
        assert rc == 1 and "epoch" in err
        assert (queue_dir / "outputs" / "old-trace").exists()

    def test_retain_pins_and_lists(self, queue_dir):
        make_output(queue_dir, "old-trace", job_type="trace", agent="lane-a", finished_days_ago=45, now=time.time())
        rc, out, err = run_cli("retain", "old-trace", "--owner", "lane-a", "--reason", "live control", queue_dir=queue_dir)
        assert rc == 0, err
        rc, out, err = run_cli("retain", "--list", queue_dir=queue_dir)
        assert json.loads(out)["old-trace"]["reason"] == "live control"
        rc, out, err = run_cli("gc", "--dry-run", "--json", queue_dir=queue_dir)
        assert json.loads(out)["candidates"] == []
        rc, out, err = run_cli("retain", "old-trace", "--owner", "x", queue_dir=queue_dir)
        assert rc == 2
