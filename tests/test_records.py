"""The shared record reader: one place that knows the shapes the queue and the collector write."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from gpu_queue import records as rec
from tests.realqueue import RealQueue


@pytest.fixture
def queue_dir(tmp_path):
    q = tmp_path / "queue"
    for sub in ("pending", "running", "done", "failed", "cancelled", "outputs"):
        (q / sub).mkdir(parents=True)
    return q


def _job_dir(queue_dir, state, job_id, status_doc):
    d = queue_dir / state / job_id
    d.mkdir(parents=True)
    (d / "request.json").write_text(json.dumps({"job_id": job_id, "job_type": "t", "output_dir": str(queue_dir / "outputs" / job_id)}))
    (d / "status.json").write_text(json.dumps(status_doc))
    return d


class TestJobRecords:
    def test_a_job_id_in_two_state_dirs_resolves_to_the_furthest_copy_whatever_the_walk_order(self, queue_dir, monkeypatch):
        _job_dir(queue_dir, "running", "j1", {"status": "running", "started_at": 10.0, "finished_at": None})
        _job_dir(queue_dir, "done", "j1", {"status": "done", "started_at": 10.0, "finished_at": 20.0})
        for order in (rec.STATE_DIRS, tuple(reversed(rec.STATE_DIRS))):
            monkeypatch.setattr(rec, "STATE_DIRS", order)
            r = rec.load_job_records(queue_dir)["j1"]
            assert r.state_dir == "done" and r.status == "done" and r.finished_at == 20.0
            assert r.duplicates == ["running"]

    def test_a_record_file_that_exists_but_does_not_parse_is_flagged_not_trusted(self, queue_dir):
        d = _job_dir(queue_dir, "done", "j2", {"status": "done", "started_at": 10.0, "finished_at": 20.0})
        (d / "receipt.json").write_text("{")           # torn write
        r = rec.load_job_records(queue_dir)["j2"]
        assert r.unreadable == ["receipt.json"] and r.readable is False
        assert r.status == "done" and r.finished_at == 20.0    # what did parse is still reported

    def test_the_real_cancel_shape_is_never_started(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        x = rq.cancel("mesh", tmp_path / "in.png", queue_dir / "outputs" / "m")
        r = rec.load_job_records(queue_dir)[x]
        assert r.state_dir == "cancelled" and r.finished_at is not None and r.started_at is None
        assert r.never_started is True


class TestGcReceipts:
    def test_deletions_merge_to_the_earliest_confirmed_and_a_later_receipt_never_downgrades_them(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        m = queue_dir / "outputs" / "m"
        a = rq.run("mesh", tmp_path / "in.png", m, agent="lane-a")
        r1 = rq.collect("m")
        a2 = rq.run("mesh", tmp_path / "in.png", m, agent="lane-a2")
        r2 = rq.collect("m")
        assert a in r2["job_ids"] and a2 in r2["job_ids"]        # the collector re-lists every job of the directory
        assert r2["deleted_at"] > r1["deleted_at"]
        # a torn third epoch that lists A again without confirming anything
        third = queue_dir / "gc-receipts" / "e3"
        third.mkdir()
        (third / "m.json").write_text(json.dumps({"schema": "gpu-greenroom.gc-receipt.v1", "epoch": "e3", "name": "m",
                                                  "path": str(m), "job_ids": [a], "deleted_at": None, "partial": True}))
        receipts, unreadable = rec.load_gc_receipts(queue_dir)
        assert unreadable == []
        for _ in range(5):
            shuffled = list(receipts)
            random.shuffle(shuffled)
            d = rec.deletions_by_job(shuffled)
            assert d[a].deleted_at == r1["deleted_at"] and d[a].epoch == r1["epoch"] and d[a].deletion == "confirmed"
            assert sorted(d[a].epochs) == sorted([r1["epoch"], r2["epoch"], "e3"])
            assert d[a2].deleted_at == r2["deleted_at"] and d[a2].epoch == r2["epoch"]

    def test_a_torn_receipt_is_reported_as_unreadable_and_skipped(self, queue_dir):
        (queue_dir / "gc-receipts" / "e1").mkdir(parents=True)
        (queue_dir / "gc-receipts" / "e1" / "m.json").write_text("{\"schema\": ")
        (queue_dir / "gc-receipts" / "e1" / "_summary.json").write_text("{}")
        receipts, unreadable = rec.load_gc_receipts(queue_dir)
        assert receipts == [] and unreadable == [str(queue_dir / "gc-receipts" / "e1" / "m.json")]
