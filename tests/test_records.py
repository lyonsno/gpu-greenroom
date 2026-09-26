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

    def test_two_terminal_copies_resolve_by_lifecycle_order_and_report_the_other(self, queue_dir, monkeypatch):
        _job_dir(queue_dir, "failed", "j3", {"status": "failed", "started_at": 10.0, "finished_at": 20.0})
        _job_dir(queue_dir, "done", "j3", {"status": "done", "started_at": 10.0, "finished_at": 21.0})
        for order in (rec.STATE_DIRS, tuple(reversed(rec.STATE_DIRS))):
            monkeypatch.setattr(rec, "STATE_DIRS", order)
            r = rec.load_job_records(queue_dir)["j3"]
            assert r.state_dir == "done" and r.duplicates == ["failed"]     # done outranks failed outranks cancelled, whatever the walk

    def test_a_record_file_that_is_not_utf8_is_unreadable_not_a_crash(self, queue_dir):
        d = _job_dir(queue_dir, "done", "j4", {"status": "done", "started_at": 10.0, "finished_at": 20.0})
        (d / "receipt.json").write_bytes(b'{"x": "\xe2\x82')
        r = rec.load_job_records(queue_dir)["j4"]
        assert r.unreadable == ["receipt.json"] and r.finished_at == 20.0

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


class TestTornMovesTheWorkerLeaves:
    """shutil.move into a destination that already exists nests the job directory inside it (seen on the live queue, four jobs)."""

    def test_a_record_nested_inside_a_pre_existing_destination_is_read_and_outranks_a_receipt_only_copy(self, queue_dir, tmp_path):
        import shutil
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        from gpu_queue.models import JobRequest
        req = JobRequest(job_type="mesh", input_path=str(tmp_path / "in.png"), output_dir=str(queue_dir / "outputs" / "m"), agent_id="lane-a")
        (queue_dir / "done" / req.job_id).mkdir()                    # a husk already sits where the worker will move the job
        rq.q.submit(req)
        assert rq.q.run_one(rq.job_types) is True
        nested = queue_dir / "done" / req.job_id / req.job_id
        assert (nested / "status.json").is_file(), "the worker nested the record inside the husk; that is the shape this test is about"
        (queue_dir / "failed" / req.job_id).mkdir()
        shutil.copy(nested / "receipt.json", queue_dir / "failed" / req.job_id / "receipt.json")   # the receipt-only copy seen live
        r = rec.load_job_records(queue_dir)[req.job_id]
        assert r.state_dir == "done" and r.nested is True and r.record_dir == nested
        assert r.status == "done" and r.finished_at is not None and r.output_dir == str(queue_dir / "outputs" / "m")
        assert r.duplicates == ["failed"] and r.unreadable == []

    def test_an_empty_husk_never_outranks_a_complete_record(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        f = rq.run_failing(tmp_path / "in.png", queue_dir / "outputs" / "broken")
        (queue_dir / "done" / f).mkdir()                              # the empty husk seen live beside a complete failed record
        r = rec.load_job_records(queue_dir)[f]
        assert r.state_dir == "failed" and r.status == "failed" and r.finished_at is not None
        assert r.duplicates == ["done"] and r.nested is False


    def test_a_torn_record_still_outranks_an_empty_husk_and_stays_reported(self, queue_dir, tmp_path):
        rq = RealQueue(queue_dir)
        (tmp_path / "in.png").write_bytes(b"IN")
        f = rq.run_failing(tmp_path / "in.png", queue_dir / "outputs" / "broken")
        for name in ("request.json", "status.json", "receipt.json"):
            (queue_dir / "failed" / f / name).write_text("{")               # every record file torn (the worst the writer could leave)
        (queue_dir / "done" / f).mkdir()                                     # and a husk beside it
        r = rec.load_job_records(queue_dir)[f]
        assert r.state_dir == "failed" and sorted(r.unreadable) == ["receipt.json", "request.json", "status.json"] and r.duplicates == ["done"]

    def test_a_torn_file_in_the_preferred_copy_is_not_silenced_by_a_fuller_other_copy(self, queue_dir):
        _job_dir(queue_dir, "done", "j5", {"status": "done", "started_at": 10.0, "finished_at": 20.0})
        (queue_dir / "done" / "j5" / "receipt.json").write_text("{")
        d = _job_dir(queue_dir, "failed", "j5", {"status": "failed", "started_at": 10.0, "finished_at": 20.0})
        (d / "receipt.json").write_text(json.dumps({"job_id": "j5"}))
        r = rec.load_job_records(queue_dir)["j5"]
        assert r.state_dir == "done" and r.unreadable == ["receipt.json"]    # three files each; done outranks failed; the torn file is reported


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

    def test_unconfirmed_deletions_take_the_epoch_written_first_in_time_and_partial_names_a_partial_epoch(self, queue_dir):
        def receipt(epoch, written_at, partial):
            (queue_dir / "gc-receipts" / epoch).mkdir(parents=True)
            (queue_dir / "gc-receipts" / epoch / "m.json").write_text(json.dumps({
                "schema": "gpu-greenroom.gc-receipt.v1", "epoch": epoch, "name": "m", "path": str(queue_dir / "outputs" / "m"),
                "job_ids": ["a"], "deleted_at": None, "partial": partial, "written_at": written_at}))
        receipt("ffff", 100.0, True)      # the partial one, written first in time, lexically last
        receipt("0000", 200.0, False)     # unconfirmed, written later, lexically first
        receipts, _ = rec.load_gc_receipts(queue_dir)
        for order in (receipts, list(reversed(receipts))):
            d = rec.deletions_by_job(order)["a"]
            assert d.deletion == "partial" and d.epoch == "ffff" and d.deleted_at is None
            assert d.epochs == ["ffff", "0000"]                       # in the order the receipts were written

    def test_a_torn_receipt_is_reported_as_unreadable_and_skipped(self, queue_dir):
        (queue_dir / "gc-receipts" / "e1").mkdir(parents=True)
        (queue_dir / "gc-receipts" / "e1" / "m.json").write_text("{\"schema\": ")
        (queue_dir / "gc-receipts" / "e1" / "_summary.json").write_text("{}")
        receipts, unreadable = rec.load_gc_receipts(queue_dir)
        assert receipts == [] and unreadable == [str(queue_dir / "gc-receipts" / "e1" / "m.json")]
