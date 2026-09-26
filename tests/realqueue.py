"""The real writers as a fixture factory: submit, cancel, run and collect through GPUQueue and gc.

Tests that assert a retention or lineage rule build their records here, not by hand, so a rule
can only be tested against shapes the queue and the collector actually write."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

FAILER = "import sys; sys.exit(3)"
WRITER = ("import pathlib, sys; out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
          "(out / 'mesh.glb').write_bytes(b'MESH')")
READER = ("import pathlib, sys; out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); "
          "(out / 'view.png').write_bytes(pathlib.Path(sys.argv[1]).read_bytes() + b'-VIEW')")


class RealQueue:
    def __init__(self, queue_dir: Path):
        from gpu_queue.queue import GPUQueue
        self.queue_dir = Path(queue_dir)
        self.q = GPUQueue(self.queue_dir)
        self.job_types = {
            "mesh": {"cmd": [sys.executable, "-c", WRITER, "{output_dir}"], "artifact_manifest": ["mesh.glb"], "output_class": "intermediate"},
            "render": {"cmd": [sys.executable, "-c", READER, "{input_path}", "{output_dir}"], "artifact_manifest": ["view.png"],
                       "output_class": "intermediate"},
            "broken": {"cmd": [sys.executable, "-c", FAILER], "output_class": "intermediate"},
        }

    def _request(self, job_type, input_path, output_dir, agent):
        from gpu_queue.models import JobRequest
        return JobRequest(job_type=job_type, input_path=str(input_path), output_dir=str(output_dir), agent_id=agent)

    def run(self, job_type: str, input_path, output_dir, agent: str = "lane") -> str:
        req = self._request(job_type, input_path, output_dir, agent)
        self.q.submit(req)
        assert self.q.run_one(self.job_types) is True
        return req.job_id

    def run_failing(self, input_path, output_dir, agent: str = "lane") -> str:
        """A job the worker records as failed (non-zero exit); its record lands in failed/."""
        req = self._request("broken", input_path, output_dir, agent)
        self.q.submit(req)
        assert self.q.run_one(self.job_types) is True
        assert (self.queue_dir / "failed" / req.job_id).is_dir()
        return req.job_id

    def cancel(self, job_type: str, input_path, output_dir, agent: str = "lane") -> str:
        req = self._request(job_type, input_path, output_dir, agent)
        self.q.submit(req)
        assert self.q.cancel(req.job_id)
        return req.job_id

    def collect(self, name: str, owner: str = "ops") -> dict:
        """Collect with the real collector (zero TTL for every class, so every eligible entry goes, not only <name>;
        policy clock past the grace window; the collector stamps deleted_at with the wall clock); return <name>'s receipt.
        Not covered here: a collection that fails partway (a partial receipt); tests that need one write it by hand."""
        from gpu_queue import gc
        t = time.time() + 60
        rows = gc.scan(self.queue_dir, self.job_types, now=t, ttl_days={"intermediate": 0.0, "witness": 0.0, "final": 0.0})
        assert {r["name"]: r for r in rows}[name]["candidate"], "the entry must be a candidate for the collector to touch it"
        cand = gc.write_candidates(self.queue_dir, rows, now=t)
        summary = gc.apply(self.queue_dir, epoch=cand["epoch"], owner=owner, now=t + 73 * 3600, job_types=self.job_types)
        assert name not in {h["name"] for h in summary["holds"]}, summary["holds"]
        return json.loads((self.queue_dir / "gc-receipts" / cand["epoch"] / f"{name}.json").read_text())
