"""Tests for list/get_job hardening, worker transition logging, injected_env receipts,
pause/resume event telemetry, and quick-route/pause efficiency telemetry."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue


def _run_cli(queue_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )


def test_list_jobs_and_get_job_tolerate_foreign_or_corrupt_status_json_and_support_limit(
    tmp_path: Path,
):
    queue = GPUQueue(tmp_path / "queue")

    req1 = JobRequest(job_type="echo", input_path="a", submitted_at=100.0, job_id="job000000001")
    req2 = JobRequest(job_type="echo", input_path="b", submitted_at=200.0, job_id="job000000002")
    req3 = JobRequest(job_type="echo", input_path="c", submitted_at=300.0, job_id="job000000003")
    for req in (req1, req2, req3):
        queue.submit(req)

    job_types = {"echo": [sys.executable, "-c", "pass"]}
    assert queue.run_one(job_types)
    assert queue.run_one(job_types)
    assert queue.run_one(job_types)

    # Plant a foreign supervisor status.json in failed/ for job000000001 (which already succeeded in done/)
    # and another standalone foreign status.json in failed/foreign00001, plus a corrupt status.json.
    dup_failed = queue.queue_dir / "failed" / "job000000001"
    dup_failed.mkdir(parents=True, exist_ok=True)
    (dup_failed / "status.json").write_text(
        json.dumps(
            {
                "schema": "external.job_supervisor_failure.v0",
                "terminal_class": "failed",
                "phase": "child_exit",
                "exit_code": -15,
            }
        )
    )

    foreign_dir = queue.queue_dir / "failed" / "foreign00001"
    foreign_dir.mkdir(parents=True, exist_ok=True)
    (foreign_dir / "status.json").write_text(
        json.dumps({"schema": "foreign.v0", "terminal_class": "failed"})
    )

    corrupt_dir = queue.queue_dir / "failed" / "corrupt00001"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "status.json").write_text("{not-valid-json")

    # list_jobs() must not crash on foreign/corrupt status.json files
    all_jobs = queue.list_jobs()
    assert [j.job_id for j in all_jobs] == ["job000000001", "job000000002", "job000000003"]

    # limit returns the N most recent jobs by submission/start time, and <=0 returns []
    limited = queue.list_jobs(limit=2)
    assert [j.job_id for j in limited] == ["job000000002", "job000000003"]
    assert queue.list_jobs(limit=0) == []

    # Even if a stale/foreign status.json in failed/job000000001 parses as JobState(FAILED),
    # list_jobs(JobStatus.FAILED) must not return job000000001 because done/job000000001 has higher completeness.
    (dup_failed / "status.json").write_text(
        json.dumps(
            {
                "job_id": "job000000001",
                "status": "failed",
                "job_type": "echo",
                "input_path": "a",
                "output_dir": "",
                "params": {},
                "submitted_at": 100.0,
            }
        )
    )
    assert queue.list_jobs(JobStatus.FAILED) == []

    # get_job() must resolve job000000001 from done/ without crashing on the foreign husk in failed/
    resolved = queue.get_job("job000000001")
    assert resolved is not None
    assert resolved.status == JobStatus.DONE
    assert queue.get_job("foreign00001") is None

    # CLI list and list --limit must succeed
    cli_res = _run_cli(queue.queue_dir, "list", "--limit", "2")
    assert cli_res.returncode == 0, cli_res.stderr
    assert "job000000002" in cli_res.stdout
    assert "job000000003" in cli_res.stdout
    assert "job000000001" not in cli_res.stdout


def test_run_one_logs_start_and_end_and_records_injected_env_and_dispatch_telemetry(
    tmp_path: Path, capsys
):
    queue = GPUQueue(tmp_path / "queue")
    queue.configure_dispatch(
        {
            "schema": "gpu-greenroom.dispatch-policy.v4",
            "mode": "two-class",
            "quick_job_types": {},
            "quick_routes": {"route/quick-smoke": "verified sub-10s smoke"},
            "producer_hints": {"max_seconds": 30},
        },
        owner="test-operator",
    )

    # Enqueue two older normal jobs and one newer quick job that jumps ahead of them
    norm1 = JobRequest(
        job_type="command",
        input_path="",
        command_argv=[sys.executable, "-c", "pass"],
        command_cwd=str(tmp_path),
        route_identity="route/heavy-1",
        submitted_at=100.0,
        job_id="norm00000001",
    )
    norm2 = JobRequest(
        job_type="command",
        input_path="",
        command_argv=[sys.executable, "-c", "pass"],
        command_cwd=str(tmp_path),
        route_identity="route/heavy-2",
        submitted_at=101.0,
        job_id="norm00000002",
    )
    quick1 = JobRequest(
        job_type="command",
        input_path="",
        command_argv=[sys.executable, "-c", "pass"],
        command_cwd=str(tmp_path),
        route_identity="route/quick-smoke",
        submitted_at=102.0,
        job_id="quick0000001",
    )
    queue.submit(norm1)
    queue.submit(norm2)
    queue.submit(quick1)

    capsys.readouterr()
    assert queue.run_one({})
    captured = capsys.readouterr().out
    assert "[job-start] quick0000001 class=quick" in captured
    assert "route=route/quick-smoke" in captured
    assert "[job-end] quick0000001 status=done" in captured

    receipt = json.loads((queue.queue_dir / "done" / "quick0000001" / "receipt.json").read_text())
    assert receipt["injected_env"]["GREENROOM_JOB_ID"] == "quick0000001"
    assert receipt["injected_env"]["GPU_GREENROOM_JOB_ID"] == "quick0000001"
    assert receipt["injected_env"]["GPU_GREENROOM_QUEUE_DIR"] == str(queue.queue_dir.resolve())
    assert isinstance(receipt["queue_wait_seconds"], float) and receipt["queue_wait_seconds"] >= 0.0
    assert isinstance(receipt["execution_seconds"], float) and receipt["execution_seconds"] >= 0.0

    disp = receipt["dispatch"]
    assert disp["service_class"] == "quick"
    assert disp["pending_count_at_claim"] == 2
    assert disp["bypassed_pending_normal_count"] == 2
    assert disp["bypassed_pending_normal_job_ids"] == ["norm00000001", "norm00000002"]


def test_pause_and_resume_emit_events_and_record_paused_seconds(tmp_path: Path):
    queue = GPUQueue(tmp_path / "queue")
    pause_ack = queue.pause(owner="operator-test", contention_class="manual-check")
    assert pause_ack["effective_paused"] is True
    time.sleep(0.02)
    resume_ack = queue.resume(owner="operator-test", epoch=pause_ack["epoch"])
    assert resume_ack["effective_paused"] is False
    assert isinstance(resume_ack.get("paused_seconds"), float)
    assert resume_ack["paused_seconds"] >= 0.01

    events = [
        json.loads(p.read_text())
        for p in sorted(queue.events_dir.glob("*.json"))
    ]
    kinds = [e["kind"] for e in events]
    assert "queue_paused" in kinds
    assert "queue_resumed" in kinds
    resumed_ev = next(e for e in events if e["kind"] == "queue_resumed")
    assert resumed_ev["payload"]["epoch"] == pause_ack["epoch"]
    assert resumed_ev["payload"]["paused_seconds"] >= 0.01

    # Also verify admission_control.transition() emits queue_paused / queue_resumed with paused_seconds
    from gpu_queue import admission_control
    from gpu_queue.telemetry import summarize_telemetry

    ac_pause = admission_control.transition(
        queue, action="pause", request_id="req_pause_1", owner="console-op"
    )
    assert ac_pause["effective_paused"] is True
    time.sleep(0.01)
    ac_resume = admission_control.transition(
        queue,
        action="resume",
        request_id="req_resume_1",
        owner="console-op",
        epoch=ac_pause["epoch"],
    )
    assert ac_resume["effective_paused"] is False
    assert isinstance(ac_resume.get("paused_seconds"), float) and ac_resume["paused_seconds"] >= 0.005

    rep = summarize_telemetry(queue.queue_dir, since_hours=1.0)
    # 2 pauses and 2 resumes total, deduplicated across events/ and operator-actions/
    assert rep["pause_and_checkpoint_efficiency"]["queue_pause_events_count"] == 2
    assert rep["pause_and_checkpoint_efficiency"]["queue_resume_events_count"] == 2


def test_telemetry_summarizes_quick_wait_attribution_queue_jumps_false_normals_and_blockers(
    tmp_path: Path,
):
    from gpu_queue.telemetry import summarize_telemetry

    queue = GPUQueue(tmp_path / "queue")
    queue.configure_dispatch(
        {
            "schema": "gpu-greenroom.dispatch-policy.v4",
            "mode": "two-class",
            "quick_job_types": {},
            "quick_routes": {"route/quick-check": "verified quick"},
            "producer_hints": {"max_seconds": 30},
        },
        owner="test-operator",
    )

    def _write_synthetic_completed_job(
        job_id: str,
        *,
        service_class: str,
        route: str,
        submitted_at: float,
        started_at: float,
        finished_at: float,
        cooperative_checkpoint: bool = False,
        basis: str | None = None,
    ):
        d = queue.queue_dir / "done" / job_id
        d.mkdir(parents=True, exist_ok=True)
        req = {
            "job_id": job_id,
            "job_type": "command",
            "input_path": "",
            "output_dir": str(queue.queue_dir / "outputs" / job_id),
            "params": {},
            "agent_id": "agent-a",
            "route_identity": route,
            "cooperative_checkpoint": cooperative_checkpoint,
            "submitted_at": submitted_at,
        }
        disp = {
            "mode": "two-class",
            "service_class": service_class,
            "policy_epoch": "epoch1",
            "eligibility_basis": basis,
        }
        st = {
            "job_id": job_id,
            "status": "done",
            "job_type": "command",
            "input_path": "",
            "output_dir": req["output_dir"],
            "params": {},
            "submitted_at": submitted_at,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": 0,
            "dispatch": disp,
        }
        rc = {
            **st,
            "requested_route": route,
            "effective_route": route,
        }
        (d / "request.json").write_text(json.dumps(req))
        (d / "status.json").write_text(json.dumps(st))
        (d / "receipt.json").write_text(json.dumps(rc))

    # 1. Long non-cooperative normal job running from t=100 to t=200 (100s)
    _write_synthetic_completed_job(
        "norm_blocker",
        service_class="normal",
        route="route/heavy-sim",
        submitted_at=100.0,
        started_at=100.0,
        finished_at=200.0,
        cooperative_checkpoint=False,
    )
    # 2. Queued normal job submitted at t=110, runs AFTER quick1 from t=210 to t=510 (300s)
    _write_synthetic_completed_job(
        "norm_queued",
        service_class="normal",
        route="route/heavy-sim",
        submitted_at=110.0,
        started_at=210.0,
        finished_at=510.0,
        cooperative_checkpoint=False,
    )
    # 3. Quick job submitted at t=120 while norm_blocker is running; jumps norm_queued and runs t=200..210 (wait=80s, saves 300s!)
    _write_synthetic_completed_job(
        "quick_jumper",
        service_class="quick",
        route="route/quick-check",
        submitted_at=120.0,
        started_at=200.0,
        finished_at=210.0,
        basis="verified quick",
    )
    # 4. Two "false normal" jobs (route/fast-unlabeled) that ran in <=10s as normal but waited >60s
    _write_synthetic_completed_job(
        "fast_norm_1",
        service_class="normal",
        route="route/fast-unlabeled",
        submitted_at=130.0,
        started_at=510.0,
        finished_at=518.0,
    )
    _write_synthetic_completed_job(
        "fast_norm_2",
        service_class="normal",
        route="route/fast-unlabeled",
        submitted_at=140.0,
        started_at=518.0,
        finished_at=525.0,
    )

    report = summarize_telemetry(queue.queue_dir)
    assert report["schema"] == "gpu-greenroom.telemetry.v1"
    assert report["by_service_class"]["quick"]["count"] == 1
    assert report["by_service_class"]["normal"]["count"] == 4

    qe = report["quick_efficiency"]
    assert qe["queue_jumping_quick_jobs"] == 1
    assert qe["total_bypassed_normal_jobs"] == 1
    assert qe["estimated_wait_saved_seconds"] == 300.0
    assert qe["wait_attribution"]["blocked_by_in_flight_job"] == 1

    fn = report["unlabeled_quick_candidates"]
    assert fn["fast_normal_done_count"] == 2
    assert fn["waited_gt_60s_count"] == 2
    # Both fast_norm_1 (submitted at 130) and fast_norm_2 (submitted at 140) waited while
    # norm_queued (started at 210, finished at 510, 300s) ran between their submission and start.
    assert fn["reducible_by_quick_bypass_seconds"] == 600.0
    candidate_routes = [r["route"] for r in fn["candidate_routes"]]
    assert "route/fast-unlabeled" in candidate_routes

    pc = report["pause_and_checkpoint_efficiency"]
    assert pc["in_flight_blockers_of_quick_jobs"]["non_cooperative_blocker_count"] == 1
    assert pc["in_flight_blockers_of_quick_jobs"]["potential_wait_reducible_by_cooperative_pause_seconds"] == 80.0

    # Verify CLI telemetry command works in both text and --json modes
    cli_json = _run_cli(queue.queue_dir, "telemetry", "--json")
    assert cli_json.returncode == 0, cli_json.stderr
    parsed = json.loads(cli_json.stdout)
    assert parsed["quick_efficiency"]["estimated_wait_saved_seconds"] == 300.0

    cli_text = _run_cli(queue.queue_dir, "telemetry")
    assert cli_text.returncode == 0, cli_text.stderr
    assert "route/fast-unlabeled" in cli_text.stdout
