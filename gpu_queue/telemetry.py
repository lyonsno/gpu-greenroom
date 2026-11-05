"""Queue wait-time, quick-route bypass efficiency, and pause/checkpoint telemetry."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import dispatch

TELEMETRY_SCHEMA = "gpu-greenroom.telemetry.v1"


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return round(sorted_vals[idx], 2)


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {
            "count": 0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    s = sorted(vals)
    return {
        "count": len(s),
        "min": round(s[0], 2),
        "p50": _pct(s, 50),
        "p90": _pct(s, 90),
        "p95": _pct(s, 95),
        "max": round(s[-1], 2),
        "mean": round(sum(s) / len(s), 2),
    }


def _load_json_safe(path: Path) -> dict | None:
    try:
        if not path.is_file():
            return None
        val = json.loads(path.read_text())
        return val if isinstance(val, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def summarize_telemetry(
    queue_dir: str | Path,
    *,
    since_hours: float | None = None,
    quick_threshold_seconds: float = 30.0,
) -> dict:
    """Compute wait-time, quick-route efficiency, false-normal, and pause telemetry."""
    root = Path(queue_dir).expanduser().resolve()
    now = time.time()
    cutoff = (now - since_hours * 3600.0) if since_hours is not None else None

    try:
        policy = dispatch.policy(root)
    except Exception:
        policy = {}
    configured_quick_routes = set((policy.get("quick_routes") or {}).keys())

    # Deduplicate jobs across state directories by completeness precedence
    state_dirs = ("running", "pending", "deferred", "done", "failed", "cancelled")
    candidates_by_id: dict[str, tuple[int, str, Path, dict, dict, dict]] = {}

    for state_name in state_dirs:
        sub_dir = root / state_name
        if not sub_dir.is_dir():
            continue
        for job_dir in sorted(sub_dir.iterdir()):
            if not job_dir.is_dir():
                continue
            st = _load_json_safe(job_dir / "status.json")
            req = _load_json_safe(job_dir / "request.json")
            rc = _load_json_safe(job_dir / "receipt.json")
            # Ignore foreign supervisor files that lack valid Greenroom job status
            if st is None or not isinstance(st.get("status"), str) or not isinstance(st.get("job_id"), str):
                continue
            score = int(req is not None) + int(st is not None) + int(rc is not None)
            prev = candidates_by_id.get(st["job_id"])
            if prev is None or score > prev[0]:
                candidates_by_id[st["job_id"]] = (
                    score,
                    state_name,
                    job_dir,
                    req or {},
                    st,
                    rc or {},
                )

    records: list[dict] = []
    committed_handoffs_count = 0
    performance_window_deferrals_count = 0
    external_lease_admissions_count = 0

    for job_id, (_score, state_name, job_dir, req, st, rc) in candidates_by_id.items():
        job_type = st.get("job_type") or req.get("job_type")
        if job_type == "external-lease-admission":
            external_lease_admissions_count += 1
            continue
        sub_at = st.get("submitted_at") or req.get("submitted_at")
        start_at = st.get("started_at") or rc.get("started_at")
        fin_at = st.get("finished_at") or rc.get("finished_at")
        anchor_ts = sub_at or start_at or fin_at
        if cutoff is not None and anchor_ts is not None and anchor_ts < cutoff:
            continue

        disp = rc.get("dispatch") or st.get("dispatch") or {}
        mode = disp.get("mode") or ("two-class" if disp.get("service_class") else "fifo")
        service_class = disp.get("service_class") or req.get("service_class") or "normal"
        basis = disp.get("eligibility_basis")

        route = (
            rc.get("requested_route")
            or req.get("route_identity")
            or st.get("job_type")
            or req.get("job_type")
            or "unknown"
        )
        wait_s = (
            max(0.0, float(start_at) - float(sub_at))
            if (sub_at is not None and start_at is not None)
            else None
        )
        exec_s = (
            max(0.0, float(fin_at) - float(start_at))
            if (start_at is not None and fin_at is not None)
            else None
        )

        handoff = _load_json_safe(job_dir / "checkpoint-handoff.json")
        if handoff and handoff.get("status") == "committed":
            committed_handoffs_count += 1
        deferrals_dir = job_dir / "performance-deferrals"
        if deferrals_dir.is_dir():
            performance_window_deferrals_count += sum(1 for _ in deferrals_dir.glob("*.json"))

        records.append(
            {
                "job_id": job_id,
                "state": state_name,
                "status": st.get("status", state_name),
                "mode": mode,
                "service_class": service_class,
                "basis": basis,
                "route": route,
                "agent_id": req.get("agent_id"),
                "submitted_at": float(sub_at) if sub_at is not None else None,
                "started_at": float(start_at) if start_at is not None else None,
                "finished_at": float(fin_at) if fin_at is not None else None,
                "wait_s": wait_s,
                "exec_s": exec_s,
                "cooperative_checkpoint": bool(req.get("cooperative_checkpoint")),
                "continuation_of": (req.get("params") or {}).get("continuation_of"),
                "performance_window": req.get("performance_window"),
                "runtime_hint": req.get("runtime_hint") or rc.get("runtime_hint"),
                "runtime_hint_comparison": rc.get("runtime_hint_comparison"),
                "dispatch": disp,
            }
        )

    # Filter to two-class records when two-class jobs exist; otherwise include all
    two_class_records = [r for r in records if r["mode"] == "two-class"]
    active_records = two_class_records if two_class_records else records

    by_service_class: dict[str, dict] = {}
    for sc in ("quick", "normal"):
        sc_recs = [r for r in active_records if r["service_class"] == sc]
        waits = [r["wait_s"] for r in sc_recs if r["wait_s"] is not None]
        execs = [r["exec_s"] for r in sc_recs if r["exec_s"] is not None and r["status"] == "done"]
        basis_counts = Counter(str(r["basis"]) for r in sc_recs if r["basis"])
        status_counts = Counter(str(r["status"]) for r in sc_recs)
        by_service_class[sc] = {
            "count": len(sc_recs),
            "status_counts": dict(status_counts),
            "wait_seconds": {
                **_dist(waits),
                "wait_gt_5s_count": sum(1 for w in waits if w > 5.0),
                "wait_gt_30s_count": sum(1 for w in waits if w > 30.0),
                "wait_gt_120s_count": sum(1 for w in waits if w > 120.0),
            },
            "execution_seconds": _dist(execs),
            "eligibility_basis_counts": dict(basis_counts),
        }

    # Quick route efficiency & wait attribution
    quick_started = [
        r
        for r in active_records
        if r["service_class"] == "quick"
        and r["submitted_at"] is not None
        and r["started_at"] is not None
    ]
    normal_started = [
        r
        for r in active_records
        if r["service_class"] == "normal"
        and r["submitted_at"] is not None
        and r["started_at"] is not None
    ]
    all_started = [
        r
        for r in active_records
        if r["submitted_at"] is not None and r["started_at"] is not None
    ]
    records_by_id = {r["job_id"]: r for r in active_records}

    queue_jumping_quick_jobs = 0
    total_bypassed_normal_jobs = 0
    estimated_wait_saved_seconds = 0.0

    wait_attribution = {
        "immediate_le_5s": 0,
        "blocked_by_in_flight_job": 0,
        "waited_behind_other_dispatches": 0,
        "paused_or_external_lease_block": 0,
    }

    non_cooperative_blocker_count = 0
    cooperative_blocker_count = 0
    potential_wait_reducible_seconds = 0.0
    blocker_route_stats: dict[str, dict] = defaultdict(
        lambda: {
            "route": "",
            "blocked_quick_count": 0,
            "total_blocked_seconds": 0.0,
            "cooperative_checkpoint_supported": False,
        }
    )

    for q in quick_started:
        q_sub = q["submitted_at"]
        q_start = q["started_at"]
        q_wait = q["wait_s"] or 0.0

        # Determine bypassed normal jobs (prefer recorded claim snapshot, fallback to interval reconstruction)
        recorded_bypassed_ids = (q["dispatch"] or {}).get("bypassed_pending_normal_job_ids")
        if isinstance(recorded_bypassed_ids, list):
            bypassed_jobs = [
                records_by_id[jid]
                for jid in recorded_bypassed_ids
                if jid in records_by_id
            ]
            bypassed_count = len(recorded_bypassed_ids)
        else:
            bypassed_jobs = [
                n
                for n in normal_started
                if n["submitted_at"] <= q_sub and n["started_at"] > q_start
            ]
            bypassed_count = len(bypassed_jobs)

        if bypassed_count > 0:
            queue_jumping_quick_jobs += 1
            total_bypassed_normal_jobs += bypassed_count
            saved_for_q = sum((n["exec_s"] or 0.0) for n in bypassed_jobs)
            estimated_wait_saved_seconds += saved_for_q

        # Attribute wait time
        in_flight = [
            o
            for o in all_started
            if o["job_id"] != q["job_id"]
            and o["started_at"] <= q_sub
            and (o["finished_at"] or o["started_at"]) > q_sub
        ]
        started_before_q = [
            o
            for o in all_started
            if o["job_id"] != q["job_id"] and q_sub < o["started_at"] < q_start
        ]

        if q_wait <= 5.0:
            wait_attribution["immediate_le_5s"] += 1
        elif started_before_q:
            wait_attribution["waited_behind_other_dispatches"] += 1
        elif in_flight:
            wait_attribution["blocked_by_in_flight_job"] += 1
        else:
            wait_attribution["paused_or_external_lease_block"] += 1

        if q_wait > 5.0 and in_flight:
            # Attribute to the primary in-flight blocker
            blocker = max(
                in_flight,
                key=lambda o: min(q_start, o["finished_at"] or q_start) - q_sub,
            )
            overlap_s = max(
                0.0, min(q_start, blocker["finished_at"] or q_start) - q_sub
            )
            b_route = blocker["route"]
            entry = blocker_route_stats[b_route]
            entry["route"] = b_route
            entry["blocked_quick_count"] += 1
            entry["total_blocked_seconds"] = round(
                entry["total_blocked_seconds"] + overlap_s, 2
            )
            entry["cooperative_checkpoint_supported"] = bool(
                blocker["cooperative_checkpoint"]
            )
            if blocker["cooperative_checkpoint"]:
                cooperative_blocker_count += 1
            else:
                non_cooperative_blocker_count += 1
                potential_wait_reducible_seconds += overlap_s

    # Runtime hint accuracy on quick jobs
    hinted_quick = [
        r
        for r in active_records
        if r["service_class"] == "quick"
        and (isinstance(r.get("runtime_hint_comparison"), dict) or isinstance(r.get("runtime_hint"), dict))
        and r["exec_s"] is not None
    ]
    within_expected = 0
    within_threshold = 0
    for r in hinted_quick:
        cmp_d = r.get("runtime_hint_comparison") or {}
        hint_d = r.get("runtime_hint") or {}
        exp_s = cmp_d.get("expected_seconds") or hint_d.get("expected_seconds")
        if exp_s is not None and r["exec_s"] <= float(exp_s):
            within_expected += 1
        if r["exec_s"] <= quick_threshold_seconds:
            within_threshold += 1

    quick_count = len(quick_started)
    quick_efficiency = {
        "total_quick_jobs": quick_count,
        "queue_jumping_quick_jobs": queue_jumping_quick_jobs,
        "queue_jumping_fraction": round(queue_jumping_quick_jobs / quick_count, 3)
        if quick_count
        else 0.0,
        "total_bypassed_normal_jobs": total_bypassed_normal_jobs,
        "estimated_wait_saved_seconds": round(estimated_wait_saved_seconds, 2),
        "mean_wait_saved_per_quick_job_seconds": round(
            estimated_wait_saved_seconds / quick_count, 2
        )
        if quick_count
        else 0.0,
        "mean_wait_saved_when_jumping_seconds": round(
            estimated_wait_saved_seconds / queue_jumping_quick_jobs, 2
        )
        if queue_jumping_quick_jobs
        else 0.0,
        "wait_attribution": wait_attribution,
        "runtime_hint_accuracy": {
            "evaluated_quick_jobs": len(hinted_quick),
            "within_expected_seconds_count": within_expected,
            "within_quick_threshold_count": within_threshold,
            "exceeded_quick_threshold_count": len(hinted_quick) - within_threshold,
        },
    }

    # Unlabeled quick candidates ("false normals")
    normal_done = [
        r
        for r in active_records
        if r["service_class"] == "normal"
        and r["status"] == "done"
        and r["exec_s"] is not None
    ]
    fast_normals = [r for r in normal_done if r["exec_s"] <= quick_threshold_seconds]
    fast_waited_gt_60 = [
        r["wait_s"]
        for r in fast_normals
        if r["wait_s"] is not None and r["wait_s"] > 60.0
    ]

    normal_by_route: dict[str, list[dict]] = defaultdict(list)
    for r in normal_done:
        normal_by_route[r["route"]].append(r)

    candidate_routes: list[dict] = []
    for route, r_list in normal_by_route.items():
        if route in configured_quick_routes or len(r_list) < 2:
            continue
        execs = sorted(r["exec_s"] for r in r_list if r["exec_s"] is not None)
        waits = sorted(r["wait_s"] for r in r_list if r["wait_s"] is not None)
        p90_exec = _pct(execs, 90)
        if p90_exec <= quick_threshold_seconds:
            candidate_routes.append(
                {
                    "route": route,
                    "count": len(r_list),
                    "exec_p50_seconds": _pct(execs, 50),
                    "exec_p90_seconds": p90_exec,
                    "exec_max_seconds": round(execs[-1], 2) if execs else 0.0,
                    "wait_p50_seconds": _pct(waits, 50),
                    "wait_p90_seconds": _pct(waits, 90),
                    "total_wait_seconds": round(sum(waits), 2),
                    "strict_all_le_threshold": bool(
                        execs and execs[-1] <= quick_threshold_seconds
                    ),
                }
            )
    candidate_routes.sort(key=lambda item: (-item["total_wait_seconds"], -item["count"]))

    unlabeled_quick_candidates = {
        "quick_threshold_seconds": quick_threshold_seconds,
        "total_normal_done_count": len(normal_done),
        "fast_normal_done_count": len(fast_normals),
        "fast_normal_fraction": round(len(fast_normals) / len(normal_done), 3)
        if normal_done
        else 0.0,
        "waited_gt_60s_count": len(fast_waited_gt_60),
        "waited_gt_60s_p50_seconds": _pct(sorted(fast_waited_gt_60), 50),
        "waited_gt_60s_p90_seconds": _pct(sorted(fast_waited_gt_60), 90),
        "total_avoidable_wait_seconds": round(sum(fast_waited_gt_60), 2),
        "candidate_routes": candidate_routes[:20],
    }

    # Pause & checkpoint events from events/
    events_dir = root / "events"
    queue_pause_events_count = 0
    queue_resume_events_count = 0
    total_recorded_paused_seconds = 0.0
    job_pause_events_count = 0
    job_resume_events_count = 0

    if events_dir.is_dir():
        for ev_path in sorted(events_dir.glob("*.json")):
            ev = _load_json_safe(ev_path)
            if not ev:
                continue
            ev_ts = ev.get("timestamp")
            if cutoff is not None and isinstance(ev_ts, (int, float)) and ev_ts < cutoff:
                continue
            kind = ev.get("kind")
            payload = ev.get("payload") or {}
            if kind == "queue_paused":
                queue_pause_events_count += 1
            elif kind == "queue_resumed":
                queue_resume_events_count += 1
                paused_s = payload.get("paused_seconds")
                if isinstance(paused_s, (int, float)) and paused_s >= 0:
                    total_recorded_paused_seconds += float(paused_s)
            elif kind == "job_pause":
                job_pause_events_count += 1
            elif kind == "job_resume":
                job_resume_events_count += 1

    coop_count = sum(1 for r in active_records if r["cooperative_checkpoint"])
    cont_count = sum(1 for r in active_records if r["continuation_of"])
    perf_win_count = sum(1 for r in active_records if r["performance_window"])

    top_blocking_routes = sorted(
        blocker_route_stats.values(),
        key=lambda item: (-item["total_blocked_seconds"], -item["blocked_quick_count"]),
    )

    pause_and_checkpoint_efficiency = {
        "cooperative_checkpoint_jobs_count": coop_count,
        "cooperative_checkpoint_fraction": round(coop_count / len(active_records), 4)
        if active_records
        else 0.0,
        "committed_handoffs_count": committed_handoffs_count,
        "continuation_jobs_count": cont_count,
        "performance_window_jobs_count": perf_win_count,
        "performance_window_deferrals_count": performance_window_deferrals_count,
        "queue_pause_events_count": queue_pause_events_count,
        "queue_resume_events_count": queue_resume_events_count,
        "total_recorded_paused_seconds": round(total_recorded_paused_seconds, 2),
        "job_pause_events_count": job_pause_events_count,
        "job_resume_events_count": job_resume_events_count,
        "in_flight_blockers_of_quick_jobs": {
            "total_blocked_quick_jobs": non_cooperative_blocker_count
            + cooperative_blocker_count,
            "non_cooperative_blocker_count": non_cooperative_blocker_count,
            "cooperative_blocker_count": cooperative_blocker_count,
            "potential_wait_reducible_by_cooperative_pause_seconds": round(
                potential_wait_reducible_seconds, 2
            ),
            "top_blocking_routes": top_blocking_routes[:10],
        },
    }

    return {
        "schema": TELEMETRY_SCHEMA,
        "queue_dir": str(root),
        "generated_at": now,
        "since_hours": since_hours,
        "total_evaluated_jobs": len(active_records),
        "external_lease_admissions_excluded": external_lease_admissions_count,
        "by_service_class": by_service_class,
        "quick_efficiency": quick_efficiency,
        "unlabeled_quick_candidates": unlabeled_quick_candidates,
        "pause_and_checkpoint_efficiency": pause_and_checkpoint_efficiency,
    }


def render_telemetry_text(report: dict) -> str:
    """Format a human-readable summary of the telemetry report."""
    lines: list[str] = []
    window = (
        f"last {report['since_hours']}h"
        if report.get("since_hours") is not None
        else "all retained two-class history"
    )
    lines.append(
        f"GPU Greenroom Telemetry ({window}, {report['total_evaluated_jobs']} jobs)"
    )
    lines.append("=" * len(lines[0]))

    lines.append("\n1. Wait & Execution by Service Class:")
    for sc in ("quick", "normal"):
        info = report["by_service_class"].get(sc, {})
        w = info.get("wait_seconds", {})
        e = info.get("execution_seconds", {})
        lines.append(
            f"  - {sc:6s}: {info.get('count', 0):4d} jobs | "
            f"wait p50={w.get('p50', 0):.1f}s p90={w.get('p90', 0):.1f}s p95={w.get('p95', 0):.1f}s max={w.get('max', 0):.1f}s "
            f"(>30s: {w.get('wait_gt_30s_count', 0)}) | "
            f"exec p50={e.get('p50', 0):.1f}s p90={e.get('p90', 0):.1f}s"
        )

    qe = report["quick_efficiency"]
    wa = qe["wait_attribution"]
    lines.append("\n2. Quick Route Efficiency & Wait Attribution:")
    lines.append(
        f"  - Queue jumps: {qe['queue_jumping_quick_jobs']}/{qe['total_quick_jobs']} quick jobs "
        f"bypassed {qe['total_bypassed_normal_jobs']} queued normal jobs, "
        f"saving ~{qe['estimated_wait_saved_seconds']:.1f}s "
        f"({qe['mean_wait_saved_when_jumping_seconds']:.1f}s per jumping job)"
    )
    lines.append(
        f"  - Why quick jobs waited: immediate(<=5s)={wa['immediate_le_5s']}, "
        f"blocked_by_in_flight_job={wa['blocked_by_in_flight_job']}, "
        f"waited_behind_other_dispatches={wa['waited_behind_other_dispatches']}, "
        f"paused_or_lease={wa['paused_or_external_lease_block']}"
    )

    fn = report["unlabeled_quick_candidates"]
    lines.append(
        f"\n3. Unlabeled Quick Work ('False Normals' finishing <={fn['quick_threshold_seconds']:.0f}s as normal):"
    )
    lines.append(
        f"  - {fn['fast_normal_done_count']}/{fn['total_normal_done_count']} "
        f"({fn['fast_normal_fraction'] * 100:.1f}%) completed normal jobs finished <={fn['quick_threshold_seconds']:.0f}s; "
        f"{fn['waited_gt_60s_count']} waited >60s (wait p50={fn['waited_gt_60s_p50_seconds']:.1f}s, "
        f"p90={fn['waited_gt_60s_p90_seconds']:.1f}s, total avoidable wait={fn['total_avoidable_wait_seconds']:.1f}s)"
    )
    if fn["candidate_routes"]:
        lines.append("  - Top candidate routes for quick_routes / --expected-seconds:")
        for r in fn["candidate_routes"][:8]:
            lines.append(
                f"      * {r['route']} (n={r['count']}, exec p50={r['exec_p50_seconds']:.1f}s max={r['exec_max_seconds']:.1f}s, "
                f"wait p50={r['wait_p50_seconds']:.1f}s, total wait={r['total_wait_seconds']:.1f}s)"
            )

    pc = report["pause_and_checkpoint_efficiency"]
    blk = pc["in_flight_blockers_of_quick_jobs"]
    lines.append("\n4. Pause & Cooperative Checkpoint Efficiency:")
    lines.append(
        f"  - Cooperative checkpoint jobs: {pc['cooperative_checkpoint_jobs_count']} "
        f"({pc['cooperative_checkpoint_fraction'] * 100:.2f}%), "
        f"committed handoffs={pc['committed_handoffs_count']}, "
        f"performance deferrals={pc['performance_window_deferrals_count']}"
    )
    lines.append(
        f"  - Queue pause events={pc['queue_pause_events_count']}, "
        f"resumes={pc['queue_resume_events_count']} "
        f"(recorded paused time={pc['total_recorded_paused_seconds']:.1f}s)"
    )
    lines.append(
        f"  - In-flight blockers of quick jobs: {blk['non_cooperative_blocker_count']} non-cooperative, "
        f"{blk['cooperative_blocker_count']} cooperative "
        f"(reducible wait if cooperative={blk['potential_wait_reducible_by_cooperative_pause_seconds']:.1f}s)"
    )
    return "\n".join(lines)
