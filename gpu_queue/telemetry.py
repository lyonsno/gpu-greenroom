"""Queue wait-time, quick-route bypass efficiency, and pause/checkpoint telemetry."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from . import dispatch

TELEMETRY_SCHEMA = "gpu-greenroom.telemetry.v2"


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
            # Ignore foreign supervisor files that lack valid Greenroom job identity
            if (
                st is None
                or not isinstance(st.get("status"), str)
                or st.get("job_id") != job_dir.name
            ):
                continue
            score = (
                int(req is not None)
                + int(st is not None)
                + int(rc is not None)
                + int(st.get("status") == state_name)
            )
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

    all_records: list[dict] = []
    external_lease_admissions_count = 0

    for job_id, (_score, state_name, job_dir, req, st, rc) in candidates_by_id.items():
        job_type = st.get("job_type") or req.get("job_type")
        if job_type == "external-lease-admission":
            external_lease_admissions_count += 1
            continue
        sub_at = st.get("submitted_at") or req.get("submitted_at")
        start_at = st.get("started_at") or rc.get("started_at")
        fin_at = st.get("finished_at") or rc.get("finished_at")

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
        committed_handoff = bool(handoff and handoff.get("status") == "committed")
        deferrals_dir = job_dir / "performance-deferrals"
        deferrals_count = (
            sum(1 for _ in deferrals_dir.glob("*.json"))
            if deferrals_dir.is_dir()
            else 0
        )

        all_records.append(
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
                "committed_handoff": committed_handoff,
                "performance_deferrals_count": deferrals_count,
            }
        )

    # Filter to two-class records when two-class jobs exist; otherwise include all
    two_class_all = [r for r in all_records if r["mode"] == "two-class"]
    population_all = two_class_all if two_class_all else all_records

    # Keep unfiltered lookup maps so cross-boundary in-flight blockers and bypassed
    # normal jobs are never lost when --since-hours is used.
    all_records_by_id = {r["job_id"]: r for r in population_all}
    all_started_unfiltered = [
        r
        for r in population_all
        if r["submitted_at"] is not None and r["started_at"] is not None
    ]
    normal_started_unfiltered = [
        r for r in all_started_unfiltered if r["service_class"] == "normal"
    ]

    if cutoff is not None:
        active_records = [
            r
            for r in population_all
            if any(
                ts is not None and ts >= cutoff
                for ts in (r["submitted_at"], r["started_at"], r["finished_at"])
            )
        ]
    else:
        active_records = list(population_all)

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

    queue_jumping_quick_jobs = 0
    total_bypassed_normal_jobs = 0
    bypassed_jobs_execution_seconds = 0.0
    recorded_bypass_quick_jobs = 0
    reconstructed_bypass_quick_jobs = 0

    wait_attribution = {
        "immediate_le_5s": 0,
        "overlapped_in_flight_job": 0,
        "other_dispatches_during_wait": 0,
        "unattributed": 0,
    }

    non_cooperative_blocker_count = 0
    cooperative_blocker_count = 0
    non_cooperative_overlap_seconds = 0.0
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
                all_records_by_id[jid]
                for jid in recorded_bypassed_ids
                if jid in all_records_by_id
            ]
            bypassed_count = len(recorded_bypassed_ids)
        else:
            bypassed_jobs = [
                n
                for n in normal_started_unfiltered
                if n["submitted_at"] <= q_sub and n["started_at"] > q_start
            ]
            bypassed_count = len(bypassed_jobs)

        if bypassed_count > 0:
            queue_jumping_quick_jobs += 1
            total_bypassed_normal_jobs += bypassed_count
            if isinstance(recorded_bypassed_ids, list):
                recorded_bypass_quick_jobs += 1
            else:
                reconstructed_bypass_quick_jobs += 1
            bypassed_jobs_execution_seconds += sum((n["exec_s"] or 0.0) for n in bypassed_jobs)

        # Attribute wait time using full started-job history so boundary jobs are never misattributed
        in_flight = [
            o
            for o in all_started_unfiltered
            if o["job_id"] != q["job_id"]
            and o["started_at"] <= q_sub
            and (o["finished_at"] or o["started_at"]) > q_sub
        ]
        started_before_q = [
            o
            for o in all_started_unfiltered
            if o["job_id"] != q["job_id"] and q_sub < o["started_at"] < q_start
        ]

        if q_wait <= 5.0:
            wait_attribution["immediate_le_5s"] += 1
        elif started_before_q:
            wait_attribution["other_dispatches_during_wait"] += 1
        elif in_flight:
            wait_attribution["overlapped_in_flight_job"] += 1
        else:
            wait_attribution["unattributed"] += 1

        if q_wait > 5.0 and in_flight:
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
                non_cooperative_overlap_seconds += overlap_s

    # Runtime hint accuracy on completed quick jobs (failed/cancelled tracked separately)
    hinted_quick_all = [
        r
        for r in active_records
        if r["service_class"] == "quick"
        and (
            isinstance(r.get("runtime_hint_comparison"), dict)
            or isinstance(r.get("runtime_hint"), dict)
        )
    ]
    hinted_quick_done = [
        r
        for r in hinted_quick_all
        if r["status"] == "done" and r["exec_s"] is not None
    ]
    within_expected = 0
    within_threshold = 0
    for r in hinted_quick_done:
        cmp_d = r.get("runtime_hint_comparison") or {}
        hint_d = r.get("runtime_hint") or {}
        exp_s = cmp_d.get("expected_seconds") or hint_d.get("expected_seconds")
        actual_s = cmp_d.get("actual_seconds")
        measured_s = float(actual_s) if isinstance(actual_s, (int, float)) else float(r["exec_s"])
        if exp_s is not None and measured_s <= float(exp_s):
            within_expected += 1
        if measured_s <= quick_threshold_seconds:
            within_threshold += 1

    quick_count = len(quick_started)
    quick_efficiency = {
        "total_quick_jobs": quick_count,
        "queue_jumping_quick_jobs": queue_jumping_quick_jobs,
        "queue_jumping_fraction": round(queue_jumping_quick_jobs / quick_count, 3)
        if quick_count
        else 0.0,
        "total_bypassed_normal_jobs": total_bypassed_normal_jobs,
        "recorded_bypass_quick_jobs": recorded_bypass_quick_jobs,
        "reconstructed_bypass_quick_jobs": reconstructed_bypass_quick_jobs,
        "bypassed_jobs_observed_execution_seconds": round(bypassed_jobs_execution_seconds, 2),
        "wait_attribution": wait_attribution,
        "runtime_hint_accuracy": {
            "evaluated_quick_jobs": len(hinted_quick_done),
            "non_done_hinted_quick_jobs": len(hinted_quick_all) - len(hinted_quick_done),
            "within_expected_seconds_count": within_expected,
            "within_quick_threshold_count": within_threshold,
            "exceeded_quick_threshold_count": len(hinted_quick_done) - within_threshold,
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
    fast_waited_gt_60_jobs = [
        r
        for r in fast_normals
        if r["wait_s"] is not None and r["wait_s"] > 60.0
    ]
    fast_waited_gt_60 = [r["wait_s"] for r in fast_waited_gt_60_jobs]

    # These intervals suggest candidates; they do not simulate alternate scheduling.
    other_route_slow_normal_overlap_seconds = 0.0
    for fn_job in fast_waited_gt_60_jobs:
        fn_sub = fn_job["submitted_at"]
        fn_start = fn_job["started_at"]
        if fn_sub is None or fn_start is None:
            continue
        intervening_slow_normals = [
            o
            for o in normal_started_unfiltered
            if o["job_id"] != fn_job["job_id"]
            and o["route"] != fn_job["route"]
            and fn_sub < o["started_at"] < fn_start
            and (o["exec_s"] or 0.0) > quick_threshold_seconds
        ]
        bypassable_s = sum(
            max(0.0, min(fn_start, o["finished_at"] or fn_start) - o["started_at"])
            for o in intervening_slow_normals
        )
        other_route_slow_normal_overlap_seconds += min(fn_job["wait_s"], bypassable_s)

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

    total_wait_gt_60_s = round(sum(fast_waited_gt_60), 2)
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
        "observed_wait_gt_60s_seconds": total_wait_gt_60_s,
        "observed_wait_overlap_with_other_route_slow_normals_seconds": round(
            other_route_slow_normal_overlap_seconds, 2
        ),
        "wait_not_accounted_for_by_those_intervals_seconds": round(
            max(0.0, total_wait_gt_60_s - other_route_slow_normal_overlap_seconds), 2
        ),
        "candidate_routes": candidate_routes,
    }

    # Pause & checkpoint events from events/, operator-actions/, and dispatch-policy.json
    events_dir = root / "events"
    seen_pause_keys: set[str] = set()
    seen_resume_keys: set[str] = set()
    seen_job_action_ids: set[str] = set()

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
            ev_ts = (
                ev.get("emitted_at")
                if ev.get("emitted_at") is not None
                else ev.get("timestamp")
            )
            if cutoff is not None and isinstance(ev_ts, (int, float)) and ev_ts < cutoff:
                continue
            kind = ev.get("kind")
            payload = ev.get("payload") or {}
            if kind == "queue_paused":
                dedup_key = str(
                    payload.get("request_id")
                    or payload.get("epoch")
                    or ev.get("subject_id")
                    or ev_path.name
                )
                if dedup_key not in seen_pause_keys:
                    seen_pause_keys.add(dedup_key)
                    queue_pause_events_count += 1
            elif kind == "queue_resumed":
                dedup_key = str(
                    payload.get("request_id")
                    or payload.get("epoch")
                    or ev.get("subject_id")
                    or ev_path.name
                )
                if dedup_key not in seen_resume_keys:
                    seen_resume_keys.add(dedup_key)
                    queue_resume_events_count += 1
                    paused_s = payload.get("paused_seconds")
                    if isinstance(paused_s, (int, float)) and paused_s >= 0:
                        total_recorded_paused_seconds += float(paused_s)
            elif kind == "job_pause":
                req_id = str(payload.get("request_id") or ev_path.name)
                if req_id not in seen_job_action_ids:
                    seen_job_action_ids.add(req_id)
                    job_pause_events_count += 1
            elif kind == "job_resume":
                req_id = str(payload.get("request_id") or ev_path.name)
                if req_id not in seen_job_action_ids:
                    seen_job_action_ids.add(req_id)
                    job_resume_events_count += 1

    # Also ingest operator-actions/*.json from the --admission-control console path
    op_actions_dir = root / "operator-actions"
    if op_actions_dir.is_dir():
        for oa_path in sorted(op_actions_dir.glob("*.json")):
            oa = _load_json_safe(oa_path)
            if (
                not oa
                or oa.get("schema") != "gpu-greenroom.operator-action.v1"
                or oa.get("phase") != "completed"
            ):
                continue
            res = oa.get("result") or {}
            req_d = oa.get("request") or {}
            if res.get("idempotent"):
                continue
            oa_ts = res.get("acknowledged_at") or res.get("requested_at")
            if cutoff is not None and isinstance(oa_ts, (int, float)) and oa_ts < cutoff:
                continue
            action = res.get("action") or req_d.get("action")
            epoch = res.get("epoch") or req_d.get("epoch")
            req_id = req_d.get("request_id") or oa_path.stem
            dedup_keys = {str(k) for k in (req_id, epoch) if k}
            if action == "pause":
                if not (dedup_keys & seen_pause_keys):
                    seen_pause_keys.update(dedup_keys)
                    queue_pause_events_count += 1
            elif action == "resume":
                if not (dedup_keys & seen_resume_keys):
                    seen_resume_keys.update(dedup_keys)
                    queue_resume_events_count += 1
                    paused_s = res.get("paused_seconds")
                    if paused_s is None:
                        prev_pause = oa.get("previous_pause") or res.get("previous_pause") or {}
                        eff_at = prev_pause.get("effective_at") or prev_pause.get("requested_at")
                        if isinstance(eff_at, (int, float)) and isinstance(oa_ts, (int, float)):
                            paused_s = max(0.0, float(oa_ts) - float(eff_at))
                    if isinstance(paused_s, (int, float)) and paused_s >= 0:
                        total_recorded_paused_seconds += float(paused_s)

    # Also ingest persisted per-job pause/resume actions from dispatch-policy.json
    job_actions_map = policy.get("job_actions")
    if isinstance(job_actions_map, dict):
        for req_id, entry in job_actions_map.items():
            if not isinstance(entry, dict) or str(req_id) in seen_job_action_ids:
                continue
            req_body = entry.get("request") or {}
            act = req_body.get("action")
            if act not in ("pause", "resume"):
                continue
            # Job actions in dispatch-policy.json don't carry an action timestamp unless requeued;
            # include when cutoff is None or when the target job was active in the window.
            target_jid = req_body.get("job_id")
            if cutoff is not None and target_jid not in {r["job_id"] for r in active_records}:
                continue
            seen_job_action_ids.add(str(req_id))
            if act == "pause":
                job_pause_events_count += 1
            elif act == "resume":
                job_resume_events_count += 1

    coop_count = sum(1 for r in active_records if r["cooperative_checkpoint"])
    cont_count = sum(1 for r in active_records if r["continuation_of"])
    perf_win_count = sum(1 for r in active_records if r["performance_window"])
    committed_handoffs_count = sum(
        1 for r in active_records if r["committed_handoff"]
    )
    performance_window_deferrals_count = sum(
        r["performance_deferrals_count"] for r in active_records
    )
    active_job_holds_count = len(policy.get("job_holds") or {})

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
        "active_job_holds_count": active_job_holds_count,
        "in_flight_blockers_of_quick_jobs": {
            "total_blocked_quick_jobs": non_cooperative_blocker_count
            + cooperative_blocker_count,
            "non_cooperative_blocker_count": non_cooperative_blocker_count,
            "cooperative_blocker_count": cooperative_blocker_count,
            "observed_wait_overlap_with_non_cooperative_jobs_seconds": round(
                non_cooperative_overlap_seconds, 2
            ),
            "top_blocking_routes": top_blocking_routes,
        },
    }

    return {
        "schema": TELEMETRY_SCHEMA,
        "queue_dir": str(root),
        "generated_at": now,
        "since_hours": since_hours,
        "evidence_limits": {
            "wait_intervals": "observed job-record timestamps; overlap is not proof of cause",
            "bypass": "claim snapshot when present, otherwise interval reconstruction",
            "savings": "not measured; no alternate-schedule replay",
            "checkpoint_support": "producer declaration, not proof of effective GPU release",
        },
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
    lines.append("\n2. Quick Bypass Evidence & Wait Observations:")
    lines.append(
        f"  - Queue jumps: {qe['queue_jumping_quick_jobs']}/{qe['total_quick_jobs']} quick jobs "
        f"bypassed {qe['total_bypassed_normal_jobs']} queued normal jobs, "
        f"(claim-recorded={qe['recorded_bypass_quick_jobs']}, "
        f"reconstructed={qe['reconstructed_bypass_quick_jobs']}); "
        f"their observed execution={qe['bypassed_jobs_observed_execution_seconds']:.1f}s, not measured savings"
    )
    lines.append(
        f"  - Wait observations (not causal attribution): immediate(<=5s)={wa['immediate_le_5s']}, "
        f"overlapped_in_flight_job={wa['overlapped_in_flight_job']}, "
        f"other_dispatches_during_wait={wa['other_dispatches_during_wait']}, "
        f"unattributed={wa['unattributed']}"
    )

    fn = report["unlabeled_quick_candidates"]
    lines.append(
        f"\n3. Short Completed Normal Work (finished <={fn['quick_threshold_seconds']:.0f}s):"
    )
    lines.append(
        f"  - {fn['fast_normal_done_count']}/{fn['total_normal_done_count']} "
        f"({fn['fast_normal_fraction'] * 100:.1f}%) completed normal jobs finished <={fn['quick_threshold_seconds']:.0f}s; "
        f"{fn['waited_gt_60s_count']} waited >60s (wait p50={fn['waited_gt_60s_p50_seconds']:.1f}s, "
        f"p90={fn['waited_gt_60s_p90_seconds']:.1f}s, observed >60s wait={fn['observed_wait_gt_60s_seconds']:.1f}s; "
        f"overlap with other-route slow normals={fn['observed_wait_overlap_with_other_route_slow_normals_seconds']:.1f}s, "
        f"remaining wait unaccounted for={fn['wait_not_accounted_for_by_those_intervals_seconds']:.1f}s)"
    )
    if fn["candidate_routes"]:
        lines.append(f"  - Top candidate routes ({min(8, len(fn['candidate_routes']))}/{len(fn['candidate_routes'])}; all in JSON), not eligibility proof:")
        for r in fn["candidate_routes"][:8]:
            lines.append(
                f"      * {r['route']} (n={r['count']}, exec p50={r['exec_p50_seconds']:.1f}s max={r['exec_max_seconds']:.1f}s, "
                f"wait p50={r['wait_p50_seconds']:.1f}s, total wait={r['total_wait_seconds']:.1f}s)"
            )

    pc = report["pause_and_checkpoint_efficiency"]
    blk = pc["in_flight_blockers_of_quick_jobs"]
    lines.append("\n4. Pause & Cooperative Checkpoint Efficiency:")
    lines.append(
        f"  - Producer-declared cooperative checkpointing: {pc['cooperative_checkpoint_jobs_count']} jobs "
        f"({pc['cooperative_checkpoint_fraction'] * 100:.2f}%), "
        f"job pause actions={pc['job_pause_events_count']}, resumes={pc['job_resume_events_count']}, "
        f"committed handoffs={pc['committed_handoffs_count']}, "
        f"performance deferrals={pc['performance_window_deferrals_count']}"
    )
    lines.append(
        f"  - Queue-wide start-gate pauses (holds dispatch): pauses={pc['queue_pause_events_count']}, "
        f"resumes={pc['queue_resume_events_count']} "
        f"(recorded paused time={pc['total_recorded_paused_seconds']:.1f}s)"
    )
    lines.append(
        f"  - In-flight overlaps with quick waits: {blk['non_cooperative_blocker_count']} non-cooperative, "
        f"{blk['cooperative_blocker_count']} cooperative "
        f"(observed non-cooperative overlap={blk['observed_wait_overlap_with_non_cooperative_jobs_seconds']:.1f}s, not predicted savings)"
    )
    return "\n".join(lines)
