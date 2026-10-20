# Producer Runtime Hints

Supply a rough estimate you already have. Do not run a calibration assay or
spend time deriving a precise number just to submit a job.

```sh
gpu-greenroom submit-command \
  --repo-root /absolute/repo --cwd /absolute/worktree \
  --route-identity preview/capture --agent-id capture-agent \
  --expected-seconds 10 --runtime-reason "one low-resolution frame" \
  -- node /absolute/capture.mjs
```

Both `submit` and `submit-command` accept `--expected-seconds` and optional
`--runtime-reason`. A command manifest uses `gpu-greenroom.command.v5` and
`"runtime_hint": {"expected_seconds": 10, "reason": "one low-resolution frame"}`.
Hints round-trip through preparation and submission. Manifest fields cannot
be overridden with these flags. Old clients reject v5; workers must advertise
`runtime-hint.v1` to claim hinted jobs.

Estimate the launched job including startup and cleanup, not just its inner
loop. Exclude time waiting in the queue or waiting for operator confirmation.
An estimate is advisory: it does not install a timeout, kill a job, certify an
uncontended host, or require an exact matching history/profile. Omit it when
unknown. Existing explicit command timeouts remain separate.

## First-Submission Quick Admission

The operator can enable hints in the existing two-class policy:

```json
{
  "schema": "gpu-greenroom.dispatch-policy.v4",
  "mode": "two-class",
  "producer_hints": {"max_seconds": 30}
}
```

Apply through `dispatch-policy --config FILE --owner OWNER`, preserving existing
route rules and runtime-history settings in that file. Per-job operator
overrides and their receipts remain preserved by the queue API.

A hint at or below the threshold qualifies on its first submission. The
dispatch receipt identifies it as `producer-estimated`, distinct from empirical
history or operator placement. An explicitly normal job opts out; manual
placement wins. Existing within-class FIFO and normal/Quick alternation still
apply. Quick jobs cannot forcibly preempt running work: a running producer
needs a cooperative checkpoint before another job can use its slot.

Underestimation may occupy the slot longer than expected. This first slice
records the discrepancy; it does not automatically punish a producer, lower
their estimate, or issue an overrun notification.

## Observation

Terminal receipts retain the submitted hint and `runtime_hint_comparison`.
Ordinary jobs use execution wall time, including worker overhead. Jobs with a
performance confirmation window use the recorded command wall time, excluding
operator wait; whole execution wall time is retained separately. Missing
workload execution does not become zero-duration success. Failed runs keep
available timing and are marked unsuccessful. Contention condition remains
`unknown`, not assumed clean.

The monitor shows the producer estimate and terminal observation. Mismatched
or missing receipt identity appears unverified. This is not a GPU-busy-time
measurement or a calibrated forecast. Compatible empirical profiles remain
available separately in [runtime-history.md](runtime-history.md).
