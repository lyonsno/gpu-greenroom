# Operator job controls

The monitor exposes per-job checkpoint pause/resume and an independent checkbox
to let queued work pass a paused job. A running pause remains requested until the
producer saves a checkpoint and exits. The continuation stays pending but held;
resuming admits it to ordinary dispatch without interrupting another job. A job
without the cooperative checkpoint contract shows Pause unsupported and a saved
request to its submitting agent. The configured feedback adapter delivers that
request independently of GPU dispatch; saved, queued and submitted are distinct.

`POST /api/job-action` requires the displayed job ID, request digest, state,
started_at, an idempotent request_id and action. Actions are `pause` (boolean
allow_bypass), `resume`, `ask-pause`, and `requeue` (placement previous/second/end).
They use the same authentication and job-control switch as cancellation.

Restore cancelled includes performance jobs cancelled while pending and native
confirmation cancellations recorded as failed. It does not automatically retry
a workload that already executed. Original receipts remain unchanged. A retry
has a fresh job ID/output directory and requires fresh performance confirmation.
Explicit command arguments are retained. Existing equivalent later jobs prevent
duplicate execution. A source job restores at most once; further retries act on
the replacement if it is cancelled. Changed prevalidated inputs and application
preflights needing a new producer result refuse with an explicit explanation.

Previous means original submission order among jobs still waiting; second means
after the first waiting job (not after the currently running job); end follows
the jobs waiting at the action boundary. Those predecessor identities are
recorded separately from submitted_at. Explicit placement overrides class
ordering for this restored job; it never interrupts current execution or
releases a queue-wide pause or external lease.

The first pause/resume/requeue operation upgrades dispatch policy to v5, requiring
operator-job-controls.v1. Old workers reject that policy instead of ignoring a
hold. Install the matching worker, CLI/readers and producer checkpoint library
before activation; older source-pinned submitters cannot read v5. The browser
does not constitute evidence that an old running producer can honor a pause.
Staged retry publication is replayable from the policy's job_restores map.
Paused continuation checkpoints retain the existing explicit retention pins.

## Time until safe pause

Running checkpoint-capable jobs show an advisory countdown beside their details,
both before and after a pause request. Producers can publish it from their managed
process, including from a `run_units` perform callback:

```python
from gpu_queue.cooperative import report_pause_estimate

report_pause_estimate(30, checkpoint="end of current case")
# Do the case, then use the existing save/quiesce/continuation hook.
```

The estimate includes the remaining work until the next checkpoint plus saving,
GPU quiescence and process-exit overhead. It is not whole-job runtime. Update it
when the forecast changes or call `report_pause_estimate(None)` to withdraw it.
Without a report the monitor says Unknown; elapsed estimates become overdue,
never "paused" or "GPU free". Producer estimates are not deadlines, preemption,
or scheduling authority. The existing worker still verifies exit and releases
the mutex. Reports are bound to the managed request and execution incarnation;
stale, malformed, future-dated and other-attempt records degrade to Unknown.
The countdown refreshes with normal queue snapshots. Old producers continue to
work but cannot report a number until they adopt this optional hook and library.
No historical whole-job duration is silently substituted for checkpoint time.
