# Quick Dispatch And Cooperative Checkpoints

Compatible empirical profile admission and the finite-case adapter are described
in [Runtime History](runtime-history.md). Those opt-in profiles use dispatch
policy v3; the existing exact-route and manual-choice contracts below remain.

FIFO remains the default. Optional two-class dispatch uses the existing worker,
execution flock, pause and lease checks. Each class is FIFO. When both classes
have pending work, a quick job gets one turn, followed by a normal job. A stream
of quick arrivals therefore cannot repeatedly bypass the normal class. The
fairness cursor is durable and advances at claim; failed executions count as
their class's turn. This bounds overtaking by number of jobs, not by wall time.

Eligibility is an explicit input owned by the queue operator. Configure exact
job types or exact structured-command route identities with references to their
observed runtime evidence. Nothing schedules on occupancy estimates. A quick job
that takes longer than expected keeps running under its normal command contract;
the scheduler does not turn a classification into a killing timeout.

```json
{
  "schema": "gpu-greenroom.dispatch-policy.v1",
  "mode": "two-class",
  "quick_job_types": {"capture": "path/to/observed-capture-timings.json"},
  "quick_routes": {"captures/profile-a": "path/to/profile-a-receipts.json"}
}
```

```sh
gpu-greenroom --queue-dir /durable/queue dispatch-policy --config policy.json --owner queue-operator
gpu-greenroom --queue-dir /durable/queue dispatch-policy
```

The policy is atomically written with an epoch and an event receipt. Reapplying
the same policy is idempotent. A corrupt policy or fairness state blocks new
starts and writes `dispatch-error.json`; it does not declare the GPU free.
`--reset-fairness` explicitly resets the cursor while configuring policy.
Pause, external leases, ownership-unknown and execution-lock contention continue
to prevent dispatch. No running job is reordered or signalled by this policy.
Before live activation, verify one worker on this queue supports
`two-class-dispatch.v1`; an old process that does not read the new policy is not
made current by writing a file. Claim receipts record policy epoch, effective
class and eligibility basis. Inspect those receipts to verify the dispatch route.

Eligible profiles select quick automatically; `--service-class normal` opts a
specific request out. Explicit quick requests outside eligibility reject before
enqueueing. These flags and `--cooperative-checkpoint` use command manifest v2.
Older CLIs reject v2 rather than discarding the options; older claimants cannot
dispatch the recorded required capabilities. Existing v1 commands still work.

## Operator Per-Job Moves

In the monitor, drag a pending automatic job into Quick work, or back into
Automatic work. The per-job Queue selector provides the same operation for
keyboard and touch. The Quick column remains visible when empty in two-class
mode. Only the confirmed server classification moves the row. A stale request,
a job already claimed, or an unavailable queue rejects or disables the move.
Running work and operator-start smoke sessions are not draggable.

This is a manual scheduling choice for one job, not a claim that its route is
measured fast. It does not whitelist the route for later submissions, alter
request bytes or validation, interrupt running work, bypass pauses/leases, or
change the alternating class fairness rule. Execution receipts include the
effective class and the exact per-job action. Actor provenance is authenticated
console access, not verified human identity.

The authenticated same-origin `POST /api/dispatch-class` takes `job_id`,
`request_digest` (SHA-256 of the exact submitted request bytes), `service_class`
(`normal` or `quick`), and an idempotent UUID `request_id`. Classification and
the action receipt commit together under the existing claim lock in
`dispatch-policy.json`. Repeating the same action returns its original receipt,
even after execution; reusing the UUID for different inputs rejects. Later
policy configuration preserves these receipts and overrides.

The first move upgrades the policy to `gpu-greenroom.dispatch-policy.v2`.
Before exposing the controls, update the queue worker to support
`operator-dispatch-override.v1`, the canonical submission CLI, and policy-reading
monitors. Older policy readers deliberately reject v2 rather than silently
misclassify work. An old claimant cannot advance a v2 queue. Do not roll back to
a v1 worker after manual moves without reconciling that state. No automatic
deletion or expiry is applied to overrides or their action receipts.

## Managed Work Unit Continuations

Quick selection cannot shorten a job already running. A cooperating workload
can finish a natural finite unit, save resumable state, confirm GPU quiescence,
and queue its next unit. The current process then exits normally. Only after
the worker verifies its process group has exited can the next job acquire the
execution flock. This protocol does not preempt a process or suspend a kernel.

The first adapter is Python for managed structured-command workloads. Submit
with an explicit owner and `--cooperative-checkpoint`, or set
`cooperative_checkpoint: true` in a command-v2 manifest. The worker supplies a
fresh context, clears stale inherited contexts and publishes child identity
through a readiness pipe. The hook verifies the request digest and actual
process-group identity before mutating the queue. Ordinary direct CPU programs
without that context get a false/no-op result from the optional hook.

```python
from gpu_queue.cooperative import load_checkpoint, yield_if_requested
from gpu_queue.models import JobRequest

state = load_checkpoint() or initial_state()
for unit in remaining_units(state):
    finish_unit(unit)
    yield_if_requested(
        save_checkpoint=lambda directory: save_state_and_return_descriptor(directory),
        quiesce=release_owned_gpu_resources_and_confirm,
        continuation=lambda checkpoint: JobRequest(
            job_type="command", input_path="", agent_id="example-owner",
            repo_root=repo_root, command_cwd=repo_root,
            route_identity="training/profile-a",
            command_argv=[python_executable, script_path],
            cooperative_checkpoint=True,
        ),
        on_submitted=register_next_unit_completion,
    )
```

The hook runs at a safe unit boundary. With no pause and no quick work due next,
it continues normally. `quiesce` returns exactly True to confirm the producer's
GPU release; False declines the handoff. Backend synchronization and save/load
correctness remain with the workload adapter and require backend-specific proof.
The CPU conformance fixture establishes queue mechanics, not physical GPU
quiescence or numerical fidelity of any model checkpoint.

Continuation preserves owner, repo, cwd, requested route and executable. A new
job ID and submission time put it back into normal queue competition. Its
checkpoint hash is bound into the request and verified on load. The original
environment overlay and command timeout are inherited when omitted; the timeout
belongs to each explicit unit command, not an estimated whole-workload deadline.
Pause leaves the continuation pending. Preparation lives in continuation-staging,
outside execution discovery. Registration callbacks run before a durable handoff
commit and atomic publication into pending. Callback exceptions, exits and crashes
leave an uncommitted preparation which cannot execute. Worker startup publishes
only verified committed preparations, without repeating the parent command.
`checkpoint-handoff.json` links both job IDs and the checkpoint path; publication
is idempotent. Registration targets the future job ID, not an already runnable job.

Saved descriptors live under `outputs/<parent-job-id>/checkpoints`. The helper
pins that output entry without expiry before saving, including on a declined or
failed handoff. Existing pins owned by another producer reject. There is no
automatic unpin or deletion: the owner uses `retain --unpin <parent-job-id>` only
after declaring that checkpoint unnecessary. Backend code should choose units
and checkpoint sizes with their storage cost in view; the helper preserves data
rather than selecting a retention cap. A volatile queue root remains volatile
even when Greenroom retention is pinned; use a durable queue for real work.

Smoke progress updates can carry `current_job_id` when advancing to a continuation.
The original request/digest remains immutable, and projection verifies the new
job's owner and containment. Publish meaningful counts through `smoke-request
update`, and drive operator-needed notifications from projected phase rather
than treating a full preparation count as an operator response.
