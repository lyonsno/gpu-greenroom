# Quick Dispatch And Cooperative Checkpoints

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
Pause leaves the continuation pending. Failed completion-registration callbacks
cancel that newly submitted pending unit rather than silently running it.
`checkpoint-handoff.json` links both job IDs and the checkpoint path.

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
