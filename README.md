# gpu-greenroom

Filesystem-backed GPU job queue with flock serialization. One GPU job at a time, strict FIFO, crash-safe receipts.

## Problem

Heavy spatial-AI generation jobs (TRELLIS2MLX, Pixal3D, SuperMat, MoGe) share a single Mac GPU. Running multiple jobs concurrently risks kernel panics, Metal scheduler deadlocks, and OOM crashes. gpu-greenroom serializes GPU-bound work so only one job runs at a time.

## Install

```bash
cd ~/dev/gpu-greenroom
uv pip install -e .
```

## Usage

```bash
# Submit a job (output goes to durable dir in queue)
gpu-greenroom submit trellis2mlx /path/to/image.png

# Submit with explicit output dir
gpu-greenroom submit trellis2mlx /path/to/image.png /path/to/output/

# Submit with custom params
gpu-greenroom submit trellis2mlx /path/to/image.png /path/to/output/ -p seed=99 resolution=768

# List queue
gpu-greenroom list
gpu-greenroom list -s pending

# Check job status
gpu-greenroom status <job-id>

# Cancel a pending job
gpu-greenroom cancel <job-id>

# Start the worker (runs jobs sequentially, polls every 2s)
gpu-greenroom worker

# Pause the queue (finishes current job, then waits)
gpu-greenroom pause

# Resume a paused queue
gpu-greenroom resume

# Recover stale jobs after crash
gpu-greenroom recover

# Claim/renew/status/release a cooperative external GPU lease
gpu-greenroom lease claim \
  --owner neural-fire \
  --agent-id render-holder \
  --repo-root ~/dev/kaminos \
  --pid "$PID" \
  --process-group "$PGID" \
  --effective-route "chrome neural-fire --metal" \
  --backend metal \
  --device mps:0 \
  --profile interactive-render \
  --supports-checkpoints \
  --ttl-seconds 300
gpu-greenroom lease renew <lease-id> --interruptible
gpu-greenroom lease status
gpu-greenroom lease release <lease-id> --released-by neural-fire --reason "capture checkpoint complete"

# Request, answer, and wait for a cooperative bump
gpu-greenroom bump request \
  --bump-id resident-cold-load \
  --requester resident-loader \
  --agent-id resident-loader \
  --repo-root ~/dev/kaminos \
  --intended-route "sam31 cold-load --mps" \
  --workload-class cold-model-load \
  --memory-pressure 3.32GB \
  --estimated-occupancy 90s \
  --full-quiescence-required \
  --reason "prepare resident model before capture" \
  --callback-address "file:///tmp/resident-greenroom-callback"
gpu-greenroom bump list
gpu-greenroom bump grant resident-cold-load --granted-by neural-fire --checkpoint capture-42 --quiescence-confirmed
gpu-greenroom bump decline resident-cold-load --declined-by neural-fire --reason "training cannot checkpoint"
gpu-greenroom bump wait resident-cold-load --timeout 600
```

## Queue directory layout

```
~/.local/state/gpu-greenroom/
  gpu.lock              # flock file for mutual exclusion
  coordination.lock     # short-lived metadata transition lock, not execution authority
  paused                # present when queue is paused (touch to pause, rm to resume)
  job_types.json        # optional: custom job type configs (overrides defaults)
  outputs/              # durable output directory for jobs submitted without explicit output_dir
  leases/
    current.json        # current external lease, released lease, handoff, or ownership_unknown
    receipts/           # claim, renew, release, handoff, and unknown-state receipts
  bumps/
    <bump-id>.json      # inbound cooperative bump request state
    receipts/           # request, grant, and decline receipts
  events/               # filesystem-watch wake events for bump wait
  pending/
    <job-id>/
      request.json      # what was submitted
      status.json       # current state
  running/              # at most one job
    <job-id>/
      request.json
      status.json
      stdout.log
      stderr.log
  done/
    <job-id>/
      request.json
      status.json
      stdout.log
      stderr.log
      receipt.json      # full route identity and outcome
  failed/
    <job-id>/...        # same as done, with failure_phase
  cancelled/
    <job-id>/...
```

Override the queue directory with `GPU_GREENROOM_DIR` or `--queue-dir`.

## Cooperative external leases and bumps

Greenroom can coordinate with GPU work that was not launched by the Greenroom
worker, such as an interactive Chrome/WebGPU render or resident model process.
The protocol is cooperative: it records ownership and safe handoff windows; it
does not preempt, pause, kill, signal, timeout, or reorder the current holder.

External workloads claim a renewable lease with explicit identity:

- owner and agent ID;
- repo root;
- PID and process group when known;
- effective route, backend, device, and profile;
- checkpoint support and current interruptibility;
- acquisition and renewal timestamps;
- release or ownership-unknown receipts.

Bump requests describe another lane's intended work:

- requester and agent ID;
- intended route;
- workload class;
- expected memory pressure;
- estimated occupancy, recorded as diagnostic only;
- whether full quiescence is required;
- reason and callback address.

The existing `gpu.lock` remains the only execution mutex for Greenroom worker
jobs. `lease claim` briefly acquires `gpu.lock` to prove no worker owns the GPU
at the claim boundary, then persists an external lease. While a lease is
`active`, `handoff`, or `ownership_unknown`, worker dispatch returns without
consuming FIFO jobs. `coordination.lock` only serializes JSON state transitions;
it is not execution authority.

A holder can answer a bump three ways:

- Grant now with `bump grant --quiescence-confirmed`: the current lease enters
  `handoff`, the requester wakes, and FIFO worker dispatch stays blocked.
- Grant after checkpoint without `--quiescence-confirmed`: the bump enters
  `grant_pending_checkpoint`; `lease release` at the checkpoint converts it to
  `granted` and moves the lease into `handoff`.
- Decline with a reason: ownership remains with the holder, and the requester
  receives a durable decline receipt.

The requester must still claim its own lease with `lease claim --handoff-bump-id
<bump-id>` before running external work. A bump grant is not execution
authority by itself; it is a wakeup that a cooperative handoff window exists.

Lease TTL is a coordination-state freshness check only. Expiry transitions the
lease to `ownership_unknown`, never ownership-free. A dead holder PID also
transitions to `ownership_unknown` unless a release receipt already exists.
Operators or agents must recover authority explicitly before assuming the GPU
is safe for new work.

`bump wait` uses filesystem events on platforms that expose them and emits no
agent-side polling loop. It returns when the bump is granted, declined, or
closed, and prints the resulting JSON state.

## Job type config

Job types can be configured via `job_types.json` in the queue directory, or via the defaults in `cli.py`.

Each job type is a dict with:

| Field | Required | Description |
|-------|----------|-------------|
| `cmd` | yes | Command template as list of strings. `{input_path}`, `{output_dir}`, and any param key are substituted. |
| `cwd` | no | Working directory for the subprocess. |
| `env` | no | Environment variables merged into `os.environ`. |
| `defaults` | no | Default param values. User params override defaults. Reserved keys (`input_path`, `output_dir`) always win. |
| `timeout` | no | Timeout in seconds. `null`/absent = no timeout. |

See `job_types.example.json` for TRELLIS2MLX, SuperMat, MoGe, and Pixal3D templates.

Bare command lists are also accepted for simple cases: `{"echo": ["echo", "{input_path}"]}`.

## Route identity shield

A Greenroom route is the local route that actually works on this machine, not
just an upstream/default command that happens to launch. Before adding,
reviewing, or approving a `job_types.json` route, preserve the effective runner,
environment, device, backend, and fallback knobs that made the route safe.

For each new or reviewed job type, record these four lines in the review,
return packet, or closeout:

- **Known-good local runner checked:** `yes/no/path`, including nearby wrappers
  like `generate.py`, `batch_generate.sh`, repo READMEs, or prior receipts.
- **Effective env/device/backend preserved:** exact env or args for device,
  dtype, backend, fallback knobs, and any forbidden default that was avoided.
- **First receipt/log proves backend/device:** stdout/status snippet showing the
  effective route identity before treating a heavy run as accepted.
- **Heavy run accepted before proof:** `no` by default; if `yes`, explain why
  the missing identity proof cannot lie about route/backend.

If a route launches with upstream/CUDA/default backend identity while a local
Mac/MPS runner needs different settings, the route is still candidate-only. A
started subprocess is not accepted Greenroom evidence until the receipt or log
proves the local route identity.

TRELLIS.2 on Apple Silicon is the current scar. Preserve and prove the local
Mac route's `ATTN_BACKEND=sdpa` / `SPARSE_ATTN_BACKEND=sdpa` identity where
that route applies. Accepting upstream `flash_attn` or a later MPS device
mismatch as incidental is a route-identity failure, not a model failure.

## Receipt schema

Every completed or failed job gets a `receipt.json`:

```json
{
  "job_id": "ab8647e17eb0",
  "job_type": "trellis2mlx",
  "status": "done",
  "input_path": "/path/to/image.png",
  "output_dir": "/path/to/output",
  "effective_route": "python -u generate.py --image /path/to/image.png ...",
  "effective_cwd": "/Users/noahlyons/dev/trellis2mlx",
  "effective_env": {"PYTHONPATH": "."},
  "effective_defaults": {"seed": "42", "resolution": "512", ...},
  "effective_timeout": null,
  "ignored_params": null,
  "started_at": 1718000000.0,
  "finished_at": 1718001200.0,
  "exit_code": 0,
  "failure_phase": null,
  "error_message": null
}
```

## Failure phases

| Phase | Meaning |
|-------|---------|
| `dispatch` | Unknown job type |
| `launch` | Subprocess failed to start |
| `execution` | Subprocess exited with non-zero code |
| `timeout` | Subprocess exceeded configured timeout |
| `stale_recovery` | Process died without cleanup; recovered by `recover` |

## Serialization

Uses `flock(LOCK_EX | LOCK_NB)` on `gpu.lock`. Only one worker can run a job at a time. If the lock is held, `run_one()` returns immediately without queuing or blocking. Cancel also acquires the lock to prevent races. External leases and handoffs block worker dispatch before and after the worker acquires `gpu.lock`, so a granted handoff cannot be stolen by the next FIFO job.

## Tests

```bash
uv run --extra test python -m pytest tests/ -v
```

105 tests covering serialization, failure receipts, stale recovery, cancel safety, FIFO order, param injection prevention, rich config (cwd/env/defaults), receipt route identity, configurable timeout, pause/resume, durable output dirs, volatile path warnings, CLI, cooperative external leases, bump handoffs, ownership-unknown lease expiry, dead/live PID handling, duplicate bump requests, concurrent grants, release/grant races, handoff identity binding, wait wakeup races, and worker race prevention.
