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
```

## Queue directory layout

```
~/.local/state/gpu-greenroom/
  gpu.lock              # flock file for mutual exclusion
  paused                # present when queue is paused (touch to pause, rm to resume)
  job_types.json        # optional: custom job type configs (overrides defaults)
  outputs/              # durable output directory for jobs submitted without explicit output_dir
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

Uses `flock(LOCK_EX | LOCK_NB)` on `gpu.lock`. Only one worker can run a job at a time. If the lock is held, `run_one()` returns immediately without queuing or blocking. Cancel also acquires the lock to prevent races.

## Tests

```bash
uv run --extra test python -m pytest tests/ -v
```

72 tests covering serialization, failure receipts, stale recovery, cancel safety, FIFO order, param injection prevention, rich config (cwd/env/defaults), receipt route identity, configurable timeout, pause/resume, durable output dirs, volatile path warnings, and CLI.
