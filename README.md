# gpu-greenroom

Filesystem-backed GPU job queue with flock serialization and durable smoke evidence custody. One GPU job at a time, strict FIFO, crash-safe receipts, and default outputs that survive worker crashes and machine reboots.

## Problem

Heavy spatial-AI generation jobs (TRELLIS2MLX, Pixal3D, SuperMat, MoGe) share a single Mac GPU. Running multiple jobs concurrently risks kernel panics, Metal scheduler deadlocks, and OOM crashes. gpu-greenroom serializes GPU-bound work so only one job runs at a time.

Greenroom is also an evidence surface. A completed smoke is only useful if its primary artifacts, logs, route identity, seed, input image, and effective config remain inspectable after the box falls over. Do not put proof-bearing GLBs, render witnesses, or generated assets under `/tmp`, `/private/tmp`, `/var/tmp`, or cleanup worktrees. Those paths are disposable diagnostics, not smoke custody.

## Install

```bash
cd ~/dev/gpu-greenroom
uv pip install -e .
```

## Usage

```bash
# Submit a job; output auto-goes to durable queue_dir/outputs/<job-id>/
gpu-greenroom submit trellis2mlx /path/to/image.png

# Submit with explicit durable output dir
gpu-greenroom submit trellis2mlx /path/to/image.png ~/.local/state/gpu-greenroom/outputs/manual/trellis-smoke-2026-06-11/

# Submit with custom params; provenance is recorded in request/status/receipt
gpu-greenroom submit trellis2mlx /path/to/image.png -p seed=99 resolution=512 texture_size=4096

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

## Durable smoke output contract

For proof-bearing smokes, prefer omitting `output_dir`. Greenroom then assigns:

```text
<queue-dir>/outputs/<job-id>/
```

With the default queue directory, that is:

```text
~/.local/state/gpu-greenroom/outputs/<job-id>/
```

This is the normal path for Trellis2MLX/Pixal3D smoke runs where the GLB, texture outputs, renders, or witness files need to survive reboot and be reviewed later.

Explicit output directories are allowed, but they are caller custody. Before submitting a proof-bearing job with an explicit output path, make sure the path is durable and recorded in the source note, manifest, issue, or operator handoff that asked for the run. Good explicit bases look like:

```text
~/.local/state/gpu-greenroom/outputs/<project>/<run-id>/
/Users/noahlyons/dev/<project>/artifacts/<run-id>/
```

Bad evidence bases include `/tmp`, `/private/tmp`, `/var/tmp`, and worktrees under cleanup locations such as `/private/tmp/<repo>-<slice>`. Greenroom does not reject them because they are useful for quick disposable diagnostics, but it marks the job with `volatile_output` in `status.json` and `receipt.json`. Treat that warning as: this output is not acceptable as proof unless it has been copied or regenerated into a durable path before the volatile location disappears.

Logs and receipts are not replacements for primary artifacts. After a kernel panic or reboot, a volatile-output job may still leave enough queue metadata to prove route identity, input provenance, seed, timing, and approximate output stats, but the generated GLB or render may be gone. That is recovery evidence, not a completed smoke artifact.

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

If you override the queue directory for evidence-bearing work, choose a durable filesystem location. A queue under `/tmp` makes the queue records and default outputs volatile together.

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

The receipt records the effective route, cwd, defaults, timeout, ignored params, warnings, and output directory that actually ran. Use the receipt to verify seed provenance and image provenance, but inspect the output artifact itself before calling a visual smoke successful.

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
