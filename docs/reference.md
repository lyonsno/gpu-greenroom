# Reference

Exhaustive CLI, on-disk layout, job type configuration, receipt schema, and
failure phases. The [README](../README.md) explains why the system is shaped
this way; this file is the manual.

## CLI

```bash
# Jobs
gpu-greenroom submit <job-type> <input-path> [output-dir] [-p key=value ...] [--cwd DIR]
gpu-greenroom list [-s pending|running|done|failed|cancelled]
gpu-greenroom status <job-id>
gpu-greenroom cancel <job-id>          # pending jobs only

# Exact-argv jobs (no job type, no substitution)
gpu-greenroom submit-command --agent-id NAME --repo-root DIR --cwd DIR --route-identity LABEL \
  [--env K=V ...] [--output-dir DIR] [--timeout SECONDS] -- <argv...>
gpu-greenroom submit-command --manifest /path/to/gpu-greenroom.command.json

# Worker
gpu-greenroom worker [--poll SECONDS]  # runs jobs sequentially, reloads job_types.json every poll
gpu-greenroom pause --owner NAME --epoch ID     # finishes the current job, then waits
gpu-greenroom resume --owner NAME --epoch ID    # epoch must match the observed pause
gpu-greenroom recover                  # move running jobs whose worker PID is gone to failed/
gpu-greenroom doctor --json            # executable discovery, imports, queue writes, dispatch availability

# Retention
gpu-greenroom gc --dry-run [--grace-hours 72] [--authority TEXT] [--no-size] [--json]
gpu-greenroom gc --apply --epoch ID --owner NAME
gpu-greenroom retain NAME --owner NAME --reason TEXT [--until ISO-DATETIME]
gpu-greenroom retain --list
gpu-greenroom retain NAME --unpin

# Operator console
gpu-greenroom operator --port 8765                       # one-shot; prints a token URL
gpu-greenroom operator --local-operator --admission-control --identity-label Agent --port 8766

# Several queues as one contention class
gpu-greenroom queues register NAME --queue-dir DIR --contention-class CLASS
gpu-greenroom queues status
gpu-greenroom queues pause  --contention-class CLASS --owner NAME --epoch ID
gpu-greenroom queues resume --contention-class CLASS --owner NAME --epoch ID

# Cooperative external leases
gpu-greenroom lease claim \
  --owner render-holder \
  --agent-id render-holder \
  --repo-root /path/to/repo \
  --pid "$PID" \
  --process-group "$PGID" \
  --effective-route "chrome --enable-unsafe-webgpu app.html" \
  --backend metal \
  --device mps:0 \
  --profile interactive-render \
  --supports-checkpoints \
  --ttl-seconds 300
gpu-greenroom lease renew <lease-id> [--interruptible|--not-interruptible] [--ttl-seconds N]
gpu-greenroom lease status
gpu-greenroom lease release <lease-id> --released-by render-holder --reason "capture checkpoint complete"

# Bump requests (asking a lease holder for a handoff window)
gpu-greenroom bump request \
  --bump-id resident-cold-load \
  --requester resident-loader \
  --agent-id resident-loader \
  --repo-root /path/to/repo \
  --intended-route "python load_model.py --device mps" \
  --workload-class cold-model-load \
  --memory-pressure 3.3GB \
  --estimated-occupancy 90s \
  --full-quiescence-required \
  --reason "prepare resident model before capture" \
  --callback-address "file:///tmp/resident-callback"
gpu-greenroom bump list [--status pending|grant_pending_checkpoint|granted|declined|closed]
gpu-greenroom bump grant <bump-id> --granted-by render-holder --checkpoint capture-42 [--quiescence-confirmed]
gpu-greenroom bump decline <bump-id> --declined-by render-holder --reason "training cannot checkpoint"
gpu-greenroom bump wait <bump-id> [--timeout SECONDS]

# After a grant, the requester claims its own lease bound to the bump:
gpu-greenroom lease claim ... --handoff-bump-id resident-cold-load
```

Every command accepts `--queue-dir DIR`; the default is `GPU_GREENROOM_DIR`
or `~/.local/state/gpu-greenroom`.

## Queue directory layout

```
~/.local/state/gpu-greenroom/
  gpu.lock              # the execution mutex: flock(LOCK_EX | LOCK_NB)
  coordination.lock     # short-lived lock for JSON state transitions; never execution authority
  paused                # present while the queue is paused
  job_types.json        # optional job type configs, hot-reloaded every poll
  outputs/              # durable output dirs for jobs submitted without an explicit output_dir
  leases/
    current.json        # the current external lease (active, handoff, released, or ownership_unknown)
    receipts/           # claim, renew, release, handoff, and ownership-unknown receipts
  bumps/
    <bump-id>.json      # inbound bump request state
    receipts/           # request, grant, and decline receipts
  events/               # filesystem-watch wake events for `bump wait`
  pending/<job-id>/     # request.json, status.json
  running/<job-id>/     # at most one; adds stdout.log, stderr.log
  done/<job-id>/        # adds receipt.json
  failed/<job-id>/      # same, with failure_phase set
  cancelled/<job-id>/
```

A job's state is the directory it lives in. Transitions are `rename(2)`
moves, so a crash mid-transition leaves the job in exactly one place. A job
whose process group cannot be confirmed quiesced stays in `running/` with
`finished_at: null`, a `*_quiescence_unresolved` failure phase, the warning
`ownership_unknown:process_group_live`, and a receipt whose `status` is
`ownership_unknown`; `recover` and operator action are the only exits.

## Job type configuration

`job_types.json` maps a job type name to either a bare command list or a
rich config:

| Field | Required | Description |
|---|---|---|
| `cmd` | yes | Command as a list of strings. `{input_path}`, `{output_dir}`, and any param key are substituted. Substitution is single-pass: a param value containing `{input_path}` stays literal. |
| `cwd` | no | Working directory for the subprocess. Overridable per job with `submit --cwd`. |
| `env` | no | Variables merged over the worker's environment. |
| `defaults` | no | Default param values. User params override defaults; `input_path` and `output_dir` always win. |
| `timeout` | no | Seconds. Absent or `null` means no timeout. |
| `source_attestation` | no | `"git-clean-input"`: before launch, hash the input file and record its repository root, commit, and dirty state; refuse to run if the tree is dirty; re-check after the run and fail the job if the input or commit changed underneath it. |
| `runtime_identity_cmd` | no | Command list run before the job. Its stdout, stderr, exit code, and the hash of its executable are recorded in the receipt. Non-zero exit fails the job in the `runtime_identity` phase. Use it to prove which interpreter, backend, or device the job actually ran on. |
| `output_class` | no | Retention class for this job type's outputs: `final`, `witness`, or `intermediate`. Absent means unclassified, which `gc` reports and never collects. |
| `artifact_manifest` | no | Glob patterns relative to `output_dir`. After a successful run, every match is hashed into the receipt. A pattern that matches nothing fails the job. |

Params the template does not consume are recorded in the receipt as
`ignored_params` instead of being silently dropped.

`source_attestation`, `runtime_identity_cmd`, and `artifact_manifest` apply
only to job types declared here. Structured command jobs (`submit-command`)
carry an exact argv and get none of the three; their receipts record those
fields as `null`.

Example:

```json
{
  "trellis2mlx": {
    "cmd": [
      "/path/to/trellis2mlx/.venv/bin/python", "-u", "generate.py",
      "--image", "{input_path}",
      "--output", "{output_dir}/output.glb",
      "--seed", "{seed}",
      "--resolution", "{resolution}"
    ],
    "cwd": "/path/to/trellis2mlx",
    "env": {"PYTHONPATH": ".", "ATTN_BACKEND": "sdpa"},
    "defaults": {"seed": "42", "resolution": "512"},
    "runtime_identity_cmd": ["/path/to/trellis2mlx/.venv/bin/python", "-c", "import mlx.core as mx; print(mx.default_device())"],
    "artifact_manifest": ["*.glb"]
  },
  "echo": ["echo", "{input_path}"]
}
```

See [`job_types.example.json`](../job_types.example.json) for more.

## Receipt schema

Every job that started, or was stopped by timeout, shutdown, or an
operator, gets a `receipt.json`. An unknown job type fails at `dispatch`
with a status record and no receipt. `status` is `done`, `failed`, or
`ownership_unknown` (the job is still in `running/`):

```json
{
  "job_id": "ab8647e17eb0",
  "job_type": "sf3d",
  "status": "done",
  "input_path": "/inputs/skull.png",
  "output_dir": "/outputs/skull/sf3d",
  "repo_root": null,
  "requested_route": null,
  "effective_route": "/path/to/sf3d/.venv/bin/python -u run_greenroom.py --image /inputs/skull.png --output-dir /outputs/skull/sf3d --texture-resolution 1024 --dtype float16",
  "effective_argv": ["/path/to/sf3d/.venv/bin/python", "-u", "run_greenroom.py", "--image", "/inputs/skull.png", "--output-dir", "/outputs/skull/sf3d", "--texture-resolution", "1024", "--dtype", "float16"],
  "effective_cwd": "/path/to/sf3d",
  "effective_env": {"PYTHONPATH": ".", "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
  "environment_inheritance": "worker-plus-overlay",
  "effective_defaults": {"texture_resolution": "1024", "dtype": "float16"},
  "effective_timeout": null,
  "worker_pid": 89531,
  "child_pid": 90112,
  "child_process_group": 90112,
  "child_start_identity": "Tue Sep 23 05:02:11 2026",
  "ignored_params": null,
  "started_at": 1787117889.52,
  "finished_at": 1787117921.06,
  "exit_code": 0,
  "failure_phase": null,
  "error_message": null,
  "warnings": null,
  "request_path": "/queue/done/ab8647e17eb0/request.json",
  "stdout_path": "/queue/done/ab8647e17eb0/stdout.log",
  "stderr_path": "/queue/done/ab8647e17eb0/stderr.log",
  "worker": {"pid": 89531, "capabilities": ["structured-command.v1"], "source": {"…": "…"}},
  "input_artifact": {"path": "/inputs/skull.png", "sha256": "…", "size_bytes": 412331},
  "source_attestation": {"mode": "git-clean-input", "root": "/inputs", "commit": "…", "clean_before": true, "clean_after": true, "…": "…"},
  "runtime_identity": {"command": ["…"], "exit_code": 0, "stdout": "Device(gpu, 0)\n", "stderr": "", "executable": {"path": "…", "sha256": "…", "size_bytes": 0}},
  "artifact_manifest": [{"path": "mesh.glb", "sha256": "…", "size_bytes": 8812031}]
}
```

`repo_root` and `requested_route` are set for structured command jobs.
`worker` records the claiming worker's PID, capabilities, and source
checkout (path, commit, dirty state). `input_artifact`,
`source_attestation`, `runtime_identity`, and `artifact_manifest` are
`null` unless the job type asked for them.

Successful jobs also get a `metadata.json` written into `output_dir`
(name, job type, params, output file list, duration) so asset browsers can
display results without reading the queue.

## Failure phases

| Phase | Meaning |
|---|---|
| `dispatch` | Unknown job type, or a structured command with an empty argv; status record only, no receipt |
| `source_preflight` | Input attestation failed: missing input, dirty tree, or unsupported mode |
| `runtime_identity` | The identity probe failed or exited non-zero |
| `launch` | Subprocess failed to start |
| `execution` | Subprocess exited non-zero |
| `timeout` | Subprocess exceeded the configured timeout and its process group was quiesced |
| `worker_shutdown` | The worker was asked to stop; the owned process group was quiesced |
| `source_postflight` | Input or its commit changed during execution |
| `metadata` | Could not write `metadata.json` |
| `artifact_manifest` | A manifest pattern matched nothing or a non-file |
| `stale_recovery` | Worker died without cleanup; moved to `failed/` by `recover` |
| `completion_quiescence_unresolved` | Leader exited but the process group could not be confirmed dead; job stays in `running/`, receipt status `ownership_unknown` |
| `timeout_quiescence_unresolved` | Timed out and the process group could not be quiesced; same handling |
| `worker_shutdown_quiescence_unresolved` | Worker stopped and the process group could not be quiesced; same handling |
| `launch_quiescence_unresolved` | Launch failed and a process group is still live; same handling |

## Lease and bump states

Lease `lifecycle_state`:

| State | Meaning |
|---|---|
| `active` | An external process owns the GPU. Worker dispatch is blocked. |
| `handoff` | A bump was granted; the requester may claim a new lease bound to that bump. Worker dispatch stays blocked. |
| `released` | The holder released. Worker dispatch resumes. |
| `ownership_unknown` | TTL expired or the holder PID died without a release receipt. Worker dispatch stays blocked until someone recovers authority explicitly. |

Bump `status`:

| State | Meaning |
|---|---|
| `pending` | Waiting for the holder to answer |
| `grant_pending_checkpoint` | Granted, but the holder has not reached its checkpoint; `lease release` completes it |
| `granted` | Handoff window is open; `bump wait` returns |
| `declined` | Holder declined with a reason |
| `closed` | Administratively closed |

## Structured command jobs

`submit-command` admits a repository-local accelerator command without
adding a global job type. It accepts flags or a caller-owned
`gpu-greenroom.command.v1` JSON manifest:

```json
{
  "schema": "gpu-greenroom.command.v1",
  "agent_id": "example-agent",
  "repo_root": "/path/to/repo",
  "cwd": "/path/to/repo",
  "env": {"BACKEND": "mlx"},
  "output_dir": "/durable/results/grid32",
  "route_identity": "assays/grid32-bounded",
  "argv": ["/path/to/repo/.venv/bin/python", "-u", "scripts/grid32.py"],
  "timeout": null
}
```

`agent_id` is the exact owning identity declared by the caller. It is
optional for compatibility with historical requests, which the console
labels `not recorded`; it is never inferred from a route, worktree, or
environment. A manifest and `--agent-id` cannot be combined. `argv` runs
with `shell=False`; nothing is reparsed and no template substitution is
applied. `timeout: null` means no Greenroom-authored timeout. The
environment is the worker environment plus the recorded overlay.

Submission returns JSON with the job id, queue directory, request path,
output directory, and requested route. Receipts preserve requested route,
exact effective argv, repo root, cwd, environment overlay, timeout,
stdout/stderr paths, exit code, failure phase, the claiming worker's PID,
source checkout, commit and dirty state, and effective capabilities. A
launch failure still leaves request, status, logs, and a receipt.
Manifest validation failures are written under `submission-failures/` and
returned as structured stderr.

Structured requests require the `structured-command.v1` worker
capability. A worker compares request requirements with its effective
capabilities while holding `gpu.lock` and before moving the FIFO head to
`running`. An incapable worker leaves the request untouched and does not
skip to younger compatible work. Workers support every capability their
code implements unless `GPU_GREENROOM_WORKER_CAPABILITIES` supplies a
comma-separated deployment override.

## Registered queues and execution-start pause

The queue registry stores only one-time adapter identity: name, queue
directory, contention class, and adapter kind. Its path comes from
`--registry` or `GPU_GREENROOM_REGISTRY`, default
`~/.local/state/gpu-greenroom/queues.json`.

`queues status` reads pending, running, and paused state from each native
queue; it copies nothing into the registry. `queues pause` preflights every
selected queue and then creates each queue's native `paused` marker.
Submission stays open, running jobs finish normally, and workers cannot
move pending work to running until `queues resume` removes the markers.
Control actions write receipts under `queue-control-receipts/`. The marker
records owner, epoch, requested and effective times, queue identity, and
contention class. Resume requires the exact epoch, so a stale controller
cannot remove a newer pause.

The per-queue lock remains the only execution mutex. Aggregate control
authorizes no execution and adds no check to ordinary dispatch; the
existing coordination lock linearizes marker creation against the final
pending-to-running transition and releases before the workload runs.

## Operator console

`gpu-greenroom operator` serves a small web console on loopback. In
one-shot mode it prints a URL carrying a bearer token. With
`--local-operator` it runs at a stable URL and seats the current process
credential when the page loads, so a launchd-supervised console survives
restarts. `--admission-control` enables receipted pause and resume from the
page; `--identity-label` sets the word used for the owner column. Every
control action carries a request id and lands as a receipt under
`operator-actions/`, and an action interrupted between marker mutation and
completion is never replayed as successful.

## Retention and garbage collection

`gc` manages only direct children of the queue's own `outputs/`. Caller-owned
output directories elsewhere are recorded in receipts and never touched.

Each entry gets a class from the longest-TTL class any of its jobs declared
(`output_class` on the job type, or `output_class` in a structured command
manifest or `--output-class`), an owner from the job's `agent_id`, and an
age from the newest job's `finished_at` (directory mtime when no job record
exists). TTL by class: `intermediate` 30 days, `witness` 60 days, `final`
180 days. An entry is a candidate when it is past its TTL, not pinned, and
not referenced by a pending or running job. `unclassified` entries are
listed with their size and never collected.

`gc --dry-run` writes `gc-candidates.json` with an `epoch`, totals, every
row, and `apply_not_before` (now plus the grace window, 72 hours by default).
`gc --apply --epoch <epoch> --owner <who>` refuses when the epoch does not
match the current candidate list, when the grace window has not elapsed, or
when any listed path resolves outside `outputs/`; otherwise it re-checks
each row for pins and active jobs, writes
`gc-receipts/<epoch>/<name>.json` (class, owner, size, age, job ids, and any
`artifact_manifest` and `input_artifact` digests from those jobs' receipts),
removes the directory, then stamps `deleted_at` on the receipt. A summary
lands at `gc-receipts/<epoch>/_summary.json`.

`retain <name> --owner --reason [--until]` writes `retention/pins.json`; a
pinned entry is never a candidate until its `until` passes or it is
unpinned. Pins are the only retention authority; citation counts in reports
are diagnostics.

## Reviewing a new job type

A route is the command that actually works on this machine, not the
upstream default that happens to launch. Before trusting a new job type,
record:

- **Known-good local runner checked:** which wrapper, README, or prior receipt
  proved this command works here.
- **Effective env/device/backend preserved:** the exact env or args for
  device, dtype, backend, and fallback knobs.
- **First receipt proves backend/device:** the stdout or `runtime_identity`
  snippet showing the effective route before a heavy run is accepted.
- **Heavy run accepted before proof:** `no` by default.

A started subprocess is not evidence until its receipt proves the route.
TRELLIS.2 on Apple Silicon is the standing example: the Mac route needs
`ATTN_BACKEND=sdpa` and `SPARSE_ATTN_BACKEND=sdpa`; a job that launches with
upstream `flash_attn` identity is a route failure, not a model failure.
