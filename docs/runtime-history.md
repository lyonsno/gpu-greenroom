# Compatible Runtime History

Runtime history estimates command wall time from retained queue receipts. It
does not estimate queue arrival time, prove GPU-busy time, or certify a quiet
host. A slow observation remains evidence; foreground contention can make a
future run much slower than every historical run.

## Read-Only Queries

```sh
gpu-greenroom --queue-dir /durable/queue history refresh --index /durable/history.sqlite
gpu-greenroom --queue-dir /durable/queue history estimate --index /durable/history.sqlite --manifest command.json
```

The caller chooses the index path. Refresh reads only request/status/receipt
metadata, resolves duplicate/nested records through the shared reader, and
updates changed observations in SQLite. The index is a rebuildable query cache,
not queue authority. Repeating a refresh preserves unchanged entries. It binds
one queue root; reuse for another queue rejects. Queries open it read-only.

Every retained terminal record is indexed, including failures, incomplete or
zero clocks, and historical records without source binding. Only consistent
done/exit-zero records with positive clocks supply successful totals. The
ordinary execution clock includes startup, cleanup and any operator wait.
Performance-window command wall is returned separately, never substituted for
whole-slot occupancy. Legacy unbound history remains explicitly unverified;
it cannot establish automatic Quick eligibility.

## Producer Profiles

One producer integration supplies `runtime_profile` on a `command.v4` manifest.
Older CLIs reject that schema rather than silently discarding its meaning.
Profiled requests require `runtime-profile.v1` from the claiming worker.

```json
{
  "schema": "gpu-greenroom.runtime-profile.v1",
  "name": "example/finite-capture",
  "compatibility": "capture-v1",
  "backend": "webgpu",
  "device": "apple-m4-max",
  "host": "actual-host-name",
  "config": {"width": 512, "height": 512, "frames": 12, "cold_start": true},
  "output_argv_indices": [4],
  "bindings": [
    {"path": "/absolute/node", "sha256": "<executable-sha256>"},
    {"path": "/absolute/capture.mjs", "sha256": "<source-sha256>"}
  ]
}
```

The producer includes every relevant model/package identity, input size, work
count, precision, cold/warm lifecycle and routing setting in its config and
explicit command/environment. Bind the executable, entry script and imported
cost-affecting code/config files. Package identity may be a bound manifest;
querying need not hash model weights. The worker checks these files before and
after execution and records the actual argv/cwd/host. Backend/device strings
are producer declarations, not hardware-probe proof. The profile's evidence
ceiling is a source-bound command envelope on that host.

Compatibility is exact: all argv, cwd, environment overlay, input path, params,
performance-window mode, timeout and checkpoint mode remain in the key. Run
route labels do not. The only excluded argv values are declared output slots;
they must be absolute paths inside that request's explicit output directory.
Only genuine output destinations belong there, never model/input/config files.
Relative paths beneath the output root still distinguish different operations.
Changing configuration or source bytes creates a new key, not pooled history.
Profile definitions are separately approved by digest, including their output
slot semantics. Unknown additive profile labels carry no matching authority.

An owner can deliberately version compatible source implementations later;
this first implementation does not extrapolate or pool different source bytes.
It therefore prefers an explicit insufficient-data answer over a guessed
equivalence. Median/minimum/maximum and all excluded observations accompany
the evidence IDs. These are empirical observations, not calibrated probability
bounds. Predictive usefulness needs chronological held-out producer evidence.

## Automatic Quick Screen

Operator configuration can enable a measured profile without registering each
run-specific route. Dispatch policy v3 adds:

```json
{
  "schema": "gpu-greenroom.dispatch-policy.v3",
  "mode": "two-class",
  "quick_routes": {},
  "quick_job_types": {},
  "runtime_history": {
    "index": "/durable/history.sqlite",
    "profiles": {
      "example/finite-capture": {
        "profile_digest": "<runtime_history.definition_digest(profile)>",
        "minimum_samples": 3,
        "max_seconds": 30,
        "evidence": "/durable/profile-observations-and-owner-contract.json"
      }
    }
  }
}
```

The sample requirement and threshold are operator-selected screening settings,
not a statistically calibrated guarantee. The exact profile definition and
cost key must agree, enough compatible successful observations must exist, no
matching failed/unverified observation may be present, and every observed wall
time must satisfy the configured threshold. An explicitly normal request opts
out. A missed screen stays normal; explicit Quick without eligibility rejects.
Manual per-job controls retain their distinct operator authority.

The worker updates one indexed receipt after a completion, rather than rescanning
all history per job. Queries detect changed terminal namespace generations and
verify cited metadata digests; another/older writer makes the index stale until
explicit refresh. A previously stale cache cannot be made current by appending
one new receipt. Missing/corrupt/stale history lowers classification to normal.
Index failures do not erase workload completion; the worker writes a diagnostic.
Canonical metadata remains authoritative. An out-of-band edit to an unrelated
existing record requires refresh; this cache is not a filesystem change monitor.

Policy v3 requires a worker supporting both manual overrides and runtime
profiles; old claimants cannot advance it. Existing two-class alternation,
within-class FIFO, pause, leases, unknown ownership and the one execution flock
remain authoritative. A prediction never creates a timeout, signal or process
preemption. Slow completed jobs remain in the index and demote future matches.

## Finite Batch Producers

The shared Python `gpu_queue.cooperative.run_units` adapter wraps the existing
checkpoint protocol between completed cases:

```python
results = run_units(
    cases,
    perform=run_case_and_persist_result_descriptor,
    quiesce=release_owned_resources_and_confirm,
    continuation=build_same_owner_repo_route_command,
    on_submitted=register_successor_completion,
)
```

Cases are a finite JSON-described list. Each successful case returns a durable
JSON result descriptor. The saved cursor, result descriptors and case-list hash
are bound into the existing checkpoint. A successor resumes the next distinct
case; a different list rejects. With no managed context the same code executes
directly and does not self-enqueue. At the final case it finishes instead of
creating an empty successor.

The producer owns backend synchronization, child cleanup, numerical state,
durability/retention of referenced results, the successor command, and registered
completion delivery. The adapter owns cursor mechanics and cancellation checks.
The existing hook stages/commits registration before publication; the worker
verifies process-group exit before releasing the flock. Persistent models need
real supported checkpoint/release or completed independent cases; a screenshot
or authored setup is not evolved-state recovery. Resident interactive suspension
is a separate protocol. Compare save/reload cost with the short-job opportunity
before adopting a boundary.
