# Assay Runtime Projection

Status: architecture proposal; the interfaces below are not implemented.

## Consumer And Decision

An agent preparing an assay needs a cheap answer to: how much execution does
this plan buy, how long might the first result take, and what makes those
numbers uncertain? The operator needs the same account when choosing between
assays. A local deterministic query should answer recurring cases without a
model call, archive search, benchmark launch, or new monitoring service.

Foreground rendering can stretch execution dramatically and nonlinearly.
The useful output is a conditional estimate, not one duration that mixes work,
queueing, operator participation, and interference. Estimates remain advisory;
execution timeouts, service classes, ordering, leases, and pause authority keep
their existing contracts.

## Separate The Clocks

Expose these quantities independently, with unavailable values left unknown:

- Queue/admission delay: submission to dispatch, including pause and leases.
- Operator wait: confirmation or interactive participation, not computation.
- Command wall time: actual command launch to exit, including startup and I/O.
- Useful stage time: producer-observed model load, warmup, compute, readback,
  capture, and postprocessing where instrumentation actually distinguishes them.
- Execution-slot occupancy: the interval under the execution mutex. This is
  neither GPU-busy time nor necessarily command wall time.

Current request/status/receipt records supply submission, start and finish
times. Performance-window receipts additionally separate operator wait and
command wall time. Older records generally cannot separate these phases.
Missing effective device/backend identity cannot be recovered from the
Greenroom worker commit or a requested route label. The console currently says
GPU hold time is not recorded; this proposal preserves that distinction.

## Components

### 1. Reusable Cost Profiles

A producer registers a small versioned profile beside its normal preparation
code. It identifies the workload family, extracts cost-relevant fields from
the existing manifest, and names the work units. Agents use that integration;
they do not rewrite a cost description for every submission.

Examples of relevant fields are model/weights identity, effective backend and
device, precision, input dimensions, batch size, simulation extent, frames or
steps, process count, and cold/warm model lifetime. The owner decides which
apply. An output directory is not a workload difference. A seed is irrelevant
only when the owner has established that for the family.

Keep exact execution identity separate from performance compatibility. Preserve
source revision and effective configuration on every observation. A profile
can explicitly name compatible revisions; a performance-relevant change needs
a new compatibility version. No automatic pooling across revisions, backends,
or devices merely because commands have similar names.

The generic path supports exact-configuration history without a custom
profile. This is useful for recurring jobs but does not justify extrapolation
to different work sizes. Profiles add extraction and supported scaling, not a
second submission or execution system.

### 2. Small Incremental Observation Index

Normalize request/receipt metadata into a caller-selected local SQLite index
using the standard library. Canonical job records remain authoritative; the
index is a rebuildable query accelerator, not an execution registry. Historical
backfill is explicit. Subsequent queries ingest changed metadata rather than
rescan model outputs, browser profiles, agent conversations, or every log.

Each observation retains queue/job identity, source paths and digests,
effective route/configuration, outcome, phase boundaries, work completed,
condition evidence, and the derivation used for each duration. Duplicate
observations and conflicting terminal records cannot silently count twice.
Source absence, cache freshness and incomplete identity remain visible.

Successful work contributes completed-duration observations. Failed and
interrupted work remains available with its phase and completed units; elapsed
time before failure is not a successful total or a zero-cost sample. A failed
run that terminated at a known point may supply a censored observation only
when the stop mechanism supports that interpretation. Live operator waits
have no automatic finite upper bound.

### 3. Explicit Condition Evidence

Record condition events over the run rather than treating the entire job as
quiet or busy from one snapshot. Reuse existing lease, pause, foreground-smoke,
and producer timing/progress events. Add host pressure observations only where
an existing low-cost collector supplies them or a concrete prediction failure
justifies new collection.

Condition labels distinguish observed foreground overlap, operator-reported
contention, memory/thermal pressure where observed, an acknowledged measurement
window, and unknown conditions. Acknowledgement is not verified quietness;
an absent managed lease does not prove no foreground GPU work exists.

Slow observations are preserved. Timing alone does not prove contamination,
and a fast observation does not prove quietness. Use comparable condition
strata instead of quietly dropping slow samples or absorbing them into the
normal baseline. If only unknown-condition history exists, label its estimate
mixed-condition rather than quiet-box.

### 4. Read-Only Projection

The proposed query accepts an existing prepared command or a small assay plan,
plus explicit queue/index/profile paths, and returns human-readable and JSON
forms of the same result. It performs no new GPU work. Missing evidence returns
an insufficient-data result with its specific reason and the cheapest useful
next observation, not an invented default or a mandatory operator gate.

Return these fields conceptually:

- Requested plan and effective workload match, including mismatches/unknowns.
- Central estimate and empirical spread for the named condition class.
- Evidence type: exact repeat, owner-declared compatible history, supported
  interpolation, explicit owner model, or insufficient data.
- Sample count, dates, spread, source references and exclusion reasons.
- Current-condition assessment and any distinct contended estimate.
- Queue/start forecast separately, with snapshot time and unresolved blockers.

For a future job, the target route is requested, not already effective. Compare
it against observed sample identities and say which target fields are still
unverified. At dispatch, a different effective backend/device/configuration
invalidates that prediction; record the mismatch without manufacturing an
effective identity or turning an estimate into execution authority.

Start with exact matches and simple empirical summaries. A single observation
is one observed duration, not a calibrated interval. A range of observed runs
is not a promised percentile. Calibrated coverage claims need held-out evidence.
Recent slow observations under unknown conditions remain consequential evidence
of uncertainty rather than automatically removable outliers.

For supported scaling, use a producer-specific model such as startup plus
units times cost per unit. Validate its useful size range and keep cold start,
warmup and repeated model loading explicit. Memory-pressure cliffs, nonlinear
resolution changes and sparse-work changes invalidate naive linear scaling.
Outside the supported range, return a labeled extrapolation or no estimate.

There is no shop-wide contention multiplier and no division by GPU-utilization
percentage. Report a contended range only when comparable overlap evidence
supports it. Otherwise keep the normal-condition work estimate and state that
current wall time has no defensible upper estimate under observed contention.
The result can still support assay selection without pretending to know finish
time.

### 5. Assay Composition And Progress

A plan may contain repeated cases, distinct cold processes and dependent
stages. The extractor expands counts and lifecycle, not a Cartesian product of
guessed settings. Distinguish total serialized work from the critical path;
do not multiply a warm per-case time by all cases while forgetting repeated
loads, or assume every case pays a cold load within one persistent process.

Keep component estimates and unsupported phases visible. Summing central
estimates can provide a planning total. Summed marginal percentiles do not
become a calibrated whole-assay percentile, especially when one foreground
smoke affects many cases together.

When producer progress records completed units and stage identity, compare
observed throughput to the relevant baseline and revise remaining-work time.
Separate a slow run making progress from a stalled or operator-waiting run.
Elapsed wall time alone is not percent complete. Progress remains useful
without a remaining-time estimate when throughput is unstable.

## Illustrative Profile And Plan

This is a proposed data shape, not a measured result or a new runnable command:

```json
{
  "schema": "gpu-greenroom.assay-cost-plan.v0",
  "profile": "example-render-capture.v1",
  "request_manifest": "/path/to/prepared-command.json",
  "work": {
    "cases": 8,
    "frames_per_case": 120,
    "processes": 1,
    "model_lifetime": "shared-within-process"
  },
  "condition": "current-host"
}
```

The profile extracts route/backend/device/shape from the normal manifest and
uses the counts and process lifetime to compose this plan. Unknown preparation
or model-load time stays unknown. A plain prepared command can use exact-repeat
history without this extra plan wrapper. Invocation inputs, persisted plan,
query output and derived index remain distinct, caller-owned paths.

## First Implementation Slice

Build only the observation normalizer, incremental index and deterministic
exact-match query, with the reusable profile interface. Reuse current receipt
fields and the explicit performance-window command clock. Return sample counts,
empirical durations, condition provenance and missing evidence; do not infer
clean historical baselines. This gives agents cheap observed-repeat estimates
before introducing a host monitor or a learned predictor.

Exercise it through one existing browser assay producer and one cold/warm
model producer, preserving each owner's implementation authority. The first
consumer gate is an agent estimating an actual prepared plan and understanding
both the result and its missing coverage without manually reading history.
No new GPU assay is required to establish read/query mechanics.

Then add useful stage/progress joins and an owner-supported scaling rule for
those consumers. Live queue ETA is a later projection over the actual dispatch
policy and current pause/lease/operator state, not a parallel scheduler.
Unresolved human participation or ownership means start time remains unknown.

## Evidence For The First Slice

Use observed request/receipt fixtures from effective routes and schema versions,
alongside synthetic negative controls. Acceptance must expose wrong backend or
device matching, incompatible source/configuration, cold/warm confusion,
operator wait counted as compute, duplicate/stale/conflicting receipts,
partial or failed runs treated as successful totals, and missing timing fields
silently filled by defaults. Additive unknown fields remain compatible.

For contention, replay a preserved observation series containing fast and slow
cases with explicit overlapping conditions. Adding a contended run must preserve
it and its effect on the appropriate estimate without silently redefining a
verified baseline or inventing causality from duration. Unknown-condition slow
history must remain visible in uncertainty. Existing CPU/native fixtures can
prove operator-wait subtraction without claiming real GPU timing.

Before claiming useful predictive accuracy, perform chronological holdout on
retained comparable runs: predict using only evidence available beforehand,
then compare against the observed result. Report signed error, absolute error,
range coverage where claimed, and insufficient-data rate by workload and
condition class. A model that always refuses has not demonstrated utility.
Planning-slice documentation and replay design do not themselves establish
estimator implementation or predictive coverage.
