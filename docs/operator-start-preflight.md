# Operator Start And CPU Preflight

An operator-dependent session can be prepared without reserving the GPU. Its
CPU-side command declaration waits in smoke-requests, not pending or running.
Start enqueues the exact prepared job once; ordinary pause, lease, capability,
class selection and execution-flock gates then apply. Waiting for a human is
not itself an execution job. Finite automatic preparation jobs retain their path.

Publish through the local CLI with an optional operator_command:

```json
{
  "schema": "gpu-greenroom.interactive-smoke.v1",
  "id": "9c0a03f6-b2d8-43f4-b7b7-5e848e733661",
  "kind": "interactive-smoke",
  "source": {"agent_id": "example-owner", "repo_root": "/source/worktree"},
  "title": "Voice session",
  "prompt": "Start the session, then report your observation.",
  "url": "http://127.0.0.1:8766/",
  "availability": "prepared",
  "availability_note": "Inputs prepared; GPU not acquired.",
  "operator_command": {
    "schema": "gpu-greenroom.command.v3",
    "agent_id": "example-owner",
    "repo_root": "/source/worktree",
    "cwd": "/source/worktree",
    "route_identity": "voice/profile-a",
    "argv": ["/source/worktree/.venv/bin/python", "/source/worktree/live.py"]
  }
}
```

```sh
gpu-greenroom --queue-dir /durable/queue smoke-request submit request.json
gpu-greenroom --queue-dir /durable/queue smoke-request get <request-id>
```

The stored prepared job ID is available before Start for completion registration.
The producer registers that ID's terminal delivery before publishing the human
handoff. The monitor's Start session button activates it. Headless consumers use
smoke-request start with the exact request digest. Repeated/concurrent starts
and recovery after enqueue-before-activation-record preserve the same job ID.
An operator response is unavailable before the prepared execution finishes.

HTTP Start accepts only a known request identity and digest. HTTP creation cannot
publish commands, and HTTP reads hide argv/environment/preflight details. The
operator credential authorizes this action; actor attribution remains
unverified-caller, as with existing smoke responses.

## Early Validation

Command v1/v2 syntax, directory and executable checks run before submission.
Known registered-type CLI submissions reject unknown types, malformed registry,
missing command/executable or cwd before enqueue. These static checks do not
establish backend initialization or output. Arbitrary job arguments cannot be
validated semantically without the application's declared contract.

Command v3 can include preflight with argv and an explicit list of absolute input
file paths. The validator receives JSON on stdin containing schema
gpu-greenroom.preflight-input.v1, request_digest and the normalized request.
GPU_GREENROOM_VALIDATION_MODE is cpu-only; GPU_GREENROOM_VALIDATION_OUTPUT names
the result file. The validator writes:

```json
{"schema":"gpu-greenroom.preflight-result.v1","valid":true,"mode":"cpu-only","request_digest":"<provided digest>"}
```

Exit zero alone, missing/blank/malformed output, wrong mode or wrong command
identity is not a pass. Additional non-authority result fields remain compatible.
The validator contract is CPU-only; this declaration is not an OS GPU sandbox or
a hardware observation. Use a pure metadata/config checker, not a model-loading
entry point. Logs and failure-phase receipts remain under validation-reports.

Bindings cover command bytes, executable/main script and the declared inputs,
including validator code. Declare semantic dependencies such as imported modules
and configuration explicitly. This is not an attestation of an entire repository,
model checkpoint or runtime. Start/submission rechecks the bindings; dispatch
rechecks again before running. Changed data fails validation-invalidated without
a child, started_at or fairness turn. Runtime/backend checks still run at execution.
New jobs require command-preflight.v1; old workers leave them unclaimed.
Python direct-script forms support common flags such as -u and -B and relative
paths resolved from command cwd. Node and shell direct scripts and inline source
are supported; unfamiliar interpreter options/module loading are refused with
the direct-script or inline-wrapper continuation. Imported code still requires
explicit inputs. Command v3 with cooperative_checkpoint is refused before any
save or execution until successor validation is supported; the separate v2
checkpoint protocol is unchanged.

Queue retention follows unconsumed prepared smoke plans and pending/running
validation bindings; unreadable relevant records withhold collection. An external
plan saved to an invoker-selected path is outside queue discovery: that invoker
owns retaining its dependencies until submission or retirement. Creating such a
file does not claim an automatic queue pin.

```sh
gpu-greenroom --queue-dir /durable/queue prepare-command --manifest command-v3.json --output plan.json
gpu-greenroom --queue-dir /durable/queue submit-prepared plan.json
```

The invoker owns the plan path. Validation preserves a command without enqueueing;
submit-prepared reuses its job ID. Do not edit the bound plan to change work;
prepare a new command. The GPU mutex remains the only execution authority.

## Producer Boundary

The live entry point starts GPU load/warmup only after this admission. Keep its
audio/privacy controls explicit. Between conversations, a producer must reach a
supported safe release boundary and return to CPU-side waiting. A field saying
awaiting_operator does not authorize release of a loaded model. For RAON's
current implementation, process exit and a fresh Start is the source-owned
integration recommendation; deleting the session alone retains the model.

## Finished Sessions

Process completion does not record that the operator tried the application.
Finished cards show the native end time and exit status, remove live directions
and terminal navigation, and ask whether the operator tried or missed the session.
Responses retain optional participation (`tried` or `not-tried`) separately from
their text; legacy responses make no participation claim. Needs attention counts
unresolved requests, not running applications.

View last run reads the exact linked native job's complete stdout/stderr. Producers
may add retained audio through the existing local configuration command:

```json
{"review_artifacts":[{"path":"conversation/playback.wav","label":"Recorded playback"}]}
```

Paths must be relative to that job's recorded output directory. Configuration
binds the file bytes; missing, changed, empty or escaped files are unavailable,
not playable evidence. Audio requires the same bearer credential as the API.
HTTP accepts a published artifact index, never a caller-selected file path.

Request another session records an idempotent `repeat_request` in the existing
request. It grants no execution authority. The producer consumes this receipt,
prepares a distinct request/job with an independent output directory, registers
its completion delivery, and links it through local configuration:

```json
{"next_request_id":"<new prepared request UUID>"}
```

The finished card then offers Start new session against that new request's digest;
after activation it offers View new session. Original request, finished job and
retained output remain unchanged. Preparation itself never enqueues a GPU job.
Missing or invalid successor preparation is explicit; the UI cannot guess argv,
reuse a finished job, or silently restart a model. Producers own repeat-request
consumption and publication of the successor, just as they own first preparation.
