# Producer checkpoint declarations

A repository can declare exact checkpoint-capable script entrypoints in
`.gpu-greenroom-producer.json`:

```json
{
  "schema": "gpu-greenroom.producer-checkpoints.v1",
  "entrypoints": ["generate.py"]
}
```

Normal structured submissions inherit checkpoint admission for a matching
entrypoint. Normal typed submissions resolve their configured defaults, output,
executable and environment into an exact command. The receipt retains the typed
defaults and ignored parameters. Unrelated modules, inline commands and shell
wrappers do not inherit a neighboring declaration. Malformed declarations and
entrypoints escaping their source root fail visibly.
Typed receipt metadata requires `defaults`, `ignored_params` and `substitutions`
objects. Unknown additive fields are preserved. Malformed metadata is rejected
at admission; a malformed persisted request fails worker preflight with a terminal
receipt before launching a child.

The request binds declaration and entrypoint contents under
`params._producer_checkpoint`. The worker checks the binding before executing
the child. Continuations preserve it. This declaration asserts the producer's
support; it does not implement saving or quiescence. Producers must use the
existing SDK to save complete progress, release their resources and stage a
successor. The worker still waits for process-group exit before dispatching
other work.

Caller attribution and checkpoint custody are distinct. An omitted `agent_id`
remains omitted. Such a declared job receives `job:<original-job-id>` checkpoint
custody, preserved through successors and retention pins. It requires worker
capability `producer-owned-checkpoints.v1`; an older worker cannot claim it.
Attributed jobs retain their existing caller-owned checkpoint contract.

Prepared commands retain their input-bound validation. Admission binds the
producer before validating the complete request. At handoff the SDK first
checks the original validation against current inputs, then revalidates the
successor with the original CPU preflight and complete declared input list.
Changed original inputs prevent publication of a successor. Such jobs require
`prepared-checkpoint-continuation.v1`. Existing old validation records lacking
a necessary preflight specification fail visibly rather than weakening the
validation contract.

Typed definitions with separate source-attestation/runtime-identity/artifact
contracts currently require an adapter preserving those contracts; automatic
freezing does not silently discard them. Existing queued requests are immutable
and do not acquire these capabilities retroactively. Keep in-flight source
checkouts unchanged during runtime deployment.
