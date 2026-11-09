"""Bind a producer's exact entrypoint declaration during ordinary admission."""

import json
from pathlib import Path
import re

from . import admission


PARAM = "_producer_checkpoint"
OWNER_CAPABILITY = "producer-owned-checkpoints.v1"


def checkpoint_owner(request):
    """Checkpoint custody is distinct from an attributed caller identity."""
    if isinstance(request.agent_id, str) and request.agent_id.strip():
        return request.agent_id
    binding = request.params.get(PARAM, {})
    owner = binding.get("owner") if isinstance(binding, dict) else None
    if not isinstance(owner, str) or not re.fullmatch(r"job:[A-Za-z0-9_-]+", owner):
        raise ValueError("checkpoint continuation requires caller or job-owned checkpoint custody")
    return owner


def discover(repo_root, argv, cwd):
    if not repo_root or not argv or not cwd:
        return None
    root = Path(repo_root).resolve()
    manifest = root / ".gpu-greenroom-producer.json"
    if not manifest.exists():
        return None
    value = json.loads(manifest.read_text())
    entries = value.get("entrypoints") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema") != "gpu-greenroom.producer-checkpoints.v1"
            or not isinstance(entries, list) or not entries
            or not all(isinstance(p, str) and p and not Path(p).is_absolute()
                       and ".." not in Path(p).parts for p in entries)):
        raise ValueError("invalid producer checkpoint declaration")
    try:
        scripts = admission.entry_script(argv, cwd)
    except ValueError:
        # Discovery is narrower than prevalidated interpreter admission. A
        # module/shell wrapper is not evidence of this declared entrypoint.
        return None
    if len(scripts) != 1:
        return None
    script = Path(scripts[0]).resolve()
    declared = [root / p for p in entries]
    if any(not p.resolve().is_relative_to(root) for p in declared):
        raise ValueError("producer checkpoint entrypoint escapes its source root")
    if script not in [p.resolve() for p in declared]:
        return None
    return {"schema": "gpu-greenroom.producer-checkpoint-binding.v1",
            "declaration": admission.file_binding(manifest),
            "entrypoint": admission.file_binding(script)}


def assert_current(request):
    record = request.params.get(PARAM)
    if record is None:
        return
    if (not isinstance(record, dict)
            or record.get("schema") != "gpu-greenroom.producer-checkpoint-binding.v1"):
        raise ValueError("invalid producer checkpoint binding")
    for key in ("declaration", "entrypoint"):
        value = record[key]
        if admission.file_binding(value["path"]) != value:
            raise ValueError(f"producer checkpoint {key} changed after submission")
    discovered = discover(request.repo_root, request.command_argv, request.command_cwd)
    if discovered is None or any(record.get(key) != value for key, value in discovered.items()):
        raise ValueError("producer checkpoint binding does not name this command")


def bind(request):
    if PARAM in request.params:
        assert_current(request)
        declaration = request.params[PARAM]
    else:
        declaration = discover(request.repo_root, request.command_argv, request.command_cwd)
    if declaration is None:
        return
    if request.agent_id is None and "owner" not in declaration:
        declaration = {**declaration, "owner": "job:" + request.job_id}
    request.cooperative_checkpoint = True
    request.params = {**request.params, PARAM: declaration}
    checkpoint_owner(request)
    for capability in ("structured-command.v2", "checkpoint-continuation.v1", *([OWNER_CAPABILITY] if request.agent_id is None else [])):
        if capability not in request.required_worker_capabilities:
            request.required_worker_capabilities.append(capability)
    if admission.CAPABILITY in request.required_worker_capabilities:
        if "prepared-checkpoint-continuation.v1" not in request.required_worker_capabilities:
            request.required_worker_capabilities.append("prepared-checkpoint-continuation.v1")
