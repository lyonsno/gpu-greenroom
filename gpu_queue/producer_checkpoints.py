"""Bind a producer's exact entrypoint declaration during ordinary admission."""

import json
from pathlib import Path

from . import admission


PARAM = "_producer_checkpoint"


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
    scripts = admission.entry_script(argv, cwd)
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
    if discover(request.repo_root, request.command_argv, request.command_cwd) != record:
        raise ValueError("producer checkpoint binding does not name this command")


def bind(request):
    if PARAM in request.params:
        assert_current(request)
        declaration = request.params[PARAM]
    else:
        declaration = discover(request.repo_root, request.command_argv, request.command_cwd)
    if declaration is None:
        return
    if admission.CAPABILITY in request.required_worker_capabilities:
        raise ValueError("producer checkpoint continuation cannot discard prepared-command validation")
    if not isinstance(request.agent_id, str) or not request.agent_id.strip():
        raise ValueError("checkpoint-capable producer requires an explicit job owner")
    request.cooperative_checkpoint = True
    request.params = {**request.params, PARAM: declaration}
