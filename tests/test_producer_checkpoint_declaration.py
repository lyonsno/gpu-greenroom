"""Source-owned checkpoint declarations require no per-caller switch."""

import json
from pathlib import Path
import sys
import os
import pytest

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def producer(tmp_path):
    root = tmp_path / "producer"
    root.mkdir()
    (root / "generate.py").write_text("print('producer')\n")
    (root / ".gpu-greenroom-producer.json").write_text(json.dumps({
        "schema": "gpu-greenroom.producer-checkpoints.v1",
        "entrypoints": ["generate.py"],
    }))
    return root


def test_ordinary_structured_submission_inherits_source_checkpoint_declaration(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "-u", "generate.py"], route_identity="producer/default")
    job = queue.submit(request)
    saved = json.loads((job / "request.json").read_text())
    assert saved["cooperative_checkpoint"] is True, "ordinary caller lost the producer's checkpoint capability"
    assert "checkpoint-continuation.v1" in saved["required_worker_capabilities"]
    assert saved["params"]["_producer_checkpoint"]["entrypoint"]["path"] == str(root / "generate.py")


def test_other_entrypoints_do_not_inherit_a_neighboring_declaration(tmp_path):
    root = producer(tmp_path)
    (root / "opaque.py").write_text("pass\n")
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "opaque.py"], route_identity="producer/opaque")
    job = queue.submit(request)
    assert json.loads((job / "request.json").read_text())["cooperative_checkpoint"] is False


@pytest.mark.parametrize("owner", [None, "ordinary-caller"])
def test_ordinary_typed_submission_resolves_the_declared_source_route(tmp_path, capsys, monkeypatch, owner):
    from gpu_queue.cli import main

    root = producer(tmp_path)
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "job_types.json").write_text(json.dumps({"example-producer": {
        "cmd": [sys.executable, "-u", "generate.py", "--output", "{output_dir}/mesh.glb", "--seed", "{seed}"],
        "cwd": str(root), "env": {"PYTHONPATH": "."}, "defaults": {"seed": "17"},
    }}))
    argv = ["gpu-greenroom", "--queue-dir", str(queue), "submit", "example-producer", "input.png"]
    if owner is not None:
        argv += ["--agent-id", owner]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    job = next((queue / "pending").iterdir())
    saved = json.loads((job / "request.json").read_text())
    assert saved["cooperative_checkpoint"] is True, "typed route dropped the source declaration"
    assert saved["agent_id"] == owner, "checkpoint ownership must not invent a caller identity"
    assert saved["repo_root"] == str(root)
    assert saved["command_argv"][-1] == "17"
    assert saved["command_argv"][4] == str(Path(saved["output_dir"]) / "mesh.glb")


def test_changed_declared_entrypoint_fails_before_child_execution(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", agent_id="ordinary-caller",
                         repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "generate.py"], route_identity="producer/default")
    queue.submit(request)
    marker = tmp_path / "must-not-execute"
    (root / "generate.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    assert queue.run_one({}) is True
    state = queue.get_job(request.job_id)
    assert state.status.value == "failed"
    assert "changed after submission" in state.error_message
    assert not marker.exists()


def test_unattributed_caller_gets_job_owned_checkpoint_without_identity_flags(tmp_path):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    request = JobRequest(job_type="command", input_path="", repo_root=str(root), command_cwd=str(root),
                         command_argv=[sys.executable, "generate.py"], route_identity="producer/default")
    job = queue.submit(request)
    saved = json.loads((job / "request.json").read_text())
    assert saved["cooperative_checkpoint"] is True
    assert saved["agent_id"] is None
    assert saved["params"]["_producer_checkpoint"]["owner"] == "job:" + request.job_id
    assert "producer-owned-checkpoints.v1" in saved["required_worker_capabilities"]


@pytest.mark.parametrize("argv", [[sys.executable, "-m", "pytest"], ["bash", "-lc", "true"]])
def test_unrelated_interpreter_modes_remain_ordinary_commands(tmp_path, argv):
    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    job = queue.submit(JobRequest(job_type="command", input_path="", repo_root=str(root),
                                  command_cwd=str(root), command_argv=argv, route_identity="ordinary/unrelated"))
    assert json.loads((job / "request.json").read_text())["cooperative_checkpoint"] is False


@pytest.mark.parametrize("prepared,tamper", [(False, False), (True, False), (True, True)])
def test_job_owned_checkpoint_releases_and_resumes_with_validation_preserved(tmp_path, prepared, tamper):
    from gpu_queue import admission

    root = producer(tmp_path)
    queue = GPUQueue(tmp_path / "queue")
    (root / "generate.py").write_text('''import json,pathlib,time,uuid
from dataclasses import replace
from gpu_queue.cooperative import _context,load_checkpoint,yield_if_requested
q,job,r,s=_context()
state=load_checkpoint() or {'step':0}
pathlib.Path('result.json').write_text(json.dumps({'step':state['step'],'agent_id':r.agent_id}))
if state['step']==0:
 if TAMPPER: pathlib.Path('preflight.py').write_text('changed')
 q.pause(owner='isolated-fixture')
 yield_if_requested(save_checkpoint=lambda p:{'step':1},quiesce=lambda:True,
  continuation=lambda p:replace(r,job_id=uuid.uuid4().hex[:12],submitted_at=time.time()))
'''.replace('TAMPPER', repr(tamper)))
    manifest = {"schema": "gpu-greenroom.command.v3" if prepared else "gpu-greenroom.command.v1",
                "repo_root": str(root), "cwd": str(root), "argv": [sys.executable, "generate.py"],
                "route_identity": "producer/job-owned-conformance", "env": {"PYTHONPATH": str(Path(__file__).resolve().parents[1])}}
    if prepared:
        checker = root / "preflight.py"
        checker.write_text('''import json,os,pathlib,sys
r=json.load(sys.stdin)
pathlib.Path(os.environ['GPU_GREENROOM_VALIDATION_OUTPUT']).write_text(json.dumps({
 'schema':'gpu-greenroom.preflight-result.v1','request_digest':r['request_digest'],'valid':True,'mode':'cpu-only'}))
''')
        manifest["preflight"] = {"argv": [sys.executable, str(checker)], "inputs": [str(checker)]}
        plan = admission.prepare(queue, manifest)
        parent = admission.submit_prepared(queue, plan)
    else:
        parent = queue.submit(admission.command_request(admission.normalize_manifest(manifest)))
    assert queue.run_one({})
    if tamper:
        assert queue.get_job(parent.name).status.value == "failed"
        assert not list((queue.queue_dir / "pending").iterdir())
        assert "validated input changed" in (queue.queue_dir / "failed" / parent.name / "stderr.log").read_text()
        return
    assert queue.get_job(parent.name).status.value == "done"
    child = next((queue.queue_dir / "pending").iterdir())
    request = JobRequest.from_json((child / "request.json").read_text())
    assert request.agent_id is None
    assert request.params["_producer_checkpoint"]["owner"] == "job:" + parent.name
    if prepared:
        validation = json.loads((child / "validation.json").read_text())
        admission.assert_current(request, validation)
        assert validation["preflight"]["result"]["valid"] is True
    queue.resume(owner="isolated-fixture")
    assert queue.run_one({})
    assert queue.get_job(child.name).status.value == "done"
    assert json.loads((root / "result.json").read_text()) == {"step": 1, "agent_id": None}


def test_frozen_typed_command_preserves_executable_defaults_and_ignored_receipt(tmp_path, monkeypatch):
    from gpu_queue.cli import main

    root = producer(tmp_path)
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    (executable_dir / "python").symlink_to(sys.executable)
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "job_types.json").write_text(json.dumps({"example": {
        "cmd": ["python", "generate.py", "--seed", "{seed}", "--output", "{output_dir}/mesh.glb"],
        "cwd": str(root), "env": {"PATH": str(executable_dir) + os.pathsep + os.environ["PATH"]},
        "defaults": {"seed": "17"}}}))
    monkeypatch.setattr(sys, "argv", ["gpu-greenroom", "--queue-dir", str(queue), "submit", "example", "image.png", "-p", "unused=9"])
    main()
    q = GPUQueue(queue)
    job = next((queue / "pending").iterdir())
    saved = json.loads((job / "request.json").read_text())
    assert saved["command_argv"][0] == str(executable_dir / "python")
    assert q.run_one({})
    receipt = json.loads((queue / "done" / job.name / "receipt.json").read_text())
    assert receipt["effective_defaults"] == {"seed": "17"}
    assert receipt["ignored_params"] == {"unused": "9"}
