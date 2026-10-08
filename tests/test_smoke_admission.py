from copy import deepcopy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from gpu_queue.queue import GPUQueue
from gpu_queue import admission
from gpu_queue.models import JobRequest
from gpu_queue.operator_server import make_handler
from gpu_queue.smoke_requests import SmokeRequestConflict, SmokeRequests


def command(tmp_path):
    return {
        "schema": "gpu-greenroom.command.v3",
        "agent_id": "smoke-owner", "repo_root": str(tmp_path), "cwd": str(tmp_path),
        "route_identity": "live-smoke/test", "output_dir": str(tmp_path / "output"),
        "argv": [sys.executable, "-c", "print('native child')"],
    }


def smoke(tmp_path, manifest):
    return {
        "schema": "gpu-greenroom.interactive-smoke.v1",
        "id": "9c0a03f6-b2d8-43f4-b7b7-5e848e733661", "kind": "interactive-smoke",
        "source": {"agent_id": "smoke-owner", "repo_root": str(tmp_path)},
        "title": "Live smoke", "prompt": "Start the smoke, then report the result.",
        "url": "http://127.0.0.1:8766/", "availability": "prepared",
        "availability_note": "Inputs prepared; GPU not acquired.",
        "operator_command": manifest,
    }


def preflight(tmp_path, *, valid=True):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"requiredStages": ["stage-a"] if valid else []}))
    script = tmp_path / "validate.py"
    script.write_text(
        "import json,os,pathlib,sys\n"
        "value=json.load(sys.stdin)\n"
        "assert os.environ['GPU_GREENROOM_VALIDATION_MODE']=='cpu-only'\n"
        "config=json.loads(pathlib.Path(sys.argv[1]).read_text())\n"
        "valid=bool(config.get('requiredStages'))\n"
        "report={'schema':'gpu-greenroom.preflight-result.v1','valid':valid,"
        "'request_digest':value['request_digest'],'mode':'cpu-only',"
        "'error':None if valid else 'requiredStages must be a non-empty array'}\n"
        "pathlib.Path(os.environ['GPU_GREENROOM_VALIDATION_OUTPUT']).write_text(json.dumps(report))\n"
        "sys.exit(0 if valid else 2)\n"
    )
    return {"argv": [sys.executable, str(script), str(config)], "inputs": [str(script), str(config)]}


def test_bad_preflight_rejects_smoke_before_queue_admission(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    manifest = command(tmp_path)
    manifest["preflight"] = preflight(tmp_path, valid=False)
    store = SmokeRequests(queue.queue_dir / "smoke-requests")
    with pytest.raises(ValueError, match="requiredStages"):
        store.submit(smoke(tmp_path, manifest))
    assert list((queue.queue_dir / "pending").iterdir()) == []
    assert list((queue.queue_dir / "running").iterdir()) == []


def test_operator_smoke_cannot_be_answered_before_start(tmp_path):
    store = SmokeRequests(tmp_path / "queue" / "smoke-requests")
    payload = smoke(tmp_path, command(tmp_path))
    record, _ = store.submit(payload)
    with pytest.raises(SmokeRequestConflict, match="not available"):
        store.respond(record["request"]["id"], "Passed")


def test_missing_executable_rejected_at_submission(tmp_path):
    manifest = command(tmp_path)
    manifest["schema"] = "gpu-greenroom.command.v1"
    manifest["argv"] = [str(tmp_path / "missing-executable")]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    queue = tmp_path / "queue"
    result = subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue),
         "submit-command", "--manifest", str(manifest_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "executable" in result.stderr
    assert list((queue / "pending").iterdir()) == []


@pytest.mark.parametrize('job_type,definitions', [('missing', {}), ('broken', {'broken': {'cmd': ['/not/an/executable']}})])
def test_registered_job_static_errors_return_before_enqueue(tmp_path, job_type, definitions):
    queue = GPUQueue(tmp_path / 'queue')
    (queue.queue_dir / 'job_types.json').write_text(json.dumps(definitions))
    result = subprocess.run([sys.executable, '-m', 'gpu_queue.cli', '--queue-dir', str(queue.queue_dir),
                             'submit', job_type, 'input'], capture_output=True, text=True)
    assert result.returncode == 2
    assert not list((queue.queue_dir / 'pending').iterdir())
    assert list((queue.queue_dir / 'submission-failures').glob('*.json'))


def test_operator_start_enqueues_once_without_reserving_gpu_beforehand(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    store = SmokeRequests(queue.queue_dir / "smoke-requests")
    payload = smoke(tmp_path, command(tmp_path))
    record, _ = store.submit(payload)
    assert store.display(record)["phase"] == "awaiting-start"
    assert not list((queue.queue_dir / "pending").iterdir())
    assert not list((queue.queue_dir / "running").iterdir())
    assert queue.is_paused() is False
    result = store.start(payload["id"], record["request_digest"])
    replay = store.start(payload["id"], record["request_digest"])
    assert replay["activation"]["job_id"] == result["activation"]["job_id"]
    assert len(list((queue.queue_dir / "pending").iterdir())) == 1
    assert store.display(result)["phase"] == "waiting-gpu"


def test_start_rejects_changed_validation_input_without_enqueuing(tmp_path):
    queue = GPUQueue(tmp_path / "queue")
    store = SmokeRequests(queue.queue_dir / "smoke-requests")
    manifest = command(tmp_path)
    manifest["preflight"] = preflight(tmp_path)
    payload = smoke(tmp_path, manifest)
    record, _ = store.submit(payload)
    (tmp_path / "config.json").write_text('{"requiredStages": []}')
    with pytest.raises(ValueError, match="changed"):
        store.start(payload["id"], record["request_digest"])
    assert not list((queue.queue_dir / "pending").iterdir())


def test_legacy_schema_cannot_silently_drop_preflight(tmp_path):
    manifest = command(tmp_path)
    manifest.update(schema="gpu-greenroom.command.v2", preflight=preflight(tmp_path))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    result = subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(tmp_path / "queue"),
         "submit-command", "--manifest", str(path)], capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "command.v3" in result.stderr
    assert not list((tmp_path / "queue" / "pending").iterdir())


def test_waiting_operator_does_not_block_other_work_and_start_respects_pause(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    record, _ = store.submit(smoke(tmp_path, command(tmp_path)))
    other = JobRequest(job_type='command', input_path='', command_argv=[sys.executable, '-c', "print('other')"],
                       command_cwd=str(tmp_path), agent_id='other')
    queue.submit(other)
    assert queue.run_one({}) is True
    assert (queue.queue_dir / 'done' / other.job_id / 'receipt.json').is_file()
    epoch = queue.pause(owner='test')['epoch']
    started = store.start(record['request']['id'], record['request_digest'])
    assert queue.run_one({}) is False
    assert store.display(started)['phase'] == 'waiting-gpu'
    queue.resume(owner='test', epoch=epoch)
    assert queue.run_one({}) is True
    current = store.get(record['request']['id'])
    assert store.display(current)['phase'] == 'operator-needed'


def test_concurrent_starts_preserve_one_job(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    record, _ = store.submit(smoke(tmp_path, command(tmp_path)))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: store.start(record['request']['id'], record['request_digest']), range(2)))
    assert results[0]['activation']['job_id'] == results[1]['activation']['job_id']
    assert len(list((queue.queue_dir / 'pending').iterdir())) == 1


def test_recovery_after_submission_before_activation_receipt_does_not_rerun(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    record, _ = store.submit(smoke(tmp_path, command(tmp_path)))
    job = admission.submit_prepared(queue, record['prepared'])
    assert queue.run_one({}) is True
    recovered = store.start(record['request']['id'], record['request_digest'])
    assert recovered['activation']['job_id'] == job.name
    assert store.display(recovered)['job_state'] == 'done'
    assert not list((queue.queue_dir / 'pending').iterdir())


def test_dispatch_invalidates_mutated_inputs_without_starting_or_advancing_fairness(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    manifest = command(tmp_path)
    manifest['preflight'] = preflight(tmp_path)
    plan = admission.prepare(queue, manifest)
    directory = admission.submit_prepared(queue, plan)
    (tmp_path / 'config.json').write_text('{"requiredStages": []}')
    assert queue.run_one({}) is True
    status = json.loads((queue.queue_dir / 'failed' / directory.name / 'status.json').read_text())
    receipt = json.loads((queue.queue_dir / 'failed' / directory.name / 'receipt.json').read_text())
    assert status['failure_phase'] == 'validation-invalidated'
    assert status['started_at'] is None
    assert receipt['execution_started'] is False
    assert not (queue.queue_dir / 'dispatch-state.json').exists()


@pytest.mark.parametrize('body', ['None', "{'schema':'gpu-greenroom.preflight-result.v1','valid':True,'mode':'cpu-only','request_digest':'wrong'}", "{'schema':'gpu-greenroom.preflight-result.v1','valid':True,'mode':'gpu','request_digest':value['request_digest']}"])
def test_exit_zero_without_trustworthy_preflight_is_not_valid(tmp_path, body):
    queue = GPUQueue(tmp_path / 'queue')
    manifest = command(tmp_path)
    spec = preflight(tmp_path)
    script = tmp_path / 'validate.py'
    script.write_text("import json,os,pathlib,sys\nvalue=json.load(sys.stdin)\nresult=" + body +
                      "\nif result is not None: pathlib.Path(os.environ['GPU_GREENROOM_VALIDATION_OUTPUT']).write_text(json.dumps(result))\n")
    manifest['preflight'] = spec
    with pytest.raises(ValueError):
        admission.prepare(queue, manifest)
    reports = list((queue.queue_dir / 'validation-reports').glob('*/validation.json'))
    assert reports and json.loads(reports[0].read_text())['valid'] is False
    assert not list((queue.queue_dir / 'pending').iterdir())


def test_http_start_only_activates_local_published_command_and_hides_environment(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    manifest = command(tmp_path)
    manifest['env'] = {'PRIVATE_EXAMPLE': 'not-for-the-browser'}
    payload = smoke(tmp_path, manifest)
    record, _ = store.submit(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(queue.queue_dir, 'secret', admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'

    def request(path, body=None):
        headers = {'Authorization': 'Bearer secret', 'Content-Type': 'application/json'}
        data = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(base + path, data=data, headers=headers)) as response:
            return json.loads(response.read())

    try:
        with pytest.raises(HTTPError) as error:
            request('/api/smoke-requests', payload)
        assert error.value.code == 400
        public = request('/api/smoke-requests/' + payload['id'])
        assert 'not-for-the-browser' not in json.dumps(public)
        assert 'prepared' not in public
        assert request('/api/smoke-requests')['items'][0]['display']['phase'] == 'awaiting-start'
        path = '/api/smoke-requests/' + payload['id'] + '/start'
        with pytest.raises(HTTPError) as error:
            request(path, {'request_digest': record['request_digest'], 'argv': ['bad']})
        assert error.value.code == 400
        activated = request(path, {'request_digest': record['request_digest']})
        replay = request(path, {'request_digest': record['request_digest']})
        assert activated['activation']['job_id'] == replay['activation']['job_id']
        assert 'not-for-the-browser' not in json.dumps(activated)
        assert len(list((queue.queue_dir / 'pending').iterdir())) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
