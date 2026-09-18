import json
import threading
import subprocess
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from gpu_queue.operator_server import make_handler, queue_snapshot
from gpu_queue.queue import GPUQueue


def test_shared_page_preserves_actions_and_explains_legacy_pause():
    from gpu_queue.operator_server import PAGE
    script = PAGE.split('<script>', 1)[1].split('</script>', 1)[0]
    fixture = {'paused': True, 'pause_state': None, 'admission_control': True,
               'observed_at': 100, 'jobs': [
                   {'job_id': 'pending', 'status': 'pending', 'consistent': True},
                   {'job_id': 'running', 'status': 'running', 'consistent': True},
               ]}
    harness = '''const vm=require('node:vm'); const nodes={};
const context={URLSearchParams, location:{hash:'',pathname:'/'},history:{replaceState(){}},
sessionStorage:{getItem(){return ''}},fetch(){return new Promise(()=>{})},setInterval(){},setTimeout(){},
document:{body:{dataset:{}},querySelector(s){return nodes[s]??=( {style:{}} )},querySelectorAll(){return []}}};
vm.createContext(context);
'''
    harness += 'vm.runInContext('+json.dumps(script)+',context);\n'
    harness += 'vm.runInContext('+json.dumps('render('+json.dumps(fixture)+')')+',context);\n'
    harness += "console.log(JSON.stringify(nodes));"
    result = subprocess.run(['node', '-e', harness], capture_output=True, text=True, check=True)
    nodes = json.loads(result.stdout)
    assert '<span>' in nodes['#jobs']['innerHTML']
    assert 'Force stop running job' in nodes['#jobs']['innerHTML']
    assert 'native queue CLI' in nodes['#pauseMeta']['textContent']
    assert nodes['#resume']['disabled'] is True


def test_timings_do_not_invent_queue_wait_without_start(tmp_path):
    GPUQueue(tmp_path)
    job = tmp_path / 'done' / 'job'
    job.mkdir()
    (job/'status.json').write_text(json.dumps({
        'job_id': 'job', 'status': 'done', 'submitted_at': 100, 'finished_at': 200,
    }))
    (job/'request.json').write_text(json.dumps({'agent_id': 'fixture-owner'}))
    row = queue_snapshot(tmp_path, read_only=True)['jobs'][0]
    assert row['queue_wait_seconds'] is None
    assert row['execution_wall_seconds'] is None
    assert row['agent_id'] == 'fixture-owner'
    (job/'status.json').write_text(json.dumps({
        'job_id': 'job', 'status': 'done', 'submitted_at': 100,
        'started_at': 140, 'finished_at': 200,
    }))
    row = queue_snapshot(tmp_path, read_only=True)['jobs'][0]
    assert row['queue_wait_seconds'] == 40
    assert row['execution_wall_seconds'] == 60


def test_admission_only_http_pause_resume_receipts(tmp_path):
    queue = GPUQueue(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tmp_path, 'secret', admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def post(action, **body):
        return json.load(urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/{action}',
            data=json.dumps(body).encode(), headers={'Authorization':'Bearer secret'})))
    try:
        first = post('pause', request_id='pause-1', requested_by='operator')
        assert queue.is_paused()
        receipt = tmp_path / 'operator-actions' / 'pause-1.json'
        initial = receipt.read_bytes()
        assert json.loads(initial)['phase'] == 'completed'
        assert post('pause', request_id='pause-1', requested_by='operator') == first
        assert receipt.read_bytes() == initial
        with pytest.raises(HTTPError) as error:
            post('resume', request_id='wrong', requested_by='operator', epoch='stale')
        assert error.value.code == 409
        assert queue.is_paused()
        for action in ['cancel', 'terminate']:
            with pytest.raises(HTTPError) as error:
                post(action, job_id='not-a-job')
            assert error.value.code == 403
        result = post('resume', request_id='resume-1', requested_by='operator', epoch=first['epoch'])
        assert not queue.is_paused()
        assert result['previous_pause']['epoch'] == first['epoch']
        assert json.loads((tmp_path/'operator-actions/resume-1.json').read_text())['phase'] == 'completed'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_paused_snapshot_never_claims_running_work_is_clear(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.pause(owner='operator')
    running = tmp_path / 'running' / 'job'
    running.mkdir()
    (running/'status.json').write_text(json.dumps({'job_id':'job','status':'running'}))
    snapshot = queue_snapshot(tmp_path, 'done', read_only=True)
    assert snapshot['admission_state'] == 'paused_running'
    assert snapshot['running_job_ids'] == ['job']


def test_prepare_failure_cannot_pause(tmp_path, monkeypatch):
    from gpu_queue import admission_control
    queue = GPUQueue(tmp_path)
    monkeypatch.setattr(admission_control, 'write_durable', lambda *args: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        admission_control.transition(queue, action='pause', request_id='p', owner='operator')
    assert not queue.is_paused()


def test_incomplete_receipt_is_visible_and_not_replayed(tmp_path, monkeypatch):
    from gpu_queue import admission_control
    queue = GPUQueue(tmp_path)
    original = admission_control.write_durable
    def write(path, payload):
        if payload.get('phase') == 'completed':
            raise OSError('completion receipt failure')
        original(path, payload)
    monkeypatch.setattr(admission_control, 'write_durable', write)
    with pytest.raises(OSError):
        admission_control.transition(queue, action='pause', request_id='p', owner='operator')
    assert queue.is_paused()
    assert json.loads((tmp_path/'operator-actions/p.json').read_text())['phase'] == 'prepared'
    with pytest.raises(RuntimeError, match='incomplete'):
        admission_control.transition(queue, action='pause', request_id='p', owner='operator')
