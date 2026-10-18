import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest

from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


def submit(queue, tmp_path, code='pass'):
    request = JobRequest(job_type='command', input_path='', agent_id='cancel-fixture',
        route_identity='fixture/cancel', command_cwd=str(tmp_path),
        command_argv=[sys.executable, '-c', code],
        command_env={'PYTHONPATH': str(Path(__file__).resolve().parents[1])})
    queue.submit(request)
    return request


def cancel(queue, request, *, status='pending', started=None, action='cancel-1', force=False):
    path = queue.queue_dir / status / request.job_id / 'request.json'
    return queue.request_cancel(job_id=request.job_id,
        request_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_status=status, expected_started_at=started,
        request_id=action, requested_by='operator-console', force=force)


def wait_until(predicate, seconds=8):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    pytest.fail('fixture did not reach its synchronization point')


@pytest.mark.parametrize('status', ['pending', 'deferred'])
def test_cancel_queued_is_durable_idempotent_and_never_launches(tmp_path, status):
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    if status == 'deferred':
        job = queue.queue_dir/'pending'/request.job_id
        state = JobState.from_json((job/'status.json').read_text())
        state.status = JobStatus.DEFERRED
        (job/'status.json').write_text(state.to_json())
        job.rename(queue.queue_dir/'deferred'/request.job_id)
    result = cancel(queue, request, status=status)
    assert result['status'] == 'cancelled'
    receipt = json.loads(Path(result['receipt_path']).read_text())
    assert receipt['phase'] == 'completed'
    assert receipt['request']['job_id'] == request.job_id
    assert cancel_replay(queue, receipt['request']) == result
    assert queue.get_job(request.job_id).status == JobStatus.CANCELLED
    assert not queue.run_one({})
    assert not queue.is_paused()


def cancel_replay(queue, body):
    return queue.request_cancel(**body)


def test_stale_confirmation_cannot_cancel_new_dispatch_attempt(tmp_path):
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    path = queue.queue_dir/'pending'/request.job_id
    digest = hashlib.sha256((path/'request.json').read_bytes()).hexdigest()
    path.rename(queue.queue_dir/'running'/request.job_id)
    with pytest.raises(ValueError, match='changed'):
        queue.request_cancel(job_id=request.job_id, request_digest=digest,
            expected_status='pending', expected_started_at=None, request_id='stale',
            requested_by='operator-console')
    assert not (queue.queue_dir/'running'/request.job_id/'cancel-request.json').exists()
    assert not queue.is_paused()


def test_running_old_worker_fails_explicitly_without_signalling(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    path = queue.queue_dir/'pending'/request.job_id
    state = JobState.from_json((path/'status.json').read_text())
    state.status, state.started_at = JobStatus.RUNNING, 123
    (path/'status.json').write_text(state.to_json())
    path.rename(queue.queue_dir/'running'/request.job_id)
    monkeypatch.setattr(os, 'killpg', lambda *args: pytest.fail('API must not signal'))
    with pytest.raises(ValueError, match='worker'):
        cancel(queue, request, status='running', started=123)


@pytest.mark.parametrize('cooperative', [False, True])
def test_running_cancel_waits_for_cleanup_and_releases_only_after_group_exit(tmp_path, cooperative):
    queue = GPUQueue(tmp_path/'queue')
    marker, cleaning, release, cleaned = (tmp_path/name for name in ('started','cleaning','release','cleaned'))
    code = ("from gpu_queue.cancellation import cancellation\n" if cooperative else '') + '''import pathlib,sys,time
paths = [pathlib.Path(p) for p in sys.argv[1:]]
'''
    if cooperative:
        code += '''try:
    with cancellation() as stop:
        paths[0].touch()
        while True:
            stop.checkpoint()
            time.sleep(.01)
finally:
    paths[1].touch()
    while not paths[2].exists(): time.sleep(.01)
    paths[3].touch()
'''
    else:
        code += 'paths[0].touch()\nwhile True: time.sleep(1)\n'
    request = submit(queue, tmp_path, code)
    path = queue.queue_dir/'pending'/request.job_id/'request.json'
    raw = json.loads(path.read_text())
    raw['command_argv'] += [str(p) for p in (marker, cleaning, release, cleaned)]
    path.write_text(json.dumps(raw))
    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        wait_until(marker.exists)
        started = queue.get_job(request.job_id).started_at
        result = cancel(queue, request, status='running', started=started)
        assert result['status'] == 'cancelling'
        if cooperative:
            wait_until(cleaning.exists)
            assert queue.get_job(request.job_id).status == JobStatus.RUNNING
            with queue.lock_path.open('a') as stream:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
            release.touch()
        worker.join(8)
        assert not worker.is_alive()
        state = queue.get_job(request.job_id)
        assert state.status == JobStatus.CANCELLED
        receipt = json.loads((queue.queue_dir/'cancelled'/request.job_id/'receipt.json').read_text())
        assert receipt['cancellation']['request_id'] == 'cancel-1'
        assert receipt['failure_phase'] == 'operator_cancelled'
        if cooperative:
            assert cleaned.exists()
        assert not queue.is_paused()
        with queue.lock_path.open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX|fcntl.LOCK_NB)
    finally:
        release.touch()
        if worker.is_alive():
            state = queue.get_job(request.job_id)
            if state and state.child_process_group:
                os.killpg(state.child_process_group, signal.SIGKILL)
            worker.join(8)


def test_standalone_hook_preserves_existing_signal_handler(monkeypatch):
    from gpu_queue.cancellation import cancellation
    monkeypatch.delenv('GPU_GREENROOM_CANCEL_REQUEST', raising=False)
    before = signal.getsignal(signal.SIGTERM)
    with cancellation() as stop:
        assert not stop.requested
        stop.checkpoint()
        assert signal.getsignal(signal.SIGTERM) == before
    assert signal.getsignal(signal.SIGTERM) == before


def test_force_stop_is_explicit_and_never_releases_a_live_job(tmp_path):
    queue = GPUQueue(tmp_path/'queue')
    marker = tmp_path/'started'
    request = submit(queue, tmp_path,
        f"import signal,pathlib,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pathlib.Path({str(marker)!r}).touch(); time.sleep(60)")
    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        wait_until(marker.exists)
        state = queue.get_job(request.job_id)
        cancel(queue, request, status='running', started=state.started_at)
        progress = queue.queue_dir/'running'/request.job_id/'cancel-progress.json'
        wait_until(progress.exists)
        assert queue.get_job(request.job_id).status == JobStatus.RUNNING
        assert json.loads(progress.read_text())['signal'] == signal.SIGTERM
        cancel(queue, request, status='running', started=state.started_at, action='force-1', force=True)
        worker.join(8)
        assert not worker.is_alive()
        assert queue.get_job(request.job_id).status == JobStatus.CANCELLED
        assert json.loads((queue.queue_dir/'cancelled'/request.job_id/'cancel-progress.json').read_text())['signal'] == signal.SIGKILL
        next_job = submit(queue, tmp_path, 'import time; time.sleep(.1)')
        assert queue.run_one({})
        assert queue.get_job(next_job.job_id).status == JobStatus.DONE
    finally:
        if worker.is_alive():
            os.killpg(queue.get_job(request.job_id).child_process_group, signal.SIGKILL)
            worker.join(8)


@pytest.mark.parametrize('change', ['digest', 'action-id', 'traversal', 'wrong-force'])
def test_bad_or_reused_identity_preserves_the_queued_job(tmp_path, change):
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    body = dict(job_id=request.job_id,
        request_digest=hashlib.sha256((queue.queue_dir/'pending'/request.job_id/'request.json').read_bytes()).hexdigest(),
        expected_status='pending', expected_started_at=None, request_id='test', requested_by='operator-console', force=False)
    if change == 'digest':
        body['request_digest'] = '0'*64
    if change == 'action-id':
        other = submit(queue, tmp_path)
        cancel(queue, other, action='test')
    if change == 'traversal':
        body['job_id'] = '../running'
    if change == 'wrong-force':
        body['force'] = 'false'
    with pytest.raises(ValueError):
        queue.request_cancel(**body)
    assert queue.get_job(request.job_id).status == JobStatus.PENDING


def test_http_cancel_is_opt_in_authenticated_and_snapshot_bound(tmp_path):
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from gpu_queue.operator_server import make_handler
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1',0), make_handler(queue.queue_dir, 'secret',
        admission_control=True, job_control=True, local_operator=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def post(body, token='secret'):
        return json.load(urlopen(Request(base+'/api/job-cancel', data=json.dumps(body).encode(),
            headers={'Authorization':'Bearer '+token})))
    try:
        state = json.load(urlopen(Request(base+'/api/state',headers={'Authorization':'Bearer secret'})))
        assert state['job_control'] is True
        job = state['jobs'][0]
        body = {key: job[key] for key in ('job_id','request_digest')}
        body.update(expected_status=job['status'], expected_started_at=job['started_at'], request_id='http-1', force=False)
        for bad_body, token, expected in [(body, 'wrong',401), ({**body, 'command':['echo']},'secret',409),
            ({**body,'request_digest':'0'*64},'secret',409)]:
            with pytest.raises(HTTPError) as error:
                post(bad_body, token)
            assert error.value.code == expected
            assert queue.get_job(request.job_id).status == JobStatus.PENDING
        assert post(body)['status'] == 'cancelled'
        assert post(body)['status'] == 'cancelled'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_incomplete_control_receipt_does_not_pretend_success(tmp_path, monkeypatch):
    from gpu_queue import cancellation
    queue = GPUQueue(tmp_path/'queue')
    request = submit(queue, tmp_path)
    original = cancellation.write_durable
    def write(path, payload):
        if payload.get('phase') == 'completed':
            raise OSError('disk full at acknowledgement')
        original(path, payload)
    monkeypatch.setattr(cancellation, 'write_durable', write)
    with pytest.raises(OSError, match='disk full'):
        cancel(queue, request)
    receipt = json.loads((queue.queue_dir/'operator-actions/cancel-1.json').read_text())
    assert receipt['phase'] == 'prepared'
    with pytest.raises(RuntimeError, match='incomplete'):
        cancel_replay(queue, receipt['request'])


def test_cancel_at_prelaunch_does_not_execute_workload(tmp_path, monkeypatch):
    from gpu_queue import cancellation
    queue = GPUQueue(tmp_path/'queue')
    marker = tmp_path/'must-not-exist'
    request = submit(queue, tmp_path, f"from pathlib import Path; Path({str(marker)!r}).touch()")
    original = cancellation.read_request
    requested = False
    def read(job, state):
        nonlocal requested
        if not requested and state.status == JobStatus.RUNNING and not state.child_pid:
            requested = True
            cancel(queue, request, status='running', started=state.started_at)
        return original(job,state)
    monkeypatch.setattr(cancellation, 'read_request', read)
    assert queue.run_one({})
    assert requested and not marker.exists()
    assert queue.get_job(request.job_id).status == JobStatus.CANCELLED


def test_unresolved_quiescence_is_not_cancelled_or_free(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path/'queue')
    marker = tmp_path/'started'
    request = submit(queue, tmp_path, f"import time,pathlib; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)")
    monkeypatch.setattr(queue, '_quiesce_owned_child', lambda *_: False)
    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        wait_until(marker.exists)
        cancel(queue, request, status='running', started=queue.get_job(request.job_id).started_at)
        worker.join(8)
        assert not worker.is_alive()
        state = queue.get_job(request.job_id)
        assert state.status == JobStatus.RUNNING and state.finished_at is None
        receipt = json.loads((queue.queue_dir/'running'/request.job_id/'receipt.json').read_text())
        assert receipt['status'] == 'ownership_unknown'
        submit(queue, tmp_path)
        assert not queue.run_one({})
    finally:
        if worker.is_alive():
            os.killpg(queue.get_job(request.job_id).child_process_group, signal.SIGKILL)
            worker.join(8)


def test_browser_cancel_modal_uses_real_queue_api(tmp_path):
    from tests.test_operator_server import _run_operator_browser_witness
    _run_operator_browser_witness(tmp_path, 'job-cancel')


def test_malformed_running_request_cannot_silently_close_as_cancelled(tmp_path):
    queue = GPUQueue(tmp_path/'queue')
    marker = tmp_path/'started'
    request = submit(queue, tmp_path, f"import pathlib,time; pathlib.Path({str(marker)!r}).touch(); time.sleep(60)")
    worker = threading.Thread(target=queue.run_one, args=({},))
    worker.start()
    try:
        wait_until(marker.exists)
        queue._write_json_atomic(queue.queue_dir/'running'/request.job_id/'cancel-request.json', [])
        worker.join(8)
        assert not worker.is_alive()
        assert queue.get_job(request.job_id).status == JobStatus.FAILED
        receipt = json.loads((queue.queue_dir/'failed'/request.job_id/'receipt.json').read_text())
        assert receipt['failure_phase'] == 'cancellation_invalid'
        assert 'unverified' in receipt['cancellation_error']
    finally:
        if worker.is_alive():
            os.killpg(queue.get_job(request.job_id).child_process_group, signal.SIGKILL)
            worker.join(8)


@pytest.mark.parametrize('boundary', ['save', 'registration', 'committed', 'recovery', 'published', 'legacy-producer'])
def test_cancellation_during_checkpoint_handoff_cannot_publish_or_recover_successor(tmp_path, boundary):
    queue = GPUQueue(tmp_path/'queue')
    marker, release = tmp_path/'boundary', tmp_path/'release'
    script = tmp_path/'checkpoint.py'
    script.write_text('''import os,pathlib,sys,time
from gpu_queue.cancellation import cancellation
from gpu_queue.cooperative import _context,yield_if_requested
from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue
queue,job,request,state=_context()
queue.pause(owner='fixture-checkpoint')
def block(kind):
    if kind==sys.argv[1] or (sys.argv[1] in ('recovery','legacy-producer') and kind=='registration'):
        pathlib.Path(sys.argv[2]).touch()
        while not pathlib.Path(sys.argv[3]).exists(): time.sleep(.01)
def save(directory):
    block('save')
    return {'saved':'checkpoint preserved'}
def continuation(path):
    return JobRequest(job_type='command',input_path='',agent_id=request.agent_id,
        repo_root=request.repo_root,command_cwd=request.command_cwd,route_identity=request.route_identity,
        command_argv=[sys.executable,'-c',"print('must not run')"])
original=GPUQueue.publish_continuation
def publish(self,stage):
    if sys.argv[1]=='recovery': os._exit(9)
    if sys.argv[1]=='legacy-producer': return stage.rename(self.queue_dir/'pending'/stage.name)
    block('committed')
    result=original(self,stage)
    block('published')
    return result
GPUQueue.publish_continuation=publish
with cancellation() as stop:
    yield_if_requested(save_checkpoint=save,quiesce=lambda:True,
        continuation=continuation,on_submitted=lambda request:block('registration'))
    stop.checkpoint()
''')
    request = JobRequest(job_type='command',input_path='',agent_id='fixture-checkpoint',
        route_identity='fixture/checkpoint-cancel',repo_root=str(tmp_path),command_cwd=str(tmp_path),
        command_argv=[sys.executable,str(script),boundary,str(marker),str(release)],
        command_env={'PYTHONPATH':str(Path(__file__).resolve().parents[1])},cooperative_checkpoint=True)
    queue.submit(request)
    worker = threading.Thread(target=queue.run_one,args=({},))
    worker.start()
    try:
        wait_until(marker.exists)
        cancel(queue,request,status='running',started=queue.get_job(request.job_id).started_at)
        wait_until(lambda:(queue.queue_dir/'running'/request.job_id/'cancel-progress.json').exists())
        release.touch()
        worker.join(8)
        assert not worker.is_alive()
        assert queue.get_job(request.job_id).status == JobStatus.CANCELLED
        assert not list((queue.queue_dir/'pending').iterdir())
        queue.recover_continuations()
        queue.recover_continuations()
        assert not list((queue.queue_dir/'pending').iterdir())
        assert (queue.queue_dir/'outputs'/request.job_id/'checkpoints/checkpoint.json').is_file()
    finally:
        release.touch()
        if worker.is_alive():
            os.killpg(queue.get_job(request.job_id).child_process_group,signal.SIGKILL)
            worker.join(8)


@pytest.mark.parametrize('committed', [False,True])
def test_crash_recovery_cancel_precedes_both_deferral_paths(tmp_path,monkeypatch,committed):
    from gpu_queue import performance_window
    monkeypatch.setattr(performance_window,'native_notice',lambda *_:None)
    from gpu_queue.operator_server import queue_snapshot
    queue = GPUQueue(tmp_path/'queue')
    request = JobRequest(job_type='command',input_path='',command_argv=[sys.executable,'-c','pass'],
        agent_id='fixture-recovery',performance_window='CPU-only recovery fixture')
    job = queue.submit(request)
    job = job.rename(queue.queue_dir/'running'/request.job_id)
    child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'],start_new_session=True)
    identity = queue._process_start_identity(child.pid)
    assert identity
    child.wait()
    state = JobState.from_json((job/'status.json').read_text())
    state.status,state.started_at = JobStatus.RUNNING,123
    state.worker_pid,state.child_pid,state.child_process_group = 999999,child.pid,child.pid
    state.child_start_identity,state.worker_capabilities = identity,['job-cancellation.v1']
    (job/'status.json').write_text(state.to_json())
    spec = {'job_id':request.job_id,'owner':request.agent_id,'route':request.job_type,
        'reason':request.performance_window,'receipt_path':str(job/'performance-window.json'),
        'argv':request.command_argv,'timeout':None}
    assert performance_window.run(spec,prompt=lambda _: {'defer_minutes':5})==performance_window.DEFER_EXIT
    report = performance_window.read_receipt(job/'performance-window.json',request)
    if committed:
        monkeypatch.setattr(queue,'_apply_performance_deferral',lambda *_:None)
        queue._defer_performance_job(job,request,state,report,{'fixture':'crash before apply'})
    cancel(queue,request,status='running',started=state.started_at)
    restarted = GPUQueue(queue.queue_dir)
    assert restarted.recover_stale()==[request.job_id]
    recovered = restarted.get_job(request.job_id)
    assert recovered.status==JobStatus.CANCELLED
    assert recovered.started_at==123
    row = queue_snapshot(queue.queue_dir,'all')['jobs'][0]
    assert row.get('cancellation_error') is None
    assert row['cancellation']['request_id']=='cancel-1'


@pytest.mark.parametrize('containment', ['pending','continuation-staging'])
@pytest.mark.parametrize('interruption', ['receipt','rename'])
def test_interrupted_successor_cancellation_finishes_before_parent_release(tmp_path,monkeypatch,containment,interruption):
    from gpu_queue import cancellation
    queue = GPUQueue(tmp_path/'queue')
    parent = submit(queue,tmp_path)
    job = (queue.queue_dir/'pending'/parent.job_id).rename(queue.queue_dir/'running'/parent.job_id)
    child = subprocess.Popen([sys.executable,'-c','import time; time.sleep(.1)'],start_new_session=True)
    identity = queue._process_start_identity(child.pid)
    assert identity
    child.wait()
    state = JobState.from_json((job/'status.json').read_text())
    state.status,state.started_at = JobStatus.RUNNING,123
    state.worker_pid,state.child_pid,state.child_process_group = 999999,child.pid,child.pid
    state.child_start_identity,state.worker_capabilities = identity,['job-cancellation.v1']
    (job/'status.json').write_text(state.to_json())
    accepted = cancel(queue,parent,status='running',started=123)
    assert json.loads(Path(accepted['receipt_path']).read_text())['phase']=='completed'
    successor_request = JobRequest(job_type='command',input_path='',
        command_argv=[sys.executable,'-c','pass'],params={'continuation_of':parent.job_id})
    stage = queue.submit(successor_request,_staged=True)
    checkpoint = tmp_path/'saved-checkpoint'
    checkpoint.write_text('saved work must survive')
    commit = {'schema':'gpu-greenroom.checkpoint-handoff.v2','status':'committed',
        'job_id':parent.job_id,'next_job_id':successor_request.job_id,'checkpoint':str(checkpoint),
        'next_request_sha256':hashlib.sha256((stage/'request.json').read_bytes()).hexdigest(),
        'next_state_sha256':hashlib.sha256((stage/'status.json').read_bytes()).hexdigest()}
    (stage/'checkpoint-handoff.json').write_text(json.dumps(commit))
    (job/'checkpoint-handoff.json').write_text(json.dumps(commit))
    successor = stage if containment=='continuation-staging' else stage.rename(queue.queue_dir/'pending'/stage.name)
    original_write,original_rename = cancellation.write_durable,Path.rename
    def write(path,payload):
        if interruption=='receipt' and path==successor/'receipt.json':
            raise OSError('injected successor receipt interruption')
        return original_write(path,payload)
    def rename(path,destination):
        if interruption=='rename' and path==successor:
            raise OSError('injected successor rename interruption')
        return original_rename(path,destination)
    with monkeypatch.context() as fault:
        fault.setattr(cancellation,'write_durable',write)
        fault.setattr(Path,'rename',rename)
        assert GPUQueue(queue.queue_dir).recover_stale()==[]
    restarted = GPUQueue(queue.queue_dir)
    # Exercise continuation recovery before parent recovery for staged state.
    if containment=='continuation-staging':
        restarted.recover_continuations()
    assert restarted.recover_stale()==[parent.job_id]
    terminal = queue.queue_dir/'cancelled'/successor_request.job_id
    assert terminal.is_dir()
    receipt = json.loads((terminal/'receipt.json').read_text())
    assert receipt['status']=='cancelled' and receipt['execution_started'] is False
    assert receipt['parent_job_id']==parent.job_id
    assert restarted.get_job(parent.job_id).status==JobStatus.CANCELLED
    assert checkpoint.read_text()=='saved work must survive'
    assert not list((queue.queue_dir/'pending').iterdir())
    other = submit(queue,tmp_path,'import time; time.sleep(.1)')
    assert restarted.run_one({})
    assert restarted.get_job(other.job_id).status==JobStatus.DONE
