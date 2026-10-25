import hashlib
import json
import sys
from uuid import uuid4

import pytest

from gpu_queue import dispatch
from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.operator_server import queue_snapshot
from gpu_queue.queue import GPUQueue, worker_identity


def job(queue, at=1, **fields):
    request = JobRequest(job_type='command', input_path='', agent_id='fixture-owner',
        command_argv=[sys.executable, '-c', 'pass'], command_cwd=str(queue.queue_dir),
        repo_root=str(queue.queue_dir), route_identity='fixture', submitted_at=at, **fields)
    return queue.submit(request)


def identity(path):
    state = JobState.from_json((path/'status.json').read_text())
    return dict(job_id=path.name, request_digest=hashlib.sha256((path/'request.json').read_bytes()).hexdigest(),
                expected_status=state.status.value, expected_started_at=state.started_at)


def act(queue, path, action, **values):
    return queue.operator_job_action(**identity(path), action=action, request_id=str(uuid4()),
                                     requested_by='operator-console', **values)


def cancelled(queue, at=1):
    path = job(queue, at, performance_window='fixture timing')
    queue.cancel(path.name)
    return queue.queue_dir/'cancelled'/path.name


def test_pause_unsupported_visible_and_request_is_saved_without_execution(tmp_path,monkeypatch):
    monkeypatch.setenv('GPU_GREENROOM_PAUSE_SUPPORT_MESSAGE','Please add pause!')
    q = GPUQueue(tmp_path)
    path = job(q)
    row = queue_snapshot(q)['jobs'][0]
    assert row.get('pause_control', {}).get('supported') is False
    result = act(q, path, 'ask-pause')
    saved = json.loads((tmp_path/'job-feedback'/f"{result['event_id']}.json").read_text())
    assert saved['target_diaulos'] == 'fixture-owner'
    assert saved['text'].startswith('Please add pause!')
    assert q.get_job(path.name).status == JobStatus.PENDING


@pytest.mark.parametrize('placement,position', [('previous',0),('second',1),('end',2)])
def test_cancelled_confirmation_requeues_once_with_fresh_identity(tmp_path, placement, position):
    q = GPUQueue(tmp_path)
    old = cancelled(q)
    old_bytes = {p.name:p.read_bytes() for p in old.iterdir() if p.is_file()}
    a, b = job(q, 2), job(q, 3)
    result = act(q, old, 'requeue', placement=placement)
    new = tmp_path/'pending'/result['new_job_id']
    assert new.name != old.name
    assert dispatch.pending_order(tmp_path)[position] == new
    assert {p.name:p.read_bytes() for p in old.iterdir() if p.is_file()} == old_bytes
    assert not (new/'performance-window.json').exists()
    assert json.loads((new/'request.json').read_text())['performance_window'] == 'fixture timing'
    assert act(q, old, 'requeue', placement=placement)['new_job_id'] == new.name
    assert len(list((tmp_path/'pending').iterdir())) == 3


def test_pause_holds_only_selected_job_unless_bypass_unchecked(tmp_path):
    q = GPUQueue(tmp_path)
    selected = job(q, cooperative_checkpoint=True)
    other = job(q, 2)
    act(q, selected, 'pause', allow_bypass=True)
    assert dispatch.pending_order(tmp_path) == [other]
    act(q, selected, 'pause', allow_bypass=False)
    assert dispatch.pending_order(tmp_path) == []
    act(q, selected, 'resume')
    assert dispatch.pending_order(tmp_path) == [selected, other]
    q.pause(owner='separate-owner')
    act(q, selected, 'pause', allow_bypass=True)
    act(q, selected, 'resume')
    assert q.is_paused()


def test_invalid_identity_and_unsupported_pause_do_not_mutate(tmp_path):
    q = GPUQueue(tmp_path)
    path = job(q)
    with pytest.raises(ValueError, match='unsupported'):
        act(q, path, 'pause', allow_bypass=True)
    bad = {**identity(path), 'request_digest':'0'*64}
    with pytest.raises(ValueError, match='changed'):
        q.operator_job_action(**bad, action='ask-pause', request_id=str(uuid4()), requested_by='operator-console')
    assert not (tmp_path/'job-feedback').exists()


def test_old_worker_cannot_skip_held_jobs_or_ignore_placement(tmp_path):
    q = GPUQueue(tmp_path)
    path = job(q, cooperative_checkpoint=True)
    job(q, 2)
    act(q, path, 'pause', allow_bypass=True)
    old = worker_identity()
    old['capabilities'] = [x for x in old['capabilities'] if x != 'operator-job-controls.v1']
    assert not q.run_one({}, claimant=old)
    assert len(list((tmp_path/'pending').iterdir())) == 2


def test_resume_after_successor_publication_releases_successor(tmp_path):
    from gpu_queue.job_controls import inherit_hold
    q = GPUQueue(tmp_path)
    parent = job(q, cooperative_checkpoint=True)
    act(q,parent,'pause',allow_bypass=False)
    child = job(q,2,cooperative_checkpoint=True)
    with q._coordination_lock():
        inherit_hold(q,parent.name,child.name,identity(child)['request_digest'])
    act(q,parent,'resume')
    assert dispatch.pending_order(tmp_path) == [parent,child]


def test_requeue_recovers_publication_after_crash_without_duplication(tmp_path, monkeypatch):
    import gpu_queue.job_controls as controls
    q = GPUQueue(tmp_path)
    path = cancelled(q)
    publish = controls._publish_retry
    monkeypatch.setattr(controls,'_publish_retry',lambda *a: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError, match='crash'):
        act(q,path,'requeue',placement='end')
    assert not list((tmp_path/'pending').iterdir())
    monkeypatch.setattr(controls,'_publish_retry',publish)
    controls.recover(q)
    controls.recover(q)
    result = act(q,path,'requeue',placement='end')
    assert [p.name for p in (tmp_path/'pending').iterdir()] == [result['new_job_id']]


def test_popup_cancel_failed_state_is_recoverable_but_launched_work_is_not(tmp_path):
    from gpu_queue import performance_window
    q = GPUQueue(tmp_path)
    path = job(q,performance_window='fixture timing')
    req = JobRequest.from_json((path/'request.json').read_text())
    spec = {'schema':performance_window.SCHEMA,'job_id':path.name,'owner':req.agent_id,
        'route':req.route_identity,'reason':req.performance_window,'argv':req.command_argv,
        'timeout':None,'receipt_path':str(path/'performance-window.json')}
    assert performance_window.run(spec,prompt=lambda spec:False) == 125
    state = JobState.from_json((path/'status.json').read_text())
    state.status,state.started_at,state.finished_at,state.exit_code = JobStatus.FAILED,1,2,125
    state.failure_phase='performance_window_operator-confirmation'
    (path/'status.json').write_text(state.to_json())
    report = json.loads((path/'performance-window.json').read_text())
    (path/'receipt.json').write_text(json.dumps({'job_id':path.name,'status':'failed','performance_window':report}))
    path = path.rename(tmp_path/'failed'/path.name)
    assert queue_snapshot(q,'recoverable')['jobs'][0]['job_id'] == path.name
    report['workload_started'] = True
    (path/'performance-window.json').write_text(json.dumps(report))
    assert not queue_snapshot(q,'recoverable')['jobs']
    with pytest.raises(ValueError,match='no workload execution'):
        act(q,path,'requeue')


def test_running_pause_yields_actual_child_and_holds_continuation(tmp_path):
    import threading
    import time
    from pathlib import Path
    q = GPUQueue(tmp_path/'q')
    marker, release = tmp_path/'entered',tmp_path/'checkpoint'
    script=tmp_path/'producer.py'
    script.write_text('''import sys,time,pathlib
from gpu_queue.cooperative import _context,yield_if_requested
from gpu_queue.models import JobRequest
q,job,r,s=_context()
pathlib.Path(sys.argv[1]).touch()
while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.01)
yield_if_requested(save_checkpoint=lambda p:{'done':1},quiesce=lambda:True,
 continuation=lambda p:JobRequest(job_type='command',input_path='',agent_id=r.agent_id,
 repo_root=r.repo_root,command_cwd=r.command_cwd,route_identity=r.route_identity,
 command_argv=[sys.executable,'-c','pass']))
''')
    req=JobRequest(job_type='command',input_path='',agent_id='fixture-owner',repo_root=str(tmp_path),
        command_cwd=str(tmp_path),route_identity='checkpoint',cooperative_checkpoint=True,
        command_argv=[sys.executable,str(script),str(marker),str(release)],
        command_env={'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
    q.submit(req)
    worker=threading.Thread(target=q.run_one,args=({},))
    worker.start()
    try:
        for _ in range(1000):
            if marker.exists(): break
            time.sleep(.01)
        assert marker.exists()
        running=q.queue_dir/'running'/req.job_id
        act(q,running,'pause',allow_bypass=True)
        assert queue_snapshot(q,'active')['jobs'][0]['pause_control']['phase']=='pause-requested'
        release.touch()
        worker.join(10)
        assert not worker.is_alive()
        successor=next((q.queue_dir/'pending').iterdir())
        assert q.get_job(req.job_id).status == JobStatus.DONE
        assert dispatch.pending_order(q.queue_dir)==[]
        other=job(q,3)
        assert dispatch.pending_order(q.queue_dir)==[other]
        assert q.run_one({})
        act(q,successor,'resume')
        assert q.run_one({})
        assert q.get_job(successor.name).status==JobStatus.DONE
    finally:
        release.touch()
        worker.join(10)


@pytest.mark.parametrize('read_only,job_control,code',[(True,True,403),(False,False,403),(False,True,200)])
def test_http_action_is_authenticated_and_bound(tmp_path,read_only,job_control,code):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request,urlopen
    from urllib.error import HTTPError
    from gpu_queue.operator_server import make_handler
    q=GPUQueue(tmp_path)
    path=job(q)
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(tmp_path,'secret',
        admission_control=True,job_control=job_control,read_only=read_only))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    data=json.dumps({**identity(path),'request_id':str(uuid4()),'action':'ask-pause'}).encode()
    url=f'http://127.0.0.1:{server.server_address[1]}/api/job-action'
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url,data=data))
        assert error.value.code==401
        req=Request(url,data=data,headers={'Authorization':'Bearer secret'})
        if code==200:
            assert json.load(urlopen(req))['status']=='saved'
        else:
            with pytest.raises(HTTPError) as error: urlopen(req)
            assert error.value.code==code
    finally:
        server.shutdown();server.server_close();thread.join()


def test_current_config_and_quick_moves_preserve_holds(tmp_path):
    q=GPUQueue(tmp_path)
    q.configure_dispatch({'schema':dispatch.SCHEMA,'mode':'two-class'},owner='fixture')
    path=job(q,cooperative_checkpoint=True)
    act(q,path,'pause',allow_bypass=True)
    q.override_dispatch(job_id=path.name,service_class='quick',request_digest=identity(path)['request_digest'],
                        request_id=str(uuid4()),requested_by='operator-console')
    q.configure_dispatch({'schema':dispatch.SCHEMA,'mode':'two-class'},owner='fixture')
    assert dispatch.pending_order(tmp_path)==[]


@pytest.mark.parametrize('placement,index',[('previous',0),('second',1),('end',2)])
def test_restore_placement_uses_waiting_order_even_when_dispatch_held(tmp_path,placement,index):
    q=GPUQueue(tmp_path)
    original=cancelled(q)
    first=job(q,2,cooperative_checkpoint=True)
    last=job(q,3)
    act(q,first,'pause',allow_bypass=False)
    result=act(q,original,'requeue',placement=placement)
    assert dispatch.pending_order(tmp_path)==[]
    act(q,first,'resume')
    expected=[first,last]
    expected.insert(index,tmp_path/'pending'/result['new_job_id'])
    assert dispatch.pending_order(tmp_path)==expected


@pytest.mark.parametrize('changed', [{'input_path':'/two'}, {'job_type':'other'}, {'params':{'setting':2}}, {}])
def test_restore_compares_execution_bearing_identity(tmp_path,changed):
    q=GPUQueue(tmp_path)
    request=JobRequest(job_type='echo',input_path='/one',params={'setting':1},
                       agent_id='owner',performance_window='measurement',submitted_at=1)
    path=q.submit(request)
    q.cancel(path.name)
    from dataclasses import replace
    q.submit(replace(request,job_id='later-job',submitted_at=2,**changed))
    if changed:
        assert act(q,tmp_path/'cancelled'/path.name,'requeue')['status']=='requeued'
    else:
        with pytest.raises(ValueError,match='equivalent replacement'):
            act(q,tmp_path/'cancelled'/path.name,'requeue')


def test_pause_request_retry_after_definite_failure_preserves_identity(tmp_path,monkeypatch):
    from gpu_queue.job_controls import deliver_feedback
    from subprocess import CompletedProcess
    q=GPUQueue(tmp_path)
    path=job(q)
    first=act(q,path,'ask-pause')
    record_path=tmp_path/'job-feedback'/f"{first['event_id']}.json"
    calls=[]
    def adapter(argv,**kwargs):
        calls.append(json.loads(kwargs['input']))
        return CompletedProcess(argv,0,json.dumps({'event_id':first['event_id'],
            'status':'failed' if len(calls)==1 else 'queued','receipt':{'failure_phase':'endpoint-resolution'}}),'')
    monkeypatch.setattr('subprocess.run',adapter)
    deliver_feedback(tmp_path,sys.executable)
    assert json.loads(record_path.read_text())['status']=='failed'
    retry=act(q,path,'ask-pause')
    assert retry['status']=='saved'
    assert retry['event_id']==first['event_id']
    deliver_feedback(tmp_path,sys.executable)
    deliver_feedback(tmp_path,sys.executable)
    final=json.loads(record_path.read_text())
    assert final['status']=='queued'
    assert final['attempts'][0]['status']=='failed'
    assert len(calls)==2
    assert act(q,path,'ask-pause')['status']=='queued'
