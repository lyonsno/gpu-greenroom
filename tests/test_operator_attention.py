import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import Mock
import os
import subprocess
import sys
from datetime import datetime

import pytest

from gpu_queue import performance_window


def waiting(root, attempt='a'*32):
    value = {'schema': performance_window.SCHEMA, 'job_id': 'one', 'owner': 'owner',
             'route': 'cpu/test', 'reason': 'measure', 'status': 'waiting',
             'attempt_id': attempt, 'requested_at': 100, 'workload_started': False}
    (root/'performance-window.json').write_text(json.dumps(value))
    return value


def test_native_and_web_share_one_durable_attempt_decision(tmp_path):
    waiting(tmp_path)
    decide = getattr(performance_window, 'decide', None)
    assert callable(decide), 'Native and web have no shared attempt-bound decision authority'
    barrier = threading.Barrier(2)
    def race(action):
        barrier.wait()
        return decide(tmp_path, attempt_id='a'*32, request_id=action,
                      action=action, source=action, defer_minutes=5)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(race, ['start', 'defer']))
    assert results[0] == results[1]
    assert json.loads(next(tmp_path.glob('performance-decision-*.json')).read_text()) == results[0]
    assert decide(tmp_path, attempt_id='a'*32, request_id='retry', action='cancel', source='web') == results[0]
    with pytest.raises(ValueError, match='attempt'):
        decide(tmp_path, attempt_id='b'*32, request_id='wrong', action='start', source='web')


def test_resolved_web_decision_is_consumed_without_stale_native_answer(tmp_path):
    decide = getattr(performance_window, 'decide', None)
    assert callable(decide), 'Web answer cannot reach the performance runner'
    def prompt(spec):
        decide(tmp_path, attempt_id=spec['attempt_id'], request_id='web-defer',
               action='defer', source='web', defer_minutes=7)
        return True  # Late native button cannot override the accepted web answer.
    spec = {'job_id': 'one', 'owner': 'owner', 'route': 'cpu/test', 'reason': 'measure',
            'receipt_path': str(tmp_path/'performance-window.json'), 'attempt_id': 'a'*32,
            'argv': ['must-not-launch'], 'timeout': None}
    assert performance_window.run(spec, prompt=prompt) == performance_window.DEFER_EXIT
    result = json.loads((tmp_path/'performance-window.json').read_text())
    assert result['status'] == 'deferred' and result['defer_seconds'] == 420


def test_overrun_escalation_is_durable_and_never_execution_authority(tmp_path):
    from gpu_queue import operator_server
    attention = getattr(operator_server, 'attention_snapshot', None)
    assert callable(attention), 'Monitor has no persistent operator attention producer'
    job = {'job_id':'one', 'agent_id':'owner', 'status':'running', 'consistent':True,
           'started_at':100, 'request_digest':'d'*64, 'runtime_hint':{'expected_seconds':25},
           'route_identity':'cpu/test', 'execution_wall_seconds':1000}
    snapshot = {'jobs':[job], 'paused':False}
    first = attention(tmp_path, snapshot, 'http://127.0.0.1:8766/', now=1100)
    assert first['items'] == [], 'Notify owner before escalating to operator'
    assert len(list((tmp_path/'job-feedback').glob('*.json'))) == 1
    event = json.loads(next((tmp_path/'job-feedback').glob('*.json')).read_text())
    assert event['kind'] == 'runtime-overrun' and event['status'] == 'saved'
    event['status'] = 'failed'  # Mail failure neither cancels nor prevents escalation.
    next((tmp_path/'job-feedback').glob('*.json')).write_text(json.dumps(event))
    second = attention(tmp_path, snapshot, 'http://127.0.0.1:8766/', now=1401)
    assert len(second['items']) == 1
    assert second['items'][0]['kind'] == 'runtime-overrun'
    assert second['items'][0]['owner_delivery'] == 'failed'
    assert not list(tmp_path.rglob('cancel-request.json'))
    assert attention(tmp_path, snapshot, 'http://127.0.0.1:8766/', now=1402)['items'][0]['id'] == second['items'][0]['id']
    assert attention(tmp_path, {'jobs':[]}, 'http://127.0.0.1:8766/', now=1403)['items'] == []


def test_operator_wait_is_not_compute_overrun(tmp_path):
    from gpu_queue import operator_server
    attention = getattr(operator_server, 'attention_snapshot', None)
    assert callable(attention), 'Operator waiting state has no independent attention'
    job = {'job_id':'one', 'agent_id':'owner', 'status':'running', 'consistent':True,
           'started_at':100, 'request_digest':'d'*64, 'runtime_hint':{'expected_seconds':15},
           'performance_window_reason':'measure', 'performance_window':{
               'status':'waiting', 'attempt_id':'a'*32, 'requested_at':101}}
    result = attention(tmp_path, {'jobs':[job]}, 'http://127.0.0.1:8766/', now=3600)
    assert [item['kind'] for item in result['items']] == ['performance-confirmation']
    assert not list((tmp_path/'job-feedback').glob('*.json'))


def test_web_winner_cancels_the_exact_native_notification(tmp_path,monkeypatch):
    waiting(tmp_path)
    cf = Mock()
    cf.CFUserNotificationCreate.return_value = 99
    cf.CFUserNotificationCancel.return_value = 0
    cf.CFStringCreateWithCString.return_value = 10
    monkeypatch.setattr(performance_window.sys,'platform','darwin')
    monkeypatch.setattr(performance_window.ctypes,'CDLL',lambda _:cf)
    monkeypatch.setattr(performance_window.ctypes.c_void_p,'in_dll',lambda *_:Mock(value=100))
    def receive(*args):
        assert args[0:2] == (99,.25), 'Receive slice is not an operator timeout'
        performance_window.decide(tmp_path,attempt_id='a'*32,request_id='web',action='defer',source='web')
        return 0x10004003  # SDK MACH_RCV_TIMED_OUT; observed on the host, not a guessed fixture.
    cf.CFUserNotificationReceiveResponse.side_effect = receive
    answer = performance_window.native_dialog({'job_id':'one','owner':'owner','route':'cpu/test',
        'reason':'measure','attempt_id':'a'*32,'receipt_path':str(tmp_path/'performance-window.json')})
    assert answer == {'defer_minutes':5.0}
    cf.CFUserNotificationCancel.assert_called_once_with(99)
    assert json.loads((tmp_path/('performance-native-'+'a'*32+'.json')).read_text())['status']=='dismissed-after-shared-decision'


def test_owner_progress_suppresses_escalation_only_for_its_exact_attempt(tmp_path):
    from gpu_queue import operator_attention as a
    row = {'job_id':'one','agent_id':'owner','status':'running','consistent':True,
           'started_at':100,'request_digest':'d'*64,'runtime_hint':{'expected_seconds':10}}
    a.snapshot(tmp_path,{'jobs':[row]},'http://127.0.0.1:1/',now=400)
    job = tmp_path/'running'/'one'
    job.mkdir(parents=True)
    report = {'schema':'gpu-greenroom.runtime-progress.v1','job_id':'one',
              'request_digest':'d'*64,'started_at':100,'reported_at':650,
              'remaining_seconds':500,'note':'Computing next retained sample'}
    (job/'runtime-progress.json').write_text(json.dumps(report))
    assert a.snapshot(tmp_path,{'jobs':[row]},'http://127.0.0.1:1/',now=701)['items']==[]
    report['started_at']=99
    (job/'runtime-progress.json').write_text(json.dumps(report))
    assert a.snapshot(tmp_path,{'jobs':[row]},'http://127.0.0.1:1/',now=701)['items'][0]['owner_report'] is None
    assert not list(tmp_path.rglob('cancel-request.json'))


def test_http_performance_decision_is_authenticated_and_binds_current_request(tmp_path):
    from gpu_queue.queue import GPUQueue
    from gpu_queue.models import JobRequest,JobStatus
    from gpu_queue import dispatch
    from gpu_queue.operator_server import make_handler
    queue = GPUQueue(tmp_path/'queue')
    request = JobRequest(job_type='command',input_path='',performance_window='measure',
                         route_identity='cpu/test',agent_id='owner')
    path = queue.submit(request)
    job = queue.queue_dir/'running'/request.job_id
    path.rename(job)
    state = queue.get_job(request.job_id)
    state.status,state.started_at=JobStatus.RUNNING,100.
    (job/'status.json').write_text(state.to_json())
    report = waiting(job)
    report.update(job_id=request.job_id,decision_protocol='shared.v1',quietness_proved=False,
                  authority='operator-acknowledgement-only')
    (job/'performance-window.json').write_text(json.dumps(report))
    server = ThreadingHTTPServer(('127.0.0.1',0),make_handler(queue,'secret',local_operator=True,job_control=True))
    thread = threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    body = {'job_id':request.job_id,'request_digest':dispatch.read_request(job/'request.json')[1],
            'expected_started_at':100.,'attempt_id':'a'*32,'request_id':'web','action':'defer','defer_minutes':7}
    def send(value,token='secret'):
        return urlopen(Request(base+'/api/performance-decision',data=json.dumps(value).encode(),
            headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'}))
    try:
        for filename in ('status.json','request.json','performance-decision-'+'a'*32+'.json'):
            damaged=job/filename
            original=damaged.read_bytes() if damaged.exists() else None
            try:
                for malformed in (None,[],{}):
                    damaged.write_text(json.dumps(malformed))
                    with pytest.raises(HTTPError) as error:
                        send(body)
                    assert error.value.code==409
                    assert json.load(error.value)['error']
            finally:
                if original is None:
                    damaged.unlink(missing_ok=True)
                else:
                    damaged.write_bytes(original)
        for changes,code in [({'request_digest':'e'*64},409),({'expected_started_at':99},409),
                              ({'attempt_id':'b'*32},409),({'defer_minutes':False},409)]:
            with pytest.raises(HTTPError) as error:
                send({**body,**changes})
            assert error.value.code==code
        with pytest.raises(HTTPError) as error:
            send(body,'wrong')
        assert error.value.code==401
        with send(body) as response:
            answer=json.load(response)
        assert answer['action']=='defer' and answer['defer_minutes']==7
        with send(body) as response:
            assert json.load(response)==answer
        state.status,state.started_at=JobStatus.DEFERRED,None
        (job/'status.json').write_text(state.to_json())
        job.rename(queue.queue_dir/'deferred'/job.name)
        with send(body) as response:
            assert json.load(response)==answer, 'A completed response retry must not depend on running containment'
        assert not (job/'executed').exists()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize('diagnostic',['record_error','performance_window_error','missing-performance'])
def test_unreadable_running_observation_cannot_claim_fresh_empty_attention(tmp_path,diagnostic):
    from gpu_queue.operator_attention import snapshot
    row = {'job_id':'one','status':'running','containment_status':'running','consistent':True,
           'request_digest':'d'*64,'performance_window_reason':'measure'}
    if diagnostic != 'missing-performance':
        row[diagnostic]='unreadable record'
    if diagnostic=='record_error':
        row['consistent']=False
    result=snapshot(tmp_path,{'jobs':[row]},'http://127.0.0.1:1/',now=1000)
    assert result['errors'], 'Incomplete queue evidence was published as healthy empty attention'
    assert result['items'][0]['job_id']=='one'
    assert result['queue']['running']==1
    assert datetime.fromisoformat(result['valid_until']).timestamp()==1000


@pytest.mark.parametrize('damage',['missing-status','missing-request','corrupt-request'])
def test_actual_queue_reader_preserves_missing_record_attention(tmp_path,damage):
    from gpu_queue.queue import GPUQueue
    from gpu_queue.models import JobRequest,JobStatus
    from gpu_queue.operator_server import queue_snapshot
    from gpu_queue.operator_attention import snapshot
    queue=GPUQueue(tmp_path/'queue')
    request=JobRequest(job_type='command',input_path='',performance_window='measure',
                       agent_id='owner',route_identity='cpu/test')
    pending=queue.submit(request)
    state=queue.get_job(request.job_id)
    state.status,state.started_at=JobStatus.RUNNING,100.
    (pending/'status.json').write_text(state.to_json())
    job=queue.queue_dir/'running'/request.job_id
    pending.rename(job)
    perf=waiting(job)
    perf.update(job_id=request.job_id,quietness_proved=False,authority='operator-acknowledgement-only')
    (job/'performance-window.json').write_text(json.dumps(perf))
    assert snapshot(queue.queue_dir,queue_snapshot(queue.queue_dir,'active'), 'http://127.0.0.1:1/')['items'][0]['kind']=='performance-confirmation'
    if damage=='corrupt-request':
        (job/'request.json').write_text('{')
    else:
        (job/('status.json' if damage=='missing-status' else 'request.json')).unlink()
    result=snapshot(queue.queue_dir,queue_snapshot(queue.queue_dir,'active'), 'http://127.0.0.1:1/',now=1000)
    assert result['errors'],'Actual reader composition concealed missing record uncertainty'
    assert result['queue']['running']==1
    assert result['items'][0]['job_id']==request.job_id
    assert result['valid_until']==result['observed_at']


@pytest.mark.parametrize('invalid',[None,[],{'schema':'gpu-greenroom.operator-attention.v1'},
    {'schema':'gpu-greenroom.operator-attention.v1','valid_until':'2099-01-01T00:00:00Z'}])
def test_attention_api_returns_structured_unavailable_for_invalid_persisted_shape(tmp_path,invalid):
    from gpu_queue.queue import GPUQueue
    from gpu_queue.operator_server import make_handler
    q=GPUQueue(tmp_path/'queue')
    (q.queue_dir/'operator-attention.json').write_text(json.dumps(invalid))
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(q,'secret',local_operator=True))
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/attention',headers={'Authorization':'Bearer secret'}))
        assert error.value.code==503
        assert json.load(error.value)['error']
    finally:
        server.shutdown();thread.join();server.server_close()


def test_native_witness_worker_refusal_is_failed_process_and_primary_report(tmp_path):
    result=subprocess.run([sys.executable,'-m','tests.native_attention_witness','--output',str(tmp_path/'witness'),
        '--port','0','--native','--auto-web-defer'],capture_output=True,text=True,
        env={**os.environ,'GPU_GREENROOM_WORKER_CAPABILITIES':''})
    assert result.returncode!=0, 'Failed automatic conformance returned process success'
    primary=json.loads((tmp_path/'witness'/'witness.json').read_text())
    assert primary['phase']=='failed'
    assert primary['error']
    assert json.loads((tmp_path/'witness'/'native-conformance.json').read_text())['phase']=='failed'


def test_attention_schema_accepts_additions_but_not_retyped_or_missing_authority(tmp_path):
    from gpu_queue.operator_attention import snapshot,validate
    value=snapshot(tmp_path,{'jobs':[]},'http://127.0.0.1:8766/',now=1000)
    assert validate({**value,'future_additive_field':{'supported':True}})
    for replacement in ({'items':[None]}, {'items':[{'id':'x','status':'waiting','title':'test','target_url':'javascript:alert(1)'}]},
                        {'queue':{'running':True,'waiting':0}}, {'observed_at':'1970-01-01T00:16:40'},
                        {'items':[{'id':'x','title':'missing status','target_url':'http://127.0.0.1/'}]}):
        with pytest.raises(ValueError):
            validate({**value,**replacement})


@pytest.mark.parametrize('pause_fails',[False,True])
def test_attention_pause_dismisses_popup_only_after_pause_is_accepted(pause_fails):
    from gpu_queue.operator_server import PAGE
    script=PAGE.split('<script>',1)[1].split('</script>',1)[0]
    script=script[:script.index("document.querySelector('#pause').onclick")]
    setup="globalThis.location={hash:'',pathname:'/',search:''};globalThis.sessionStorage={getItem:()=>''};"
    action='''
attention={valid_until:new Date(Date.now()+15000).toISOString()};queueAvailable=true;
last={jobs:[{job_id:'one',request_digest:'d'.repeat(64),status:'running',started_at:100}]};
const calls=[];post=async(path,body)=>{calls.push(path);if(FAIL&&path==='/api/job-action')throw Error('pause refused')};
err=()=>{};loadAttention=async()=>{};
(async()=>{if(typeof pauseAttention!=='function')throw Error('No shared pause-and-dismiss action');await pauseAttention('one','item');console.log(JSON.stringify(calls))})();
'''.replace('FAIL','true' if pause_fails else 'false')
    result=subprocess.run(['node','-e',setup+script+action],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)==(['/api/job-action'] if pause_fails else ['/api/job-action','/api/attention-acknowledge'])
