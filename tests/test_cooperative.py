import json
import os
import subprocess
import sys
import threading
import time
import pytest

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


@pytest.mark.parametrize('tamper',[False,True])
def test_native_checkpoint_continuation_releases_mutex_before_quick_then_resumes(tmp_path,tamper):
    queue=GPUQueue(tmp_path/'queue')
    queue.configure_dispatch({'schema':'gpu-greenroom.dispatch-policy.v1','mode':'two-class',
                              'quick_job_types':{},'quick_routes':{'fixture/quick':'observed CPU fixture'}},owner='fixture-operator')
    script=tmp_path/'work.py'
    script.write_text('''import json,os,pathlib,sys,time
from gpu_queue.cooperative import load_checkpoint,yield_if_requested
from gpu_queue.models import JobRequest
state=load_checkpoint() or {'unit':0}
pathlib.Path(sys.argv[1]).write_text(str(state['unit']))
if state['unit']==0:
    while not pathlib.Path(sys.argv[2]).exists(): time.sleep(.02)
    def next_request(checkpoint):
        return JobRequest(job_type='command',input_path='',command_argv=[sys.executable,__file__,*sys.argv[1:]],
            command_cwd=os.getcwd(),repo_root=os.getcwd(),route_identity='fixture/long',agent_id='fixture-agent',
            cooperative_checkpoint=True)
    yield_if_requested(save_checkpoint=lambda directory:{'unit':1},quiesce=lambda:True,continuation=next_request)
''')
    marker,proceed=tmp_path/'unit',tmp_path/'proceed'
    parent=JobRequest(job_type='command',input_path='',command_argv=[sys.executable,str(script),str(marker),str(proceed)],
                      command_cwd=str(tmp_path),repo_root=str(tmp_path),route_identity='fixture/long',
                      agent_id='fixture-agent',cooperative_checkpoint=True,
                      command_env={'PYTHONPATH':str(__import__('pathlib').Path(__file__).resolve().parent.parent)})
    queue.submit(parent)
    worker=threading.Thread(target=queue.run_one,args=({},))
    worker.start()
    while not marker.exists() and worker.is_alive(): time.sleep(.01)
    assert marker.read_text()=='0'
    quick=JobRequest(job_type='command',input_path='',command_argv=[sys.executable,'-c','pass'],
                     command_cwd=str(tmp_path),route_identity='fixture/quick')
    queue.submit(quick)
    proceed.touch()
    worker.join()
    assert (queue.queue_dir/'done'/parent.job_id).is_dir()
    pending=list((queue.queue_dir/'pending').iterdir())
    assert len(pending)==2
    continuation=next(path for path in pending if path.name!=quick.job_id)
    receipt=json.loads((queue.queue_dir/'done'/parent.job_id/'checkpoint-handoff.json').read_text())
    if tamper:
        path=__import__('pathlib').Path(receipt['checkpoint'])
        value=json.loads(path.read_text())
        value['payload']['unit']=3
        path.write_text(json.dumps(value))
    assert queue.run_one({})
    assert (queue.queue_dir/'done'/quick.job_id).is_dir()
    assert queue.run_one({})
    assert marker.read_text()==('0' if tamper else '1')
    assert (queue.queue_dir/('failed' if tamper else 'done')/continuation.name).is_dir()
    if tamper:
        assert 'checkpoint bytes changed' in (queue.queue_dir/'failed'/continuation.name/'stderr.log').read_text()
    assert receipt['next_job_id']==continuation.name
    assert receipt['quiescence_authority']=='producer-confirmed; worker verifies process-group exit'
    pins=json.loads((queue.queue_dir/'retention/pins.json').read_text())
    assert pins['pins'][parent.job_id]['until'] is None


def test_checkpoint_hook_does_not_force_ordinary_processes_into_queue():
    result=subprocess.run([sys.executable,'-c',"from gpu_queue.cooperative import yield_if_requested; assert yield_if_requested(save_checkpoint=lambda d:{},quiesce=lambda:True,continuation=lambda p:None) is False"],
                          env={key:value for key,value in os.environ.items() if key!='GPU_GREENROOM_CONTEXT'},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('case',['decline','wrong-owner','registration-failure','wrong-context','registration-exit','registration-crash'])
def test_checkpoint_negative_paths_preserve_ownership_and_do_not_run_successor(tmp_path,case):
    from pathlib import Path
    queue=GPUQueue(tmp_path/'queue')
    script=tmp_path/'negative.py'
    marker=tmp_path/'result.json'
    script.write_text('''import json,os,pathlib,sys
from gpu_queue.cooperative import load_checkpoint,yield_if_requested
from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue
load_checkpoint()
context=pathlib.Path(os.environ['GPU_GREENROOM_CONTEXT'])
payload=json.loads(context.read_text())
queue=GPUQueue(payload['queue_dir'])
queue.pause(owner='fixture-operator')
case=sys.argv[1]
if case=='wrong-context':
    payload['request_sha256']='wrong'
    context.write_text(json.dumps(payload))
def next_request(path):
    return JobRequest(job_type='command',input_path='',command_argv=[sys.executable,__file__,*sys.argv[1:]],
        command_cwd=os.getcwd(),repo_root=os.getcwd(),route_identity='fixture/negative',
        agent_id='wrong-owner' if case=='wrong-owner' else 'fixture-agent',cooperative_checkpoint=True)
def registered(request):
    if case=='registration-failure': raise RuntimeError('fixture registration failed')
    if case=='registration-exit': raise SystemExit(9)
    if case=='registration-crash': os._exit(9)
try:
    result=yield_if_requested(save_checkpoint=lambda d:{'unit':1},quiesce=lambda:case!='decline',
        continuation=next_request,on_submitted=registered)
    outcome={'result':result}
except Exception as error:
    outcome={'error':str(error)}
pathlib.Path(sys.argv[2]).write_text(json.dumps(outcome))
''')
    request=JobRequest(job_type='command',input_path='',command_argv=[sys.executable,str(script),case,str(marker)],
                       command_cwd=str(tmp_path),repo_root=str(tmp_path),route_identity='fixture/negative',
                       agent_id='fixture-agent',cooperative_checkpoint=True,
                       command_env={'PYTHONPATH':str(Path(__file__).resolve().parent.parent)})
    queue.submit(request)
    assert queue.run_one({})
    if case not in ('registration-exit','registration-crash'):
        result=json.loads(marker.read_text())
        if case=='decline':
            assert result=={'result':False}
        else:
            assert 'error' in result
    assert not list((queue.queue_dir/'pending').iterdir())
    assert not list((queue.queue_dir/'running').iterdir())
    assert not list((queue.queue_dir/'cancelled').iterdir())
    stage=queue.queue_dir/'continuation-staging'
    prepared=list(stage.iterdir()) if stage.exists() else []
    assert len(prepared)==(1 if case in ('registration-failure','registration-exit','registration-crash') else 0)


def test_fresh_cooperative_job_clears_inherited_resume_state(tmp_path,monkeypatch):
    queue=GPUQueue(tmp_path/'queue')
    monkeypatch.setenv('GPU_GREENROOM_RESUME_CHECKPOINT','/wrong/stale/checkpoint.json')
    req=JobRequest(job_type='command',input_path='',agent_id='fixture-agent',cooperative_checkpoint=True,
                   command_argv=[sys.executable,'-c',"import os; assert os.environ.get('GPU_GREENROOM_RESUME_CHECKPOINT') is None"])
    queue.submit(req)
    assert queue.run_one({})
    assert (queue.queue_dir/'done'/req.job_id).is_dir()


def test_committed_crash_recovers_publication_without_rerunning_parent(tmp_path):
    from pathlib import Path
    queue=GPUQueue(tmp_path/'queue')
    code='''import os,sys
from gpu_queue.cooperative import _context,yield_if_requested
from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue
queue,job,request,state=_context()
queue.pause(owner='fixture-operator')
GPUQueue.publish_continuation=lambda *a,**k:os._exit(9)
def successor(path):
    return JobRequest(job_type='command',input_path='',agent_id=request.agent_id,
        repo_root=request.repo_root,command_cwd=request.command_cwd,route_identity=request.route_identity,
        command_argv=[sys.executable,'-c',"print('registered successor')"])
yield_if_requested(save_checkpoint=lambda d:{'unit':1},quiesce=lambda:True,continuation=successor)
'''
    req=JobRequest(job_type='command',input_path='',agent_id='fixture-agent',cooperative_checkpoint=True,
                   repo_root=str(tmp_path),command_cwd=str(tmp_path),route_identity='fixture/crash',
                   command_argv=[sys.executable,'-c',code],command_env={'PYTHONPATH':str(Path(__file__).resolve().parent.parent)})
    queue.submit(req)
    assert queue.run_one({})
    assert not list((queue.queue_dir/'pending').iterdir())
    prepared=list((queue.queue_dir/'continuation-staging').iterdir())
    assert len(prepared)==1
    ids=queue.recover_continuations()
    assert ids==[prepared[0].name]
    assert queue.recover_continuations()==[]
    assert queue.publish_continuation(prepared[0]).name==ids[0]
    assert not queue.run_one({})
    queue.resume(owner='fixture-operator')
    assert queue.run_one({})
    assert (queue.queue_dir/'done'/ids[0]/'stdout.log').read_text().strip()=='registered successor'
