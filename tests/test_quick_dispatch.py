import json
import sys

import pytest

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue, worker_identity


def policy():
    return {'schema':'gpu-greenroom.dispatch-policy.v1','mode':'two-class',
            'quick_job_types':{'capture':'observed CPU fixture timings'},'quick_routes':{}}


def submit(queue, kind, stamp):
    request=JobRequest(job_type=kind,input_path='fixture',submitted_at=stamp,agent_id='example-agent')
    queue.submit(request)
    return request.job_id


def test_two_class_dispatch_allows_one_quick_then_normal_without_starvation(tmp_path):
    queue=GPUQueue(tmp_path/'queue')
    acknowledgement=queue.configure_dispatch(policy(),owner='fixture-operator')
    normal=submit(queue,'long',1)
    quick=submit(queue,'capture',2)
    quick2=submit(queue,'capture',3)
    config={kind:{'cmd':[sys.executable,'-c','pass']} for kind in ('long','capture')}
    assert queue.run_one(config)
    assert (queue.queue_dir/'done'/quick).is_dir()
    assert queue.run_one(config)
    assert (queue.queue_dir/'done'/normal).is_dir()
    assert queue.run_one(config)
    receipt=json.loads((queue.queue_dir/'done'/quick2/'receipt.json').read_text())
    assert receipt['dispatch']['service_class']=='quick'
    assert receipt['dispatch']['policy_epoch']==acknowledgement['epoch']


def test_fifo_default_and_pause_lease_boundaries_remain_authoritative(tmp_path):
    queue=GPUQueue(tmp_path/'queue')
    normal=submit(queue,'long',1)
    submit(queue,'capture',2)
    assert queue._next_pending().name==normal
    queue.configure_dispatch(policy(),owner='fixture-operator')
    queue.pause(owner='fixture-operator')
    assert not queue.run_one({'capture':{'cmd':[sys.executable,'-c','pass']}})
    assert not list((queue.queue_dir/'running').iterdir())


def test_policy_and_fairness_state_are_durable_and_old_claimant_cannot_advance(tmp_path):
    queue=GPUQueue(tmp_path/'queue')
    queue.configure_dispatch(policy(),owner='fixture-operator')
    normal=submit(queue,'long',1)
    quick=submit(queue,'capture',2)
    config={kind:{'cmd':[sys.executable,'-c','pass']} for kind in ('long','capture')}
    old={**worker_identity(),'capabilities':['structured-command.v1']}
    assert not queue.run_one(config,claimant=old)
    assert queue._next_pending().name==quick
    assert queue.run_one(config)
    assert GPUQueue(queue.queue_dir)._next_pending().name==normal


def test_unknown_quick_eligibility_and_corrupt_policy_fail_before_execution(tmp_path):
    queue=GPUQueue(tmp_path/'queue')
    with pytest.raises(ValueError):
        queue.configure_dispatch({**policy(),'quick_job_types':{'capture':''}},owner='fixture-operator')
    queue.configure_dispatch(policy(),owner='fixture-operator')
    req=JobRequest(job_type='long',input_path='fixture',service_class='quick')
    with pytest.raises(ValueError):
        queue.submit(req)
    assert not list((queue.queue_dir/'pending').iterdir())
    submit(queue,'long',1)
    (queue.queue_dir/'dispatch-policy.json').write_text('{}')
    assert not queue.run_one({'long':{'cmd':[sys.executable,'-c','pass']}})
    error=json.loads((queue.queue_dir/'dispatch-error.json').read_text())
    assert error['phase']=='dispatch-policy'
    assert not list((queue.queue_dir/'running').iterdir())


def test_active_lease_and_ownership_unknown_cannot_be_bypassed_by_quick_jobs(tmp_path):
    import os,time
    queue=GPUQueue(tmp_path/'queue')
    queue.configure_dispatch(policy(),owner='fixture-operator')
    submit(queue,'capture',1)
    queue.claim_lease(owner='fixture-holder',agent_id='fixture-holder',repo_root='/fixture',pid=os.getpid(),
                      process_group=os.getpgrp(),effective_route='fixture/live',backend='fixture',device='fixture:0',
                      profile='fixture',supports_checkpoints=True,interruptible=False,ttl_seconds=.01)
    config={'capture':{'cmd':[sys.executable,'-c','pass']}}
    assert not queue.run_one(config)
    time.sleep(.02)
    assert not queue.run_one(config)
    assert queue.lease_status().lifecycle_state.value=='ownership_unknown'
    assert not list((queue.queue_dir/'running').iterdir())


@pytest.mark.parametrize('file,value',[('dispatch-policy.json',{'schema':'gpu-greenroom.dispatch-policy.v1','mode':[]}),
                                       ('dispatch-state.json',{'schema':'gpu-greenroom.dispatch-state.v1','last_class':[]})])
def test_malformed_dispatch_never_takes_down_operator_snapshot(tmp_path,file,value):
    from gpu_queue.operator_server import queue_snapshot
    queue=GPUQueue(tmp_path/'queue')
    (queue.queue_dir/file).write_text(json.dumps(value))
    snapshot=queue_snapshot(queue)
    assert snapshot['dispatch']['error']
    assert snapshot['jobs']==[]
