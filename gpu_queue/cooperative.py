"""Checkpoint a finite unit and enqueue its continuation without preemption."""

import hashlib
import json
import os
from pathlib import Path
import stat
import time

from . import dispatch
from .gc import RetentionPins
from .models import JobRequest
from .queue import GPUQueue


def _context():
    path=os.environ.get('GPU_GREENROOM_CONTEXT')
    if not path:
        return None
    context=json.loads(Path(path).read_text())
    if context.get('schema')!='gpu-greenroom.job-context.v1':
        raise ValueError('unsupported managed checkpoint context')
    descriptor=os.environ.pop('GPU_GREENROOM_READY_FD',None)
    if descriptor is not None:
        try:
            fd=int(descriptor)
            info=os.fstat(fd)
            if stat.S_ISFIFO(info.st_mode) and info.st_ino==context['ready_inode'] and info.st_dev==context['ready_device']:
                os.read(fd,1)
                os.close(fd)
        except OSError:
            pass
    root=Path(context['queue_dir']).resolve()
    job=root/'running'/context['job_id']
    request_bytes=(job/'request.json').read_bytes()
    state=json.loads((job/'status.json').read_text())
    request=JobRequest.from_json(request_bytes.decode())
    if (context['job_id']!=job.name or state.get('job_id')!=job.name or state.get('status')!='running'
            or hashlib.sha256(request_bytes).hexdigest()!=context['request_sha256']
            or state.get('child_process_group')!=os.getpgrp()
            or not state.get('child_pid') or GPUQueue._process_start_identity(state['child_pid'])!=state.get('child_start_identity')
            or not request.cooperative_checkpoint):
        raise ValueError('managed checkpoint ownership or request identity is unverified')
    return GPUQueue(root),job,request,state


def load_checkpoint():
    context=_context()
    path=os.environ.get('GPU_GREENROOM_RESUME_CHECKPOINT')
    if context is None or not path:
        return None
    queue,job,request,_=context
    raw=Path(path).read_bytes()
    value=json.loads(raw)
    if (value.get('schema')!='gpu-greenroom.checkpoint.v1' or value.get('owner')!=request.agent_id
            or value.get('queue_dir')!=str(queue.queue_dir.resolve())
            or value.get('job_id')!=request.params.get('continuation_of')
            or value.get('route_identity')!=request.route_identity):
        raise ValueError('checkpoint does not belong to this continuation')
    if hashlib.sha256(raw).hexdigest()!=request.params.get('checkpoint_sha256'):
        raise ValueError('checkpoint bytes changed after continuation submission')
    return value['payload']


def yield_if_requested(*,save_checkpoint,quiesce,continuation,on_submitted=None):
    context=_context()
    if context is None:
        return False
    queue,job,request,state=context
    if list(queue.control_receipts_dir.glob('*-'+request.job_id+'-signal-*.json')):
        return False
    config=dispatch.policy(queue.queue_dir)
    pending=dispatch.pending_order(queue.queue_dir,config)
    quick_waiting=bool(pending and dispatch.classify(json.loads((pending[0]/'request.json').read_text()),config)[0]=='quick')
    if not queue.is_paused() and (state.get('dispatch',{}).get('service_class')!='normal' or not quick_waiting):
        return False
    directory=queue.queue_dir/'outputs'/request.job_id/'checkpoints'
    pins=RetentionPins(queue.queue_dir)
    existing=pins.entries().get(request.job_id)
    if existing and existing.get('owner') not in (None,request.agent_id):
        raise ValueError('checkpoint output pin is owned by a different producer')
    if not existing or existing.get('until') is not None:
        pins.pin(request.job_id,owner=request.agent_id,
                 reason='Checkpoint recovery state; release explicitly after it is no longer needed')
    directory.mkdir(parents=True,exist_ok=True)
    payload=save_checkpoint(directory)
    if not isinstance(payload,dict):
        raise ValueError('save_checkpoint must return a JSON descriptor of durable saved state')
    checkpoint=directory/'checkpoint.json'
    dispatch.atomic_write(checkpoint,{'schema':'gpu-greenroom.checkpoint.v1','job_id':request.job_id,
        'queue_dir':str(queue.queue_dir.resolve()),'owner':request.agent_id,'route_identity':request.route_identity,
        'saved_at':time.time(),'payload':payload})
    if quiesce() is not True:
        return False
    next_request=continuation(checkpoint)
    if next_request is None:
        raise SystemExit(0)
    if (not isinstance(next_request,JobRequest) or next_request.job_id==request.job_id
            or next_request.agent_id!=request.agent_id or next_request.repo_root!=request.repo_root
            or next_request.route_identity!=request.route_identity or next_request.command_cwd!=request.command_cwd
            or not next_request.command_argv or next_request.command_argv[0]!=request.command_argv[0]):
        raise ValueError('continuation must preserve owner, repo, route and executable identity')
    next_request.cooperative_checkpoint=True
    if next_request.command_timeout is None:
        next_request.command_timeout=request.command_timeout
    next_request.params={**next_request.params,'continuation_of':request.job_id,
                         'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    overlay=request.command_env if next_request.command_env is None else next_request.command_env
    next_request.command_env={**(overlay or {}),'GPU_GREENROOM_RESUME_CHECKPOINT':str(checkpoint.resolve())}
    with queue._coordination_lock():
        if queue.get_job(next_request.job_id) is not None:
            raise ValueError('continuation job identity is already in use')
        queue.submit(next_request)
    try:
        if on_submitted:
            on_submitted(next_request)
    except Exception:
        queue.cancel(next_request.job_id)
        raise
    dispatch.atomic_write(job/'checkpoint-handoff.json',{'schema':'gpu-greenroom.checkpoint-handoff.v1',
        'job_id':request.job_id,'next_job_id':next_request.job_id,'checkpoint':str(checkpoint.resolve()),
        'quiescence_authority':'producer-confirmed; worker verifies process-group exit','submitted_at':time.time()})
    raise SystemExit(0)
