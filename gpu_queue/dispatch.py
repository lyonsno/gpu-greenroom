"""Opt-in two-class selection around the queue's existing execution mutex."""

import json
from collections import deque
import os
from pathlib import Path
import tempfile
from .models import JobState, JobStatus

CAPABILITY = 'two-class-dispatch.v1'
SCHEMA = 'gpu-greenroom.dispatch-policy.v1'


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(payload,stream,indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def validate(value):
    if not isinstance(value,dict) or value.get('schema')!=SCHEMA or value.get('mode') not in {'fifo','two-class'}:
        raise ValueError('invalid dispatch policy schema or mode')
    for key in ('quick_job_types','quick_routes'):
        rules=value.get(key,{})
        if not isinstance(rules,dict) or any(not isinstance(name,str) or not name.strip()
                                           or not isinstance(basis,str) or not basis.strip() for name,basis in rules.items()):
            raise ValueError('quick eligibility needs exact names and nonblank evidence references')
    return {**value,'quick_job_types':value.get('quick_job_types',{}),'quick_routes':value.get('quick_routes',{})}


def policy(root):
    try:
        return validate(json.loads((Path(root)/'dispatch-policy.json').read_text()))
    except FileNotFoundError:
        return {'schema':SCHEMA,'mode':'fifo','quick_job_types':{},'quick_routes':{}}


def classify(request, config):
    if not isinstance(request,dict):
        raise ValueError('job request must be an object')
    declared=request.get('service_class')
    if declared not in (None,'normal','quick'):
        raise ValueError('service_class must be normal or quick')
    basis = (config['quick_routes'].get(request.get('route_identity')) if request.get('command_argv') is not None
             else config['quick_job_types'].get(request.get('job_type')))
    eligible=config['mode']=='two-class' and bool(basis)
    if declared=='quick' and not eligible:
        raise ValueError('requested quick class is not admitted by the current dispatch policy')
    return ('quick' if eligible and declared!='normal' else 'normal'),basis


def last_class(root):
    try:
        value=json.loads((Path(root)/'dispatch-state.json').read_text())
    except FileNotFoundError:
        return 'normal'
    if not isinstance(value,dict) or value.get('schema')!='gpu-greenroom.dispatch-state.v1' or value.get('last_class') not in {'normal','quick'}:
        raise ValueError('dispatch fairness state is unverified')
    return value['last_class']


def pending_order(root, config=None):
    root=Path(root)
    config=config or policy(root)
    rows=[]
    directory=root/'pending'
    for path in directory.iterdir() if directory.is_dir() else ():
        if (path/'status.json').is_file():
            state=JobState.from_json((path/'status.json').read_text())
            if state.job_id!=path.name or state.status!=JobStatus.PENDING:
                raise ValueError('pending job identity or containment is unverified')
            rows.append((state.submitted_at,path))
    rows.sort(key=lambda row:row[0])
    if config['mode']=='fifo':
        return [path for _,path in rows]
    normal,quick=deque(),deque()
    for _,path in rows:
        request=json.loads((path/'request.json').read_text())
        (quick if classify(request,config)[0]=='quick' else normal).append(path)
    result=[]
    previous=last_class(root)
    while normal or quick:
        bucket=normal if normal and (previous=='quick' or not quick) else quick
        path=bucket.popleft()
        result.append(path)
        previous='normal' if bucket is normal else 'quick'
    return result
