"""Opt-in two-class selection around the queue's existing execution mutex."""

import json
import hashlib
import math
import re
import uuid
from collections import deque
import os
from pathlib import Path
import tempfile
from .models import JobState, JobStatus
from . import runtime_hints

CAPABILITY = 'two-class-dispatch.v1'
SCHEMA = 'gpu-greenroom.dispatch-policy.v1'
OVERRIDE_SCHEMA = 'gpu-greenroom.dispatch-policy.v2'
OVERRIDE_CAPABILITY = 'operator-dispatch-override.v1'
HISTORY_SCHEMA = 'gpu-greenroom.dispatch-policy.v3'
HISTORY_CAPABILITY = 'runtime-profile.v1'
HINT_SCHEMA = 'gpu-greenroom.dispatch-policy.v4'
CONTROL_SCHEMA = 'gpu-greenroom.dispatch-policy.v5'


def validate_override(value):
    if not isinstance(value, dict) or value.get('schema') != 'gpu-greenroom.dispatch-override.v1':
        raise ValueError('invalid dispatch override receipt')
    if not isinstance(value.get('job_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value['job_id']):
        raise ValueError('invalid override job identity')
    try:
        uuid.UUID(value['request_id'])
    except (KeyError, ValueError, TypeError, AttributeError):
        raise ValueError('invalid override request identity') from None
    if not isinstance(value.get('request_digest'), str) or not re.fullmatch(r'[a-f0-9]{64}', value['request_digest']):
        raise ValueError('invalid override request digest')
    if value.get('service_class') not in ('normal', 'quick'):
        raise ValueError('invalid override service class')
    if not isinstance(value.get('requested_by'), str) or not value['requested_by'].strip():
        raise ValueError('override actor must be explicit')
    if value.get('authority') != 'authenticated-console; caller identity unverified':
        raise ValueError('invalid override authority')
    if not isinstance(value.get('at'), (int, float)) or isinstance(value['at'], bool) or not math.isfinite(value['at']):
        raise ValueError('invalid override timestamp')
    return value


def read_request(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


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
    if not isinstance(value,dict) or value.get('schema') not in (SCHEMA, OVERRIDE_SCHEMA,HISTORY_SCHEMA,HINT_SCHEMA,CONTROL_SCHEMA) or not isinstance(value.get('mode'),str) or value['mode'] not in {'fifo','two-class'}:
        raise ValueError('invalid dispatch policy schema or mode')
    for key in ('quick_job_types','quick_routes'):
        rules=value.get(key,{})
        if not isinstance(rules,dict) or any(not isinstance(name,str) or not name.strip()
                                           or not isinstance(basis,str) or not basis.strip() for name,basis in rules.items()):
            raise ValueError('quick eligibility needs exact names and nonblank evidence references')
    overrides, receipts = value.get('job_overrides', {}), value.get('action_receipts', {})
    history = value.get('runtime_history')
    if history is not None:
        if value['schema'] not in (HISTORY_SCHEMA,HINT_SCHEMA,CONTROL_SCHEMA) or not isinstance(history,dict):
            raise ValueError('runtime history requires v3 dispatch policy')
        if not isinstance(history.get('index'),str) or not Path(history['index']).is_absolute() or not isinstance(history.get('profiles'),dict):
            raise ValueError('runtime history requires an absolute index and explicit profiles')
        for name, rule in history['profiles'].items():
            if (not isinstance(name,str) or not name.strip() or not isinstance(rule,dict)
                    or type(rule.get('minimum_samples')) is not int or rule['minimum_samples'] < 1
                    or type(rule.get('max_seconds')) not in (int,float) or not math.isfinite(rule['max_seconds']) or rule['max_seconds'] <= 0
                    or not isinstance(rule.get('profile_digest'),str) or not re.fullmatch(r'[a-f0-9]{64}',rule['profile_digest'])
                    or not isinstance(rule.get('evidence'),str) or not rule['evidence'].strip()):
                raise ValueError('runtime history rule requires samples, threshold and evidence')
    hints = value.get('producer_hints')
    if hints is not None:
        if value['schema'] not in (HINT_SCHEMA,CONTROL_SCHEMA):
            raise ValueError('producer_hints require v4 dispatch policy')
        if (not isinstance(hints,dict) or type(hints.get('max_seconds')) not in (int,float)
                or not math.isfinite(hints['max_seconds']) or hints['max_seconds'] <= 0):
            raise ValueError('producer_hints require a positive finite max_seconds')
    from .job_controls import validate_controls
    validate_controls(value)
    if any(value.get(k) for k in ('job_holds','job_positions','job_actions','job_restores')) and value['schema'] != CONTROL_SCHEMA:
        raise ValueError('job controls require v5 policy')
    if value['schema'] == SCHEMA and (overrides or receipts):
        raise ValueError('dispatch overrides require v2 policy and worker capability')
    if not isinstance(overrides, dict) or not isinstance(receipts, dict):
        raise ValueError('invalid dispatch override maps')
    for identity, receipt in receipts.items():
        if validate_override(receipt)['request_id'] != identity:
            raise ValueError('dispatch action identity mismatch')
    for identity, receipt in overrides.items():
        if validate_override(receipt)['job_id'] != identity or receipts.get(receipt['request_id']) != receipt:
            raise ValueError('dispatch override identity or receipt mismatch')
    return {**value,'quick_job_types':value.get('quick_job_types',{}),'quick_routes':value.get('quick_routes',{})}


def policy(root):
    try:
        return validate(json.loads((Path(root)/'dispatch-policy.json').read_text()))
    except FileNotFoundError:
        return {'schema':SCHEMA,'mode':'fifo','quick_job_types':{},'quick_routes':{}}


def classify(request, config, *, request_digest=None, queue_dir=None):
    if not isinstance(request,dict):
        raise ValueError('job request must be an object')
    declared=request.get('service_class')
    if declared not in (None,'normal','quick'):
        raise ValueError('service_class must be normal or quick')
    hint = runtime_hints.validate(request.get('runtime_hint'))
    override = config.get('job_overrides', {}).get(request.get('job_id'))
    if override and config['mode'] == 'two-class':
        if override['request_digest'] != request_digest:
            raise ValueError('dispatch override request digest mismatch')
        return override['service_class'], 'per-job operator override: ' + override['request_id']
    basis = (config['quick_routes'].get(request.get('route_identity')) if request.get('command_argv') is not None
             else config['quick_job_types'].get(request.get('job_type')))
    eligible=config['mode']=='two-class' and bool(basis)
    profile = request.get('runtime_profile')
    history = config.get('runtime_history')
    if (not eligible and declared != 'normal' and config['mode']=='two-class' and isinstance(profile,dict)
            and history and queue_dir is not None and profile.get('name') in history['profiles']):
        from .runtime_history import estimate, definition_digest
        from .models import JobRequest
        rule = history['profiles'][profile['name']]
        result = (estimate(queue_dir,history['index'],JobRequest.from_json(json.dumps(request)),profile,
                          minimum_samples=rule['minimum_samples'],max_seconds=rule['max_seconds'])
                  if definition_digest(profile) == rule['profile_digest'] else {'eligible':False})
        if result['eligible']:
            eligible = True
            basis = 'runtime-history:' + result['profile_key'] + '; samples=' + str(result['samples'])
    hints = config.get('producer_hints')
    if (not eligible and declared != 'normal' and config['mode'] == 'two-class'
            and hints and hint and hint['expected_seconds'] <= hints['max_seconds']):
        eligible = True
        basis = 'producer-estimated: expected_seconds=' + str(hint['expected_seconds'])
    if declared=='quick' and not eligible:
        raise ValueError('requested quick class is not admitted by the current dispatch policy')
    return ('quick' if eligible and declared!='normal' else 'normal'),basis


def last_class(root):
    try:
        value=json.loads((Path(root)/'dispatch-state.json').read_text())
    except FileNotFoundError:
        return 'normal'
    if not isinstance(value,dict) or value.get('schema')!='gpu-greenroom.dispatch-state.v1' or not isinstance(value.get('last_class'),str) or value['last_class'] not in {'normal','quick'}:
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
    rows.sort(key=lambda row:(row[0],row[1].name))
    from .job_controls import order
    if config['mode']=='fifo':
        return order(root, [path for _,path in rows], config)
    normal,quick=deque(),deque()
    for _,path in rows:
        request, digest = read_request(path/'request.json')
        if not isinstance(request, dict) or request.get('job_id') != path.name:
            raise ValueError('pending request identity is unverified')
        (quick if classify(request,config,request_digest=digest,queue_dir=root)[0]=='quick' else normal).append(path)
    result=[]
    previous=last_class(root)
    while normal or quick:
        bucket=normal if normal and (previous=='quick' or not quick) else quick
        path=bucket.popleft()
        result.append(path)
        previous='normal' if bucket is normal else 'quick'
    return order(root, result, config)
