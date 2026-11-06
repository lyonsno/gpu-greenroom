"""Persistent operator attention and advisory runtime escalation, never a mutex."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import threading
import time
from urllib.parse import urlparse

from . import dispatch, performance_window

SCHEMA = 'gpu-greenroom.operator-attention.v1'
FRESH_SECONDS = 15
OWNER_GRACE_SECONDS = 300


def stamp(value):
    return datetime.fromtimestamp(value,timezone.utc).isoformat().replace('+00:00','Z')


def number(value):
    return type(value) in (int,float) and math.isfinite(value) and value >= 0


def validate(value):
    if not isinstance(value,dict) or value.get('schema') != SCHEMA:
        raise ValueError('attention snapshot must be a supported object')
    def date(field):
        raw=value.get(field)
        if not isinstance(raw,str):
            raise ValueError('attention observation/deadline missing')
        parsed=datetime.fromisoformat(raw.replace('Z','+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('attention timestamps require an explicit timezone')
        return parsed.timestamp()
    if date('valid_until') < date('observed_at'):
        raise ValueError('attention deadline predates observation')
    def url(raw):
        if not isinstance(raw,str):
            raise ValueError('attention URL missing')
        parsed=urlparse(raw)
        if (parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password
                or any(c.isspace() for c in raw)):
            raise ValueError('attention URL must be an absolute HTTP(S) address')
        parsed.port  # Validate explicit port syntax as well.
    url(value.get('monitor_url'))
    if not isinstance(value.get('queue_dir'),str) or not Path(value['queue_dir']).is_absolute():
        raise ValueError('attention queue identity missing')
    queue=value.get('queue')
    if not isinstance(queue,dict) or any(type(queue.get(k)) is not int or queue[k]<0 for k in ('running','waiting')):
        raise ValueError('attention queue counts missing or invalid')
    if not isinstance(value.get('items'),list):
        raise ValueError('attention items missing')
    ids=set()
    for item in value['items']:
        if not isinstance(item,dict) or any(not isinstance(item.get(k),str) or not item[k] for k in ('id','title','status')):
            raise ValueError('invalid attention item')
        if item['id'] in ids:
            raise ValueError('duplicate attention identity')
        ids.add(item['id'])
        url(item.get('target_url'))
    if not isinstance(value.get('errors',[]),list) or any(not isinstance(e,str) for e in value.get('errors',[])):
        raise ValueError('invalid attention diagnostics')
    return value


def attempt_key(row):
    return hashlib.sha256(json.dumps([row['job_id'],row['request_digest'],row['started_at']],
                                     separators=(',',':')).encode()).hexdigest()


def load(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def progress(root, *, job_id, request_digest, started_at, note, remaining_seconds):
    """Source-linked owner report, not proof of progress or quiescence."""
    from .queue import GPUQueue
    if (not isinstance(job_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]+',job_id)
            or not isinstance(note,str) or not note.strip() or not number(remaining_seconds)):
        raise ValueError('progress requires job identity, a note and finite remaining seconds')
    queue = GPUQueue(root)
    with queue._coordination_lock():
        path = queue.queue_dir/'running'/job_id
        request,digest = dispatch.read_request(path/'request.json')
        state = load(path/'status.json')
        if (digest != request_digest or state.get('status') != 'running'
                or state.get('started_at') != started_at or state.get('job_id') != job_id):
            raise ValueError('progress attempt changed; refresh')
        value = {'schema':'gpu-greenroom.runtime-progress.v1','job_id':job_id,
                 'request_digest':digest,'started_at':started_at,'owner':request.get('agent_id'),
                 'reported_at':time.time(),'remaining_seconds':remaining_seconds,'note':note,
                 'authority':'owner-reported; not verified progress'}
        dispatch.atomic_write(path/'runtime-progress.json',value)
        return value


def snapshot(root, state, monitor_url, *, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    items = []
    errors = []
    directory = root/'operator-attention'
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for row in state['jobs']:
            diagnostics=[str(row[k]) for k in ('record_error','performance_window_error','runtime_hint_error') if row.get(k)]
            if row.get('containment_status')=='running' and not row.get('consistent'):
                diagnostics.append('running containment has inconsistent state')
            if (row.get('status')=='running' and row.get('performance_window_reason')
                    and not row.get('performance_window') and not row.get('performance_window_error')):
                diagnostics.append('required performance receipt missing')
            if diagnostics:
                errors.extend(f'{row["job_id"]}: {message}' for message in diagnostics)
                items.append({'id':'unverified-'+row['job_id'],'job_id':row['job_id'],
                              'kind':'unverified','status':'unverified','title':'Job state unverified',
                              'summary':'; '.join(diagnostics),'target_url':monitor_url+'#job-'+row['job_id']})
                continue
            if row.get('status') != 'running' or not row.get('consistent') or not row.get('request_digest'):
                continue
            job_id = row['job_id']
            perf = row.get('performance_window')
            base = {'job_id':job_id,'agent_id':row.get('agent_id'),
                    'target_url':monitor_url+'#job-'+job_id}
            if perf and perf.get('status') == 'waiting':
                # Older runners cannot consume a web decision. Still surface their wait honestly.
                items.append({**base,'id':'performance-'+job_id,'title':'Confirm performance window',
                              'summary':row.get('performance_window_reason'), 'kind':'performance-confirmation',
                              'status':'waiting','attempt_id':perf.get('attempt_id'),
                              'web_action':perf.get('decision_protocol') == 'shared.v1'})
                continue
            if row.get('performance_window_reason') and (not perf or perf.get('status') != 'running'):
                continue
            start = perf.get('workload_started_at') if perf else row.get('started_at')
            if not number(start) or now < start:
                continue
            expected = (row.get('runtime_hint') or {}).get('expected_seconds')
            if not number(expected) or expected <= 0:
                continue
            key = attempt_key(row)
            path = directory/(key+'.json')
            record = load(path,{})
            try:
                report = load(root/'running'/job_id/'runtime-progress.json',{})
                if not isinstance(report,dict):
                    raise ValueError('progress report is not an object')
            except (ValueError,OSError) as error:
                report = {}
                errors.append(f'{job_id}: progress unverified: {error}')
            valid_report = (report.get('schema') == 'gpu-greenroom.runtime-progress.v1'
                and report.get('request_digest') == row['request_digest']
                and report.get('job_id') == job_id and report.get('started_at') == row.get('started_at')
                and number(report.get('reported_at')) and start <= report['reported_at'] <= now
                and number(report.get('remaining_seconds')) and isinstance(report.get('note'),str))
            deadline = start + max(expected*3,expected+120)
            if valid_report:
                deadline = max(deadline, report['reported_at']+report['remaining_seconds']+120)
            if now <= deadline:
                continue
            if not record:
                event_id = hashlib.sha256(('runtime-overrun'+key).encode()).hexdigest()
                record = {'schema':'gpu-greenroom.runtime-attention.v1',**base,'id':key,
                          'first_detected_at':now,'expected_seconds':expected,'started_at':row['started_at'],
                          'request_digest':row['request_digest'],'event_id':event_id}
                dispatch.atomic_write(path,record)
            event_path = root/'job-feedback'/(record['event_id']+'.json')
            if not event_path.exists():
                dispatch.atomic_write(event_path,{'schema':'gpu-greenroom.job-feedback.v1',
                    'event_id':record['event_id'],'job_id':job_id,'request_digest':row['request_digest'],
                    'target_diaulos':row.get('agent_id'),'kind':'runtime-overrun','status':'saved','created_at':now,
                    'text':(f'Greenroom runtime check: {job_id} has exceeded its {expected:g}s producer estimate. '
                            f'Execution has run {now-start:.0f}s, excluding performance confirmation wait. '
                            'Please inspect progress or failure, then report a revised remaining time with '
                            f'gpu-greenroom --queue-dir {shlex.quote(str(root))} attention-report --job-id {job_id} '
                            f'--request-digest {row["request_digest"]} --started-at {row["started_at"]} '
                            '--remaining-seconds SECONDS --note "what is happening". '
                            'This is advisory, not a stop request. No acknowledgement-only reply is needed.')})
            event = load(event_path,{})
            if now - record['first_detected_at'] < OWNER_GRACE_SECONDS:
                continue
            items.append({**base,'id':key,'title':'Job exceeded its estimate','kind':'runtime-overrun',
                          'status':'watching' if record.get('acknowledged_at') else 'waiting',
                          'summary':f'{now-start:.0f}s execution; {expected:g}s estimate. Progress unverified.',
                          'owner_delivery':event.get('status','unavailable'),
                          'owner_report':report if valid_report else None})
    # Smoke request records remain their own action authority; links lead to those existing controls.
    from .smoke_requests import SmokeRequests
    smoke_snapshot = SmokeRequests(root/'smoke-requests').snapshot()
    errors.extend(smoke_snapshot.get('errors',[]))
    for item in smoke_snapshot['items']:
        phase = item.get('display',{}).get('phase')
        if phase in {'awaiting-start','operator-needed','awaiting-response'} and not item.get('response'):
            request = item['request']
            items.append({'id':'smoke-'+request['id'],'title':request.get('title') or 'Operator smoke',
                          'kind':'smoke','status':'waiting','target_url':monitor_url+'#smoke-'+request['id']})
    return {'schema':SCHEMA,'observed_at':stamp(now),'valid_until':stamp(now if errors else now+FRESH_SECONDS),
            'monitor_url':monitor_url,'queue_dir':str(root.resolve()),
            'queue':{'running':sum((j.get('containment_status') or j.get('status'))=='running' for j in state['jobs']),
                     'waiting':sum((j.get('containment_status') or j.get('status')) in {'pending','deferred'} for j in state['jobs']),
                     'paused':state.get('paused')},'items':items,'errors':errors}


def acknowledge(root, item_id):
    if not isinstance(item_id,str) or not re.fullmatch(r'[a-f0-9]{64}',item_id):
        raise ValueError('invalid runtime attention identity')
    path = Path(root)/'operator-attention'/(item_id+'.json')
    with (path.parent/'update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        record = load(path)
        if not record or record.get('id') != item_id:
            raise ValueError('runtime attention no longer exists')
        record.setdefault('acknowledged_at',time.time())
        dispatch.atomic_write(path,record)
        return record


def start(root, monitor_url, *, notices=True):
    """One inexpensive service loop, also running while every browser is closed."""
    from .operator_server import queue_snapshot
    root = Path(root)
    stop = threading.Event()
    popups = set()
    def popup(item):
        path = root/'operator-attention'/(item['id']+'.json')
        def settled():
            record = load(path,{})
            active = load(root/'operator-attention.json',{}).get('items',[])
            return bool(record.get('acknowledged_at') or not any(i['id']==item['id'] for i in active) or stop.is_set())
        try:
            result = performance_window._native_prompt({
                'job_id':item['job_id'],'owner':item.get('agent_id'),'route':'Runtime attention',
                'reason':item['summary'],'attention_url':item['target_url']},check_answer=settled)
            if result is True and not settled():
                import subprocess
                subprocess.run(['open',item['target_url']],check=True)
        except Exception as error:
            dispatch.atomic_write(path.with_suffix('.notice.json'),{'error':str(error),'at':time.time()})
    def work():
        while not stop.is_set():
            try:
                value = snapshot(root,queue_snapshot(root,'active',read_only=True),monitor_url)
                dispatch.atomic_write(root/'operator-attention.json',validate(value))
                for item in value['items']:
                    if notices and item['kind']=='runtime-overrun' and item['status']=='waiting' and item['id'] not in popups:
                        record_path = root/'operator-attention'/(item['id']+'.json')
                        with (record_path.parent/'update.lock').open('a') as lock:
                            fcntl.flock(lock,fcntl.LOCK_EX)
                            record = load(record_path)
                            popups.add(item['id'])
                            if record.get('notice_attempted_at') or record.get('acknowledged_at'):
                                continue
                            # A crash after this receipt is unknown delivery, never an automatic resend.
                            dispatch.atomic_write(record_path,{**record,'notice_attempted_at':time.time()})
                        threading.Thread(target=popup,args=(item,),daemon=True,name='runtime-attention-popup').start()
            except Exception as error:
                dispatch.atomic_write(root/'operator-attention-health.json',{'error':str(error),'at':time.time()})
            stop.wait(5)
    threading.Thread(target=work,daemon=True,name='operator-attention').start()
    return stop
