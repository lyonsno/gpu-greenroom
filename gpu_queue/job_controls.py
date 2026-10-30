"""Operator job holds and recoverable, unstarted performance attempts.

The dispatch policy is the atomic authority for holds and ordering. Staged retry
jobs are published only after that authority and its idempotence receipt exist.
"""

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import replace

from . import admission, dispatch, performance_window
from .models import JobRequest, JobState, JobStatus

CAPABILITY = 'operator-job-controls.v1'
SCHEMA = 'gpu-greenroom.dispatch-policy.v5'


def validate_controls(value):
    for group in ('job_holds', 'job_positions', 'job_actions', 'job_restores'):
        if not isinstance(value.get(group, {}), dict):
            raise ValueError('invalid job controls map')
    for job_id, hold in value.get('job_holds', {}).items():
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', job_id) or not isinstance(hold, dict)
                or type(hold.get('allow_bypass')) is not bool
                or not re.fullmatch(r'[a-f0-9]{64}', hold.get('request_digest', ''))):
            raise ValueError('invalid job hold')
    for job_id, position in value.get('job_positions', {}).items():
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', job_id) or not isinstance(position, dict)
                or not isinstance(position.get('predecessors'), list)
                or any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', x)
                       for x in position['predecessors'])
                or not re.fullmatch(r'[a-f0-9]{64}', position.get('request_digest', ''))):
            raise ValueError('invalid job placement')


def order(root, paths, config):
    held = set()
    blocked = False
    for job_id, hold in config.get('job_holds', {}).items():
        for state in ('pending', 'running'):
            path = root/state/job_id
            if path.exists():
                if dispatch.read_request(path/'request.json')[1] != hold['request_digest']:
                    raise ValueError('held request digest changed')
                held.add(job_id)
                blocked = blocked or not hold['allow_bypass']
    result = [p for p in paths if p.name not in held]
    for job_id, placement in config.get('job_positions', {}).items():
        selected = next((p for p in result if p.name == job_id), None)
        if selected is None:
            continue
        if dispatch.read_request(selected/'request.json')[1] != placement['request_digest']:
            raise ValueError('placed request digest changed')
        result.remove(selected)
        index = max((i+1 for i,p in enumerate(result) if p.name in placement['predecessors']), default=0)
        result.insert(index, selected)
    return [] if blocked else result


def eligible_retry(path, request, state):
    if not request.performance_window or state.status not in {JobStatus.CANCELLED, JobStatus.FAILED}:
        return False
    if state.started_at is None:
        return state.status == JobStatus.CANCELLED
    report = performance_window.read_receipt(path/'performance-window.json', request)
    if (report.get('status') != 'cancelled' or report.get('workload_started') is not False
            or report.get('phase') != 'operator-confirmation' or 'workload_pid' in report
            or 'workload_started_at' in report or 'confirmed_at' in report):
        return False
    receipt = json.loads((path/'receipt.json').read_text())
    return (receipt.get('job_id') == request.job_id and receipt.get('status') in {'failed', 'cancelled'}
            and receipt.get('performance_window') == report and state.finished_at is not None
            and not str(state.failure_phase).endswith('quiescence_unresolved'))


def projection(root, path, request, state, config):
    from .pause_estimates import project
    hold = config.get('job_holds', {}).get(path.name)
    predecessor_running = bool(hold and hold.get('parent_job_id')
                               and (root/'running'/hold['parent_job_id']).exists())
    result = {'supported': request.cooperative_checkpoint is True,
              'phase': 'pause-requested' if hold and (state.status == JobStatus.RUNNING or predecessor_running) else
                       'paused' if hold and state.status == JobStatus.PENDING else None,
              'allow_bypass': hold['allow_bypass'] if hold else True,
              'estimate': project(path, request, state)}
    try:
        retry = eligible_retry(path, request, state)
        error = None
    except (ValueError, KeyError, OSError, TypeError) as exc:
        retry, error = False, str(exc)
    restored = config.get('job_restores', {}).get(path.name)
    feedback_id = hashlib.sha256((path.name+dispatch.read_request(path/'request.json')[1]+'ask-pause').encode()).hexdigest()
    feedback_path = root/'job-feedback'/f'{feedback_id}.json'
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else None
    return {'pause_control': result, 'requeue_available': retry and not restored,
            'requeue_error': error, 'restored_as': restored.get('new_job_id') if restored else None,
            'pause_support_request': feedback.get('status') if feedback else None}


def _publish_retry(queue, result):
    staging = queue.queue_dir/'continuation-staging'/result['new_job_id']
    if staging.exists():
        if dispatch.read_request(staging/'request.json')[1] != result['new_request_digest']:
            raise ValueError('staged retry changed')
        staging.rename(queue.queue_dir/'pending'/staging.name)
        from .admission_control import sync_directory
        sync_directory(staging.parent)
        sync_directory(queue.queue_dir/'pending')


def recover(queue):
    with queue._coordination_lock():
        for result in dispatch.policy(queue.queue_dir).get('job_restores', {}).values():
            _publish_retry(queue, result)


def inherit_hold(queue, parent_id, successor, digest):
    """Caller holds coordination lock; commit before exposing successor pending."""
    config = dispatch.policy(queue.queue_dir)
    hold = config.get('job_holds', {}).get(parent_id)
    if hold:
        holds = dict(config['job_holds'])
        holds[successor] = {**hold, 'request_digest':digest, 'parent_job_id':parent_id}
        dispatch.atomic_write(queue.queue_dir/'dispatch-policy.json', {**config, 'job_holds':holds})


def perform(queue, *, job_id, request_digest, expected_status, expected_started_at,
            request_id, requested_by, action, allow_bypass=True, placement='end'):
    for value in (job_id, request_id):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError('invalid job/action identity')
    if (action not in {'pause','resume','requeue','ask-pause'} or type(allow_bypass) is not bool
            or placement not in {'previous','second','end'} or expected_status not in {s.value for s in JobStatus}
            or not isinstance(requested_by,str) or not requested_by.strip()):
        raise ValueError('invalid job action')
    body = dict(job_id=job_id, request_digest=request_digest, expected_status=expected_status,
                expected_started_at=expected_started_at, request_id=request_id, requested_by=requested_by,
                action=action, allow_bypass=allow_bypass, placement=placement)
    with queue._coordination_lock():
        config = dispatch.policy(queue.queue_dir)
        previous = config.get('job_actions', {}).get(request_id)
        if previous:
            if previous['request'] != body:
                raise ValueError('action identity reused')
            if action == 'requeue':
                _publish_retry(queue, previous['result'])
            return previous['result']
        path = queue.queue_dir/expected_status/job_id
        try:
            request_raw, digest = dispatch.read_request(path/'request.json')
            request = JobRequest.from_json(json.dumps(request_raw))
            state = JobState.from_json((path/'status.json').read_text())
        except FileNotFoundError as error:
            raise ValueError('job changed; refresh before acting') from error
        if (digest != request_digest or state.status.value != expected_status
                or state.started_at != expected_started_at or state.job_id != job_id or request.job_id != job_id):
            raise ValueError('job changed; refresh before acting')
        if action == 'ask-pause':
            if not request.agent_id or not re.fullmatch(r'[A-Za-z0-9_-]+',request.agent_id):
                raise ValueError('submitting agent is not recorded')
            event_id = hashlib.sha256((job_id+digest+'ask-pause').encode()).hexdigest()
            event_path = queue.queue_dir/'job-feedback'/f'{event_id}.json'
            if not event_path.exists():
                dispatch.atomic_write(event_path, {'schema':'gpu-greenroom.job-feedback.v1',
                    'event_id':event_id, 'job_id':job_id, 'request_digest':digest,
                    'target_diaulos':request.agent_id, 'status':'saved', 'created_at':time.time(),
                    'text':(os.environ.get('GPU_GREENROOM_PAUSE_SUPPORT_MESSAGE','Please implement pause support.') + '\nPlease add cooperative checkpoint/yield '
                            'support for this job route so I can pause it from Greenroom. '
                            'No need to reply just to acknowledge.\n'
                            f'Job: {job_id}\nRoute: {request.route_identity or request.job_type}\n'
                            f'Request: {path / "request.json"}')})
            record = json.loads(event_path.read_text())
            if record['status'] == 'failed':
                record = {**record, 'status':'saved', 'retried_at':time.time(),
                          'attempts':[*record.get('attempts',[]),
                                      {key:value for key,value in record.items() if key != 'attempts'}]}
                dispatch.atomic_write(event_path,record)
            return {'status':record['status'], 'event_id':event_id}
        updated = {**config, 'schema':SCHEMA}
        holds = dict(config.get('job_holds', {}))
        if action in {'pause','resume'}:
            if state.status not in {JobStatus.RUNNING, JobStatus.PENDING}:
                raise ValueError('job is no longer running or queued; refresh')
            if not request.cooperative_checkpoint:
                raise ValueError('pause unsupported by this job')
            if state.status == JobStatus.RUNNING and CAPABILITY not in state.worker_capabilities:
                raise ValueError('pause requires an updated worker on this running attempt')
            if (path/'cancel-request.json').exists():
                raise ValueError('job is being cancelled')
            if action == 'pause':
                holds[job_id] = {'request_digest':digest, 'allow_bypass':allow_bypass, 'requested_at':time.time()}
            else:
                holds.pop(job_id,None)
            for successor, inherited in list(holds.items()):
                if inherited.get('parent_job_id') == job_id:
                    if action == 'resume':
                        holds.pop(successor)
                    else:
                        holds[successor] = {**inherited, 'allow_bypass':allow_bypass}
            updated['job_holds'] = holds
            result = {'job_id':job_id, 'status': 'pause-requested' if action=='pause' and state.status==JobStatus.RUNNING
                      else 'paused' if action=='pause' else 'resumed'}
        else:
            prior = config.get('job_restores', {}).get(job_id)
            if prior:
                _publish_retry(queue, prior)
                return prior
            if not eligible_retry(path, request, state):
                raise ValueError('only cancelled performance jobs with no workload execution can be restored')
            if state.child_process_group and queue._process_group_alive(state.child_process_group):
                raise ValueError('previous process group is still live or unknown')
            if request.params.get('continuation_of'):
                raise ValueError('checkpoint continuation needs its producer restart contract')
            validation = None
            if (path/'validation.json').exists() or admission.CAPABILITY in request.required_worker_capabilities:
                validation = json.loads((path/'validation.json').read_text())
                admission.assert_current(request, validation)
                if validation.get('preflight'):
                    raise ValueError('application-specific preflight must be rerun by the producer before retry')
            for status in ('pending','running','deferred','done'):
                for other in (queue.queue_dir/status).iterdir():
                    if not (other/'request.json').exists():
                        continue
                    candidate = JobRequest.from_json((other/'request.json').read_text())
                    if (candidate.submitted_at >= request.submitted_at and candidate.command_argv == request.command_argv
                            and candidate.command_cwd == request.command_cwd and candidate.command_env == request.command_env
                            and candidate.route_identity == request.route_identity and candidate.agent_id == request.agent_id
                            and candidate.job_type == request.job_type and candidate.input_path == request.input_path
                            and {k:v for k,v in candidate.params.items() if k != admission.DIGEST_PARAM}
                                == {k:v for k,v in request.params.items() if k != admission.DIGEST_PARAM}
                            and candidate.performance_window == request.performance_window):
                        raise ValueError(f'equivalent replacement already exists: {candidate.job_id}; inspect before retrying')
            # Placement is among waiting jobs, not only currently dispatchable jobs.
            pending = dispatch.pending_order(queue.queue_dir,{**config,'job_holds':{}})
            predecessors = ([p.name for p in pending if JobState.from_json((p/'status.json').read_text()).submitted_at < state.submitted_at]
                            if placement=='previous' else [p.name for p in pending[:1]] if placement=='second'
                            else [p.name for p in pending])
            fresh = replace(request, job_id=uuid.uuid4().hex[:12], submitted_at=time.time(),
                            output_dir='', params=dict(request.params),
                            required_worker_capabilities=list(request.required_worker_capabilities))
            fresh.output_dir = str(queue.queue_dir/'outputs'/fresh.job_id)
            if validation is not None:
                validation = {**validation, 'request_digest':admission.request_identity(fresh),
                              'retry_of':job_id, 'validated_at':time.time()}
                fresh.params[admission.DIGEST_PARAM] = admission.digest(validation)
            stage = queue.submit(fresh, _staged=True, validation=validation)
            fresh_digest = dispatch.read_request(stage/'request.json')[1]
            result = {'job_id':job_id, 'new_job_id':fresh.job_id, 'new_request_digest':fresh_digest,
                      'status':'requeued', 'placement':placement, 'at':time.time()}
            dispatch.atomic_write(stage/'retry-of.json', {**result,'original_request_digest':digest})
            updated['job_positions'] = {**config.get('job_positions', {}), fresh.job_id:
                                       {'predecessors':predecessors, 'request_digest':fresh_digest}}
            updated['job_restores'] = {**config.get('job_restores', {}),job_id:result}
        updated['job_actions'] = {**config.get('job_actions', {}), request_id:{'request':body,'result':result}}
        dispatch.atomic_write(queue.queue_dir/'dispatch-policy.json', dispatch.validate(updated))
        if action == 'requeue':
            _publish_retry(queue,result)
        return result


def feedback_event(path):
    value = json.loads(path.read_text())
    if (value.get('schema') != 'gpu-greenroom.job-feedback.v1'
            or path.stem != value.get('event_id') or not isinstance(value.get('text'),str)):
        raise ValueError('invalid saved job feedback')
    return {'schema':'gpu-greenroom.job-feedback-event.v1', 'event_id':value['event_id'],
            'request_id':value['job_id'], 'request_digest':value['request_digest'],
            'kind':'pause-support', 'source':{'agent_id':value['target_diaulos']},
            'record_path':str(path.resolve()), 'payload':{'text':value['text']}}


def deliver_feedback(root, adapter):
    import subprocess
    import os
    import fcntl
    directory = root/'job-feedback'
    if not directory.exists():
        return
    with (directory/'delivery.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in directory.glob('*.json'):
            record = json.loads(path.read_text())
            if record['status'] in {'consumed','submitted','queued','failed'}:
                continue
            event = feedback_event(path)
            operation = 'deliver' if record['status']=='saved' else 'status'
            dispatch.atomic_write(path,{**record,'status':'sending'})
            try:
                if not os.path.isabs(adapter) or not os.access(adapter,os.X_OK):
                    raise ValueError('job feedback adapter is not an absolute executable')
                completed = subprocess.run([adapter],input=json.dumps({'operation':operation,'event':event}),
                    text=True,capture_output=True,shell=False)
                if completed.returncode:
                    raise ValueError(completed.stderr)
                result = json.loads(completed.stdout)
                if result.get('event_id')!=event['event_id'] or result.get('status') not in {'queued','submitted','consumed','failed','uncertain'}:
                    raise ValueError('adapter did not return exact feedback identity')
                dispatch.atomic_write(path,{**record,'status':result['status'],'delivery':result})
            except (ValueError,OSError) as error:
                dispatch.atomic_write(path,{**record,'status':'uncertain','error':str(error)})


def start_feedback(root, adapter):
    import threading
    stop = threading.Event()
    def work():
        while not stop.is_set():
            try:
                deliver_feedback(root,adapter)
            except Exception as error:
                dispatch.atomic_write(root/'job-feedback-health.json',{'error':str(error),'at':time.time()})
            stop.wait(5)
    threading.Thread(target=work,daemon=True,name='job-feedback').start()
    return stop
