"""Exact-job cancellation requests and an optional producer cleanup hook."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import threading
import time

from .admission_control import sync_directory, write_durable
from .models import JobState, JobStatus


CAPABILITY = 'job-cancellation.v1'
SCHEMA = 'gpu-greenroom.job-cancellation.v1'


def read_request(job, state):
    try:
        record = json.loads((job/'cancel-request.json').read_text())
    except FileNotFoundError:
        return None
    if (not isinstance(record, dict) or record.get('schema') != SCHEMA
            or record.get('job_id') != state.job_id
            or record.get('expected_started_at') != state.started_at
            or record.get('request_digest') != hashlib.sha256((job/'request.json').read_bytes()).hexdigest()
            or type(record.get('force')) is not bool):
        raise ValueError('cancellation request identity is unverified')
    return record


def request(queue, *, job_id, request_digest, expected_status, expected_started_at,
            request_id, requested_by, force=False):
    for name, value in (('job_id', job_id), ('request_id', request_id)):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            raise ValueError(f'invalid {name}')
    if (not isinstance(requested_by, str) or not requested_by.strip()
            or not isinstance(request_digest, str) or not re.fullmatch(r'[a-f0-9]{64}', request_digest)
            or expected_status not in {'pending', 'deferred', 'running'} or type(force) is not bool):
        raise ValueError('invalid cancellation identity or action')
    if force and expected_status != 'running':
        raise ValueError('force stop requires a running job')
    body = dict(job_id=job_id, request_digest=request_digest, expected_status=expected_status,
                expected_started_at=expected_started_at, request_id=request_id,
                requested_by=requested_by, force=force)
    with queue._coordination_lock():
        directory = queue.queue_dir/'operator-actions'
        directory.mkdir(exist_ok=True)
        receipt_path = directory/f'{request_id}.json'
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous.get('request') != body:
                raise ValueError('request_id already belongs to a different action')
            if previous.get('phase') != 'completed':
                raise RuntimeError(f'incomplete cancellation: inspect {receipt_path}')
            return previous['result']
        job = queue.queue_dir/expected_status/job_id
        try:
            state = JobState.from_json((job/'status.json').read_text())
            digest = hashlib.sha256((job/'request.json').read_bytes()).hexdigest()
        except FileNotFoundError as error:
            raise ValueError('job changed; refresh before cancelling') from error
        if (state.job_id != job_id or state.status.value != expected_status
                or state.started_at != expected_started_at or digest != request_digest):
            raise ValueError('job changed; refresh before cancelling')
        if expected_status == 'running' and CAPABILITY not in state.worker_capabilities:
            raise ValueError('running worker does not support confirmed cancellation; refresh after worker update')
        previous = read_request(job, state)
        if previous and previous['force'] and not force:
            raise ValueError('force stop already requested')
        record = {**body, 'schema': SCHEMA, 'requested_at': time.time(),
                  'receipt_path': str(receipt_path.resolve())}
        result = {'job_id': job_id, 'status': 'cancelling' if expected_status == 'running' else 'cancelled',
                  'force': force, 'receipt_path': str(receipt_path.resolve())}
        receipt = {'schema': SCHEMA, 'phase': 'prepared', 'request': body,
                   'queue_dir': str(queue.queue_dir.resolve()), 'result': result}
        write_durable(receipt_path, receipt)
        write_durable(job/'cancel-request.json', record)
        if expected_status != 'running':
            state.status, state.finished_at = JobStatus.CANCELLED, time.time()
            state.failure_phase = 'operator_cancelled'
            write_durable(job/'status.json', json.loads(state.to_json()))
            write_durable(job/'receipt.json', {'job_id': job_id, 'status': 'cancelled',
                'execution_started': False, 'finished_at': state.finished_at,
                'requested_route': json.loads((job/'request.json').read_text()).get('route_identity'),
                'cancellation': record, 'failure_phase': 'operator_cancelled'})
            job.rename(queue.queue_dir/'cancelled'/job_id)
            sync_directory(queue.queue_dir/expected_status)
            sync_directory(queue.queue_dir/'cancelled')
        write_durable(receipt_path, {**receipt, 'phase': 'completed'})
        return result


def wait(queue, job, state, proc, timeout):
    """Only the worker signals its owned child; acceptance is not quiescence."""
    import subprocess
    started = time.monotonic()
    delivered = None
    while proc.poll() is None:
        record = read_request(job, state)
        desired = signal.SIGKILL if record and record['force'] else signal.SIGTERM
        if record and desired != delivered:
            # Popen has not reaped this live leader, preventing PID reuse here.
            if (queue._process_start_identity(proc.pid) != state.child_start_identity
                    or os.getpgid(proc.pid) != state.child_process_group):
                raise RuntimeError('owned child identity changed before cancellation')
            try:
                os.killpg(state.child_process_group, desired)
            except ProcessLookupError:
                pass
            delivered = desired
            write_durable(job/'cancel-progress.json', {'schema': SCHEMA,
                'status': 'signal-delivered', 'signal': desired, 'at': time.time(),
                'request_id': record['request_id']})
        remaining = None if timeout is None else timeout-(time.monotonic()-started)
        if remaining is not None and remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        try:
            return proc.wait(timeout=.2 if remaining is None else min(.2, remaining))
        except subprocess.TimeoutExpired:
            continue
    return proc.returncode


class Cancelled(SystemExit):
    """A cooperative safe-point exit: normal finally blocks still run."""

    def __init__(self):
        super().__init__(130)


class Cancellation:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self._event = threading.Event()

    @property
    def requested(self):
        # A file-only decision can exit the context before the worker's SIGTERM
        # arrives, restoring the default handler in the middle of cleanup.
        return self._event.is_set()

    def checkpoint(self):
        if self.requested:
            raise Cancelled()

    def wait(self, timeout=None):
        return self._event.wait(timeout)


@contextmanager
def cancellation():
    """Defer managed SIGTERM to checkpoint(); use finally for cleanup/save."""
    stop = Cancellation(os.environ.get('GPU_GREENROOM_CANCEL_REQUEST'))
    previous = None
    if stop.path is not None:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: stop._event.set())
    try:
        yield stop
    finally:
        if stop.path is not None:
            signal.signal(signal.SIGTERM, previous)
