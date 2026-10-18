"""Nonblocking browser navigation with caller-scoped durable action receipts."""

from copy import deepcopy
from pathlib import Path
import threading
from uuid import uuid4

from . import smoke_browser
from .dispatch import atomic_write


_lock = threading.RLock()
_actions = {}


def _key(directory, request_id, digest):
    return (str(Path(directory).resolve()), request_id, digest)


def latest(directory, request_id, digest):
    with _lock:
        value = _actions.get(_key(directory, request_id, digest))
        return deepcopy(value['record']) if value else None


def start(directory, application, request_id, digest, validate_current):
    key = _key(directory, request_id, digest)
    with _lock:
        existing = _actions.get(key)
        if existing and existing['record']['status'] == 'pending':
            return deepcopy(existing['record'])
        identity = str(uuid4())
        root = Path(directory).resolve().parent / 'smoke-navigation'
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        state = {'record': {'schema': 'gpu-greenroom.browser-focus-action.v1',
            'action_id': identity, 'request_id': request_id, 'request_digest': application['request_digest'],
            'destination_digest': digest, 'job_id': application['job_id'], 'status': 'pending', 'phase': 'inspect'},
            'path': root / (identity + '.json'), 'finished': threading.Event()}
        atomic_write(state['path'], state['record'])
        _actions[key] = state
        result = deepcopy(state['record'])
        threading.Thread(target=_run, args=(state, deepcopy(application), validate_current), daemon=True).start()
        return result


def _run(state, application, validate_current):
    try:
        validate_current()
        smoke_browser.exchange(application, state['record']['request_id'], 'inspect')
        validate_current()
        _update(state, phase='reveal')
        receipt = smoke_browser.exchange(application, state['record']['request_id'], 'reveal')
        validate_current()
        _update(state, status='complete', phase='complete', receipt=receipt)
    except Exception as error:
        _update(state, status='failed', error=str(error))
    finally:
        state['finished'].set()


def _update(state, **values):
    with _lock:
        state['record'].update(values)
        try:
            atomic_write(state['path'], state['record'])
        except OSError as error:
            state['record'].update(status='failed', error='Navigation receipt write failed: ' + str(error))
