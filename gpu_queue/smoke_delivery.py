"""Durable, dispatch-independent returns for interactive smokes."""

from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading

from .dispatch import atomic_write
from .smoke_requests import _digest, _now


SCHEMA = 'gpu-greenroom.smoke-event.v1'


class SmokeDelivery:
    def __init__(self, store, adapter=None):
        self.store = store
        self.directory = store.directory.parent / 'smoke-delivery'
        self.adapter = adapter

    @contextmanager
    def locked(self, name):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / (name + '.lock')).open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def activate(self):
        with self.locked('activation'):
            path = self.directory / 'activation.json'
            if not path.exists():
                atomic_write(path, {'schema': 'gpu-greenroom.smoke-delivery-activation.v1',
                                    'activated_at': _now()})

    def events(self, record):
        request = record['request']
        if record.get('presentation', {}).get('purpose', request.get('purpose')) == 'diagnostic':
            return []
        values = []
        if record.get('response'):
            values.append(('response', record['response'], record['response']['responded_at']))
        if record.get('repeat_request'):
            values.append(('repeat', record['repeat_request'], record['repeat_request']['requested_at']))
        display = self.store.display(record)
        if display.get('phase') in {'operator-needed', 'interactive'} and display.get('job_state') == 'running':
            target = self.store.destination(record)
            session = record.get('session') or record.get('application_session')
            if session and (display.get('terminal_verified') or display.get('application_navigation_available')):
                values.append(('usable', {'destination': target, 'job_id': display['job_id']}, session['observed_at']))
        result = []
        for kind, payload, occurred in values:
            identity = {'kind': kind, 'request_id': request['id'],
                        'request_digest': record['request_digest'], 'payload': payload}
            result.append({**identity, 'schema': SCHEMA, 'event_id': _digest(identity),
                           'source': request['source'], 'title': request['title'],
                           'record_path': str(self.store.path(request['id']).resolve()),
                           'occurred_at': occurred})
        return result

    def receipt(self, event):
        path = self.directory / (event['event_id'] + '.json')
        try:
            value = json.loads(path.read_text())
        except FileNotFoundError:
            return {'event_id': event['event_id'], 'status': 'saved' if event['kind'] != 'usable' else 'not-notified'}
        if not isinstance(value, dict) or not isinstance(value.get('event'), dict) or value.get('event_id') != event['event_id']:
            raise ValueError('smoke delivery receipt identity is missing or malformed')
        if {key: value['event'].get(key) for key in event if key != 'occurred_at'} != {key: item for key, item in event.items() if key != 'occurred_at'}:
            raise ValueError('smoke delivery receipt identity conflicts')
        allowed = {'sending', 'failed', 'uncertain'} | (
            {'not-notified', 'notified', 'not-notified-before-activation'} if event['kind'] == 'usable'
            else {'saved', 'queued', 'submitted', 'consumed', 'saved-before-activation'})
        if value.get('status') not in allowed:
            raise ValueError('smoke delivery receipt status is missing or unrecognized')
        return {key: value[key] for key in ('event_id', 'status', 'updated_at', 'error', 'receipt', 'explicit_retry') if key in value}

    def summary(self, record):
        result = {}
        for event in self.events(record):
            try:
                result[event['kind']] = self.receipt(event)
            except (OSError, ValueError, KeyError, TypeError) as error:
                result[event['kind']] = {'event_id': event['event_id'], 'status': 'unverified', 'error': str(error)}
        return result

    def _invoke(self, event, operation):
        if callable(self.adapter):
            return self.adapter(event, operation)
        executable = Path(self.adapter)
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            return {'event_id': event['event_id'], 'status': 'failed' if operation == 'deliver' else 'uncertain',
                    'error': 'smoke delivery adapter must be a configured absolute executable'}
        try:
            result = subprocess.run([str(executable)], input=json.dumps({'operation': operation, 'event': event}),
                                    text=True, capture_output=True, shell=False)
        except OSError as error:
            return {'event_id': event['event_id'], 'status': 'failed' if operation == 'deliver' else 'uncertain',
                    'error': str(error)}
        if result.returncode:
            raise ValueError(f'adapter exited {result.returncode}: {result.stderr}')
        return json.loads(result.stdout)

    def run_once(self):
        if not self.adapter:
            return
        self.activate()
        activated = datetime.fromisoformat(json.loads((self.directory / 'activation.json').read_text())['activated_at'])
        records, _ = self.store.scan()
        for record in records:
            for event in self.events(record):
                with self.locked(event['event_id']):
                    path = self.directory / (event['event_id'] + '.json')
                    try:
                        prior = self.receipt(event)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        atomic_write(self.directory / (event['event_id'] + '.error.json'),
                            {'event_id': event['event_id'], 'status': 'unverified', 'error': str(error),
                             'receipt_path': str(path), 'observed_at': _now()})
                        continue
                    status = prior['status']
                    if status in {'consumed', 'notified', 'failed', 'saved-before-activation', 'not-notified-before-activation'}:
                        continue
                    if status in {'saved', 'not-notified'} and datetime.fromisoformat(event['occurred_at']) < activated and not prior.get('explicit_retry'):
                        atomic_write(path, {**prior, 'event': event,
                            'status': 'not-notified-before-activation' if event['kind'] == 'usable' else 'saved-before-activation',
                            'updated_at': _now()})
                        continue
                    operation = 'deliver' if status in {'saved', 'not-notified'} else 'status'
                    # A lost acknowledgement is recovered by observation, not another send.
                    atomic_write(path, {**prior, 'event': event, 'status': 'sending', 'updated_at': _now()})
                    try:
                        answer = self._invoke(event, operation)
                        allowed = {'notified', 'failed', 'uncertain'} if event['kind'] == 'usable' else {'queued', 'submitted', 'consumed', 'failed', 'uncertain'}
                        if not isinstance(answer, dict) or answer.get('event_id') != event['event_id'] or answer.get('status') not in allowed:
                            raise ValueError('adapter did not return an exact event receipt')
                        atomic_write(path, {**answer, 'event': event, 'updated_at': _now()})
                    except Exception as error:
                        atomic_write(path, {'event_id': event['event_id'], 'event': event,
                                            'status': 'uncertain', 'error': str(error), 'updated_at': _now()})

    def retry(self, record, kind, event_id):
        event = next((value for value in self.events(record) if value['kind'] == kind and value['event_id'] == event_id), None)
        if not event:
            raise ValueError('delivery event changed; refresh before retrying')
        with self.locked(event_id):
            prior = self.receipt(event)
            if prior['status'] not in {'failed', 'saved-before-activation', 'not-notified-before-activation'}:
                raise ValueError('only a definite unsent event can be retried')
            atomic_write(self.directory / (event_id + '.json'), {'event_id': event_id, 'event': event,
                'status': 'saved' if kind != 'usable' else 'not-notified', 'updated_at': _now(), 'explicit_retry': True})

    def start(self):
        self.activate()
        stop = threading.Event()
        def work():
            while not stop.is_set():
                try:
                    self.run_once()
                except Exception as error:
                    atomic_write(self.directory / 'health.json', {'status': 'failed', 'error': str(error), 'observed_at': _now()})
                stop.wait(5)
        thread = threading.Thread(target=work, name='smoke-delivery', daemon=True)
        thread.start()
        return stop
