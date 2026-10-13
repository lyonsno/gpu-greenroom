import json
import os
import socket
import threading
import tempfile
from copy import deepcopy
from uuid import uuid4

import pytest

from gpu_queue.smoke_requests import SmokeRequests
from tests.test_smoke_progress import job, request


def browser_store(root):
    store = SmokeRequests(root / 'smoke-requests')
    value = request(root, 'test-job')
    value['availability'] = 'prepared'
    record, _ = store.submit(value)
    path = job(root, 'running')
    return store, value, record, path


def test_finished_browser_is_feedback_not_a_live_application(tmp_path):
    store, value, _, path = browser_store(tmp_path)
    record = store.publish_application(value['id'], {
        'url': 'http://127.0.0.1:61946/log', 'phase': 'operator-needed', 'label': 'Cold log accepting input',
    })
    (tmp_path / 'done').mkdir()
    path = path.rename(tmp_path / 'done' / path.name)
    (path / 'status.json').write_text(json.dumps({'job_id': 'test-job', 'status': 'done', 'exit_code': 0}))
    display = store.display(record)
    assert display['phase'] == 'awaiting-response'
    assert 'ended' in display['label']
    assert not display.get('application_reported')
    assert not display.get('application_verified')


class Owner:
    def __init__(self, path, identity, binding):
        self.directory = tempfile.TemporaryDirectory(prefix='gr-', dir='/private/tmp')
        path = os.path.join(self.directory.name, 'reveal.sock')
        identity['socket_path'] = path
        self.identity = identity
        self.binding = binding
        self.calls = []
        self.response_change = {}
        self.server = socket.socket(socket.AF_UNIX)
        self.server.bind(str(path))
        os.chmod(path, 0o600)
        self.server.listen()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            connection, _ = self.server.accept()
            with connection, connection.makefile('rwb') as stream:
                value = json.loads(stream.readline())
                if value.get('action') == 'close-fixture':
                    return
                self.calls.append(value)
                response = {'schema': 'gpu-greenroom.application-reveal-receipt.v1',
                    'action': value['action'], 'call_id': value['call_id'],
                    **self.binding, 'reveal': self.identity, 'foreground': value['action'] == 'reveal',
                    **self.response_change}
                stream.write(json.dumps(response).encode() + b'\n')
                stream.flush()

    def close(self):
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(self.identity['socket_path'])
            client.sendall(b'{"action":"close-fixture"}\n')
        self.thread.join()
        self.server.close()
        self.directory.cleanup()


def revealed_store(root, monkeypatch):
    store, value, record, path = browser_store(root)
    from gpu_queue.queue import GPUQueue
    monkeypatch.setattr(GPUQueue, '_process_start_identity', lambda pid: 'start-' + str(pid))
    state = json.loads((path / 'status.json').read_text())
    state['pid'] = os.getpid()
    (path / 'status.json').write_text(json.dumps(state))
    identity = {'schema': 'gpu-greenroom.application-reveal.v1', 'socket_path': str(root / 'reveal.sock'),
        'session_id': str(uuid4()), 'owner_pid': os.getpid(), 'owner_start_identity': 'start-' + str(os.getpid()),
        'browser_pid': os.getpid(), 'browser_start_identity': 'start-' + str(os.getpid()), 'page_id': 'observed-page'}
    binding = {'request_id': value['id'], 'request_digest': record['request_digest'],
        'job_id': 'test-job', 'url': 'http://127.0.0.1:61946/log'}
    owner = Owner(root / 'reveal.sock', identity, binding)
    try:
        record = store.publish_application(value['id'], {
            'url': binding['url'], 'phase': 'operator-needed', 'label': 'Cold log', 'reveal': identity,
        })
    except Exception:
        owner.close()
        raise
    return store, value, record, path, owner


def test_browser_focus_requests_its_retained_owner_without_url_navigation(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        destination = store.destination(record)
        assert store.display(record)['application_verified'] is True
        first = store.focus(value['id'], record['request_digest'], destination['identity_digest'])
        second = store.focus(value['id'], record['request_digest'], destination['identity_digest'])
        assert first['foreground'] is True and second['foreground'] is True
        assert first['reveal'] == second['reveal'] == owner.identity
        assert [call['action'] for call in owner.calls].count('reveal') == 2
        assert all(call['job_id'] == 'test-job' for call in owner.calls)
        assert store.get(value['id']) == record
    finally:
        owner.close()


@pytest.mark.parametrize('change', [{'job_id': 'other'}, {'url': 'http://127.0.0.1:61946/other'},
                                   {'reveal': {}}, {'call_id': 'old-call'}])
def test_wrong_owner_receipt_cannot_authorize_browser_reveal(tmp_path, monkeypatch, change):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        owner.response_change = change
        with pytest.raises(ValueError, match='identity|receipt'):
            store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest'])
        assert not any(call['action'] == 'reveal' for call in owner.calls)
    finally:
        owner.close()


def test_reveal_requires_foreground_observation_not_acknowledgement(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        owner.response_change = {'foreground': False}
        with pytest.raises(ValueError, match='foreground'):
            store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest'])
    finally:
        owner.close()


def test_ended_or_recycled_process_does_not_receive_reveal(tmp_path, monkeypatch):
    store, value, record, path, owner = revealed_store(tmp_path, monkeypatch)
    try:
        from gpu_queue.queue import GPUQueue
        monkeypatch.setattr(GPUQueue, '_process_start_identity', lambda pid: 'different-start')
        with pytest.raises(ValueError, match='process'):
            store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest'])
        assert not any(call['action'] == 'reveal' for call in owner.calls)
        monkeypatch.setattr(GPUQueue, '_process_start_identity', lambda pid: 'start-' + str(pid))
        (tmp_path / 'done').mkdir()
        path = path.rename(tmp_path / 'done' / path.name)
        (path / 'status.json').write_text(json.dumps({'job_id': 'test-job', 'status': 'done'}))
        with pytest.raises(ValueError, match='running'):
            store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest'])
        assert not any(call['action'] == 'reveal' for call in owner.calls)
    finally:
        owner.close()


def test_replaced_browser_destination_rejects_the_old_button(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        previous = store.destination(record)['identity_digest']
        changed = deepcopy(owner.identity)
        changed['session_id'] = str(uuid4())
        owner.identity = changed
        replacement = store.publish_application(value['id'], {
            'url': owner.binding['url'], 'phase': 'operator-needed', 'label': 'New attempt', 'reveal': changed,
        })
        assert store.destination(replacement)['identity_digest'] != previous
        with pytest.raises(ValueError, match='destination changed'):
            store.focus(value['id'], record['request_digest'], previous)
        assert not any(call['action'] == 'reveal' for call in owner.calls)
    finally:
        owner.close()


def test_url_publication_alone_does_not_generate_a_usable_entry_notification(tmp_path):
    from gpu_queue.smoke_delivery import SmokeDelivery
    store, value, _, _ = browser_store(tmp_path)
    record = store.publish_application(value['id'], {
        'url': 'http://127.0.0.1:61946/log', 'phase': 'operator-needed', 'label': 'Cold log',
    })
    assert SmokeDelivery(store).events(record) == []


def test_verified_entry_notification_preserves_exact_destination(tmp_path, monkeypatch):
    from gpu_queue.smoke_delivery import SmokeDelivery
    store, _, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        events = SmokeDelivery(store).events(record)
        assert len(events) == 1 and events[0]['kind'] == 'usable'
        assert events[0]['payload']['destination'] == store.destination(record)
    finally:
        owner.close()
