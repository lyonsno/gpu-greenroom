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
        self.directory = tempfile.TemporaryDirectory(prefix='gr-', dir='/tmp')
        path = os.path.join(self.directory.name, 'reveal.sock')
        identity['socket_path'] = path
        self.identity = identity
        self.binding = binding
        self.calls = []
        self.response_change = {}
        self.reject_second_reveal = False
        self.receipt_gate = threading.Event()
        self.receipt_gate.set()
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
                self.receipt_gate.wait()
                response = {'schema': 'gpu-greenroom.application-reveal-receipt.v1',
                    'action': value['action'], 'call_id': value['call_id'],
                    **self.binding, 'reveal': self.identity, 'foreground': value['action'] == 'reveal',
                    **self.response_change}
                if self.reject_second_reveal and len([call for call in self.calls if call['action'] == 'reveal']) == 2:
                    response['foreground'] = False
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


def wait_action(store, action):
    from gpu_queue import smoke_browser_actions
    key = smoke_browser_actions._key(store.directory, action['request_id'], action['destination_digest'])
    state = smoke_browser_actions._actions[key]
    assert state['finished'].wait(3), 'synthetic owner must finish its action'
    return smoke_browser_actions.latest(store.directory, action['request_id'], action['destination_digest'])


def test_browser_focus_requests_its_retained_owner_without_url_navigation(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        destination = store.destination(record)
        assert store.display(record)['application_navigation_available'] is True
        first = wait_action(store, store.focus(value['id'], record['request_digest'], destination['identity_digest']))
        second = wait_action(store, store.focus(value['id'], record['request_digest'], destination['identity_digest']))
        assert first['receipt']['foreground'] is True and second['receipt']['foreground'] is True
        assert first['receipt']['reveal'] == second['receipt']['reveal'] == owner.identity
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
        action = wait_action(store, store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest']))
        assert action['status'] == 'failed' and 'receipt identity' in action['error']
        assert not any(call['action'] == 'reveal' for call in owner.calls)
    finally:
        owner.close()


def test_reveal_requires_foreground_observation_not_acknowledgement(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    try:
        owner.response_change = {'foreground': False}
        action = wait_action(store, store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest']))
        assert action['status'] == 'failed' and 'foreground' in action['error']
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


def test_stalled_owner_does_not_block_list_focus_or_feedback(tmp_path, monkeypatch):
    store, value, record, _, owner = revealed_store(tmp_path, monkeypatch)
    focus_done, list_done, feedback_done = (threading.Event() for _ in range(3))
    results = {}
    def probe(name, done, call):
        try:
            results[name] = call()
        finally:
            done.set()
    probes = []
    owner.receipt_gate.clear()
    try:
        for name, done, call in [
            ('focus', focus_done, lambda: store.focus(value['id'], record['request_digest'], store.destination(record)['identity_digest'])),
            ('list', list_done, store.snapshot),
            ('feedback', feedback_done, lambda: store.respond(value['id'], 'Still usable while its owner is stuck')),
        ]:
            worker = threading.Thread(target=probe, args=(name, done, call))
            probes.append(worker)
            worker.start()
            assert done.wait(1), name + ' must not wait for a browser-owner receipt'
        assert results['focus']['status'] == 'pending'
        assert results['feedback']['status'] == 'responded'
    finally:
        owner.receipt_gate.set()
        for worker in probes:
            worker.join()
        if results.get('focus', {}).get('status') == 'pending':
            wait_action(store, results['focus'])
        owner.close()
