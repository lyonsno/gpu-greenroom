import json
import subprocess
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from gpu_queue.smoke_requests import SmokeRequests
from tests.test_smoke_progress import job, request


def test_malformed_session_is_reported_without_breaking_the_list(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    record, _ = store.submit(request(tmp_path, 'test-job'))
    record['presentation'] = {'target': {'kind': 'terminal'}}
    record['session'] = {'schema': 'gpu-greenroom.smoke-session.v1'}
    store._write(record)
    snapshot = store.snapshot()
    assert snapshot['items'] == []
    assert 'session' in snapshot['errors'][0]


def test_dead_terminal_process_returns_explicit_unavailable(monkeypatch):
    from gpu_queue import smoke_navigation
    def run(argv, **kwargs):
        if argv[0] == 'ps':
            raise subprocess.CalledProcessError(1, argv, stderr='process vanished')
        return subprocess.CompletedProcess(argv, 0, json.dumps([{'pane_id': 46, 'tty_name': '/dev/ttys046'}]))
    monkeypatch.setattr(smoke_navigation.subprocess, 'run', run)
    from gpu_queue.queue import GPUQueue
    monkeypatch.setattr(GPUQueue, '_process_start_identity', lambda pid: 'start')
    with pytest.raises(ValueError, match='unavailable'):
        smoke_navigation.observe(46, 40433)


def test_native_failure_replaces_stale_availability_with_actual_diagnostics(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    value = request(tmp_path, 'test-job')
    record, _ = store.submit(value)
    path = job(tmp_path, 'failed')
    state = json.loads((path / 'status.json').read_text())
    state.update(failure_phase='execution', error_message='Output directory already exists', exit_code=1)
    (path / 'status.json').write_text(json.dumps(state))
    display = store.display(record)
    assert display['error'] == 'Output directory already exists'
    assert display['failure_phase'] == 'execution'
    assert display['section'] == 'history'


def test_legacy_links_are_context_not_application_targets(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    record, _ = store.submit(request(tmp_path))
    assert store.destination(record) == {'kind': 'context', 'url': record['request']['url']}


def test_terminal_request_does_not_require_a_placeholder_browser_url(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    value = request(tmp_path)
    value.pop('url')
    value['target'] = {'kind': 'terminal'}
    record, _ = store.submit(value)
    assert store.destination(record) == {'kind': 'terminal'}


def test_finished_terminal_asks_for_feedback_not_another_live_session(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    value = request(tmp_path, 'test-job')
    value.update(availability='prepared', target={'kind': 'terminal'})
    record, _ = store.submit(value)
    job(tmp_path, 'done')
    display = store.display(record)
    assert display['phase'] == 'awaiting-response'
    assert display.get('terminal_verified') is not True
    assert store.respond(value['id'], 'Inspected before it ended')['status'] == 'responded'


def test_ui_does_not_claim_observed_use_from_a_producer_declaration():
    from gpu_queue.operator_server import PAGE
    assert "interactive:'Interactive (reported)'" in PAGE


def test_browser_destination_and_diagnostic_class_preserve_original_request(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    record, _ = store.submit(request(tmp_path))
    original = deepcopy(record['request'])
    configured = store.configure(original['id'], {
        'target': {'kind': 'browser', 'url': 'http://127.0.0.1:9901/actual-app'},
        'purpose': 'diagnostic',
    })
    assert configured['request'] == original
    assert configured['request_digest'] == record['request_digest']
    assert store.destination(configured)['url'].endswith('/actual-app')
    assert store.display(configured)['section'] == 'history'
    with pytest.raises(ValueError):
        store.configure(original['id'], {'target': {'kind': 'browser', 'url': 'javascript:alert(1)'}})
    assert store.get(original['id']) == configured


def test_terminal_publication_is_bound_to_running_job_and_live_identity(tmp_path, monkeypatch):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    value = request(tmp_path, 'test-job')
    record, _ = store.submit(value)
    path = job(tmp_path, 'pending')
    assert callable(getattr(store, 'publish_session', None)), 'terminal publication contract missing'
    from gpu_queue import smoke_navigation
    monkeypatch.setattr(smoke_navigation, 'observe', lambda pane, pid: {
        'pane_id': pane, 'pid': pid, 'tty_name': '/dev/ttys046', 'process_start_identity': 'observed-start',
    })
    payload = {'pane_id': 46, 'pid': 40433, 'phase': 'operator-needed', 'label': 'Microphone closed; press Enter'}
    with pytest.raises(ValueError, match='running'):
        store.publish_session(value['id'], payload)
    (tmp_path / 'running').mkdir()
    path.rename(tmp_path / 'running' / path.name)
    path = tmp_path / 'running' / path.name
    (path / 'status.json').write_text(json.dumps({'job_id': 'test-job', 'status': 'running'}))
    changed = store.publish_session(value['id'], payload)
    monkeypatch.setattr(smoke_navigation, 'verify', lambda identity: identity)
    display = store.display(changed)
    assert display['phase'] == 'operator-needed'
    assert display['job_state'] == 'running'
    assert store.destination(changed)['pane_id'] == 46
    calls = []
    monkeypatch.setattr(smoke_navigation, 'focus', lambda identity: calls.append(identity))
    store.focus(value['id'], record['request_digest'], store.destination(changed)['identity_digest'])
    assert calls[0]['process_start_identity'] == 'observed-start'
    monkeypatch.setattr(smoke_navigation, 'verify', lambda identity: (_ for _ in ()).throw(ValueError('stale process')))
    assert store.display(changed)['phase'] == 'unknown'
    with pytest.raises(ValueError, match='stale'):
        store.focus(value['id'], record['request_digest'], store.destination(changed)['identity_digest'])
    assert len(calls) == 1


def test_terminal_publication_rejects_wrong_owner_and_terminal_jobs(tmp_path):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    value = request(tmp_path, 'test-job')
    store.submit(value)
    job(tmp_path, 'running', owner='other')
    with pytest.raises(ValueError, match='owner'):
        store.publish_session(value['id'], {'pane_id': 46, 'pid': 40433, 'phase': 'loading', 'label': 'Loading'})


def test_http_cannot_publish_terminal_selection_or_command_surface(tmp_path):
    from http.server import ThreadingHTTPServer
    import threading
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from gpu_queue.operator_server import make_handler
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tmp_path, 'secret', admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    value = request(tmp_path)
    value['target'] = {'kind': 'terminal', 'pane_id': 46}
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/smoke-requests',
                            data=json.dumps(value).encode(), headers={'Authorization': 'Bearer secret'}))
        assert error.value.code == 400
        assert not (tmp_path / 'smoke-requests').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def terminal_store(root, monkeypatch):
    from gpu_queue import smoke_navigation
    store = SmokeRequests(root / 'smoke-requests')
    value = request(root, 'test-job')
    value['availability'] = 'prepared'
    store.submit(value)
    job(root, 'running')
    monkeypatch.setattr(smoke_navigation, 'observe', lambda pane, pid: {
        'pane_id': pane, 'pid': pid, 'tty_name': f'/dev/ttys{pane}',
        'process_start_identity': f'start-{pid}',
    })
    monkeypatch.setattr(smoke_navigation, 'verify', lambda identity: identity)
    return store, value


def test_focus_is_bound_to_the_terminal_shown_not_the_original_request(tmp_path, monkeypatch):
    from gpu_queue import smoke_navigation
    store, value = terminal_store(tmp_path, monkeypatch)
    first = store.publish_session(value['id'], {
        'pane_id': 46, 'pid': 40433, 'phase': 'operator-needed', 'label': 'First terminal',
    })
    first_destination = store.destination(first)
    assert 'identity_digest' in first_destination, 'Focus needs the displayed destination identity'
    phase_only = store.publish_session(value['id'], {
        'pane_id': 46, 'pid': 40433, 'phase': 'interactive', 'label': 'Conversation active',
    })
    assert store.destination(phase_only)['identity_digest'] == first_destination['identity_digest']
    replacement = store.publish_session(value['id'], {
        'pane_id': 47, 'pid': 40434, 'phase': 'operator-needed', 'label': 'Replacement terminal',
    })
    assert replacement['request_digest'] == first['request_digest']
    calls = []
    monkeypatch.setattr(smoke_navigation, 'focus', lambda identity: calls.append(identity) or {'pane_id': identity['pane_id']})
    with pytest.raises(ValueError, match='destination changed'):
        store.focus(value['id'], first['request_digest'], first_destination['identity_digest'])
    assert calls == []
    receipt = store.focus(value['id'], replacement['request_digest'], store.destination(replacement)['identity_digest'])
    assert receipt['pane_id'] == 47


@pytest.mark.parametrize('old_terminal_live', [True, False])
def test_browser_reconfiguration_drops_obsolete_terminal_authority(tmp_path, monkeypatch, old_terminal_live):
    from gpu_queue import smoke_navigation
    store, value = terminal_store(tmp_path, monkeypatch)
    store.publish_session(value['id'], {
        'pane_id': 46, 'pid': 40433, 'phase': 'interactive', 'label': 'Old conversation',
    })
    if not old_terminal_live:
        monkeypatch.setattr(smoke_navigation, 'verify', lambda identity: (_ for _ in ()).throw(ValueError('old terminal exited')))
    browser = store.configure(value['id'], {'target': {'kind': 'browser', 'url': 'http://127.0.0.1:9901/actual-app'}})
    assert 'session' not in browser, 'Browser destination must not inherit terminal authority'
    display = store.display(browser)
    assert display['phase'] == 'running'
    assert not display.get('terminal_verified') and not display.get('error')
    (tmp_path / 'done').mkdir()
    (tmp_path / 'running' / 'test-job').rename(tmp_path / 'done' / 'test-job')
    (tmp_path / 'done' / 'test-job' / 'status.json').write_text(json.dumps({'job_id': 'test-job', 'status': 'done'}))
    assert store.respond(value['id'], 'Inspected browser')['status'] == 'responded'


def test_host_witness_reports_source_setup_failure_without_launching(tmp_path):
    output = tmp_path / 'witness'
    witness = Path(__file__).with_name('smoke_destination_host_witness.py')
    result = subprocess.run([sys.executable, str(witness), str(tmp_path / 'missing-root' / 'navigator.py'),
                             str(tmp_path / 'missing-browser'), str(output)],
                            capture_output=True, text=True, cwd=witness.parent.parent)
    assert result.returncode == 1
    assert (output / 'report.json').is_file(), result.stderr
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed' and report['phase'] == 'source-discovery'
    assert 'FileNotFoundError' in report['error']
    assert 'greenroom_revision' in report and 'navigator_revision' not in report
    assert not (output / 'terminal-pid.json').exists()
    again = subprocess.run([sys.executable, str(witness), str(tmp_path / 'missing-root' / 'navigator.py'),
                            str(tmp_path / 'missing-browser'), str(output)],
                           capture_output=True, text=True, cwd=witness.parent.parent)
    assert again.returncode == 1
    assert json.loads((output / 'report.json').read_text()) == report


@pytest.mark.parametrize('stdout', ['[]', 'null', '"unexpected"'])
def test_focus_http_rejects_non_object_navigator_receipts(tmp_path, monkeypatch, stdout):
    from http.server import ThreadingHTTPServer
    import threading
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError
    from gpu_queue import smoke_navigation
    from gpu_queue.operator_server import make_handler
    store, value = terminal_store(tmp_path, monkeypatch)
    record = store.publish_session(value['id'], {
        'pane_id': 46, 'pid': 40433, 'phase': 'operator-needed', 'label': 'Fixture',
    })
    monkeypatch.setenv('GPU_GREENROOM_TERMINAL_NAVIGATOR', sys.executable)
    monkeypatch.setattr(smoke_navigation.subprocess, 'run',
                        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, stdout, ''))
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(tmp_path, 'secret', admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(Request(f'http://127.0.0.1:{server.server_port}/api/smoke-requests/{value["id"]}/focus',
                            data=json.dumps({'request_digest': record['request_digest'],
                                             'destination_digest': store.destination(record)['identity_digest']}).encode(),
                            headers={'Authorization': 'Bearer secret'}))
        assert error.value.code == 400
        body = json.loads(error.value.read())
        assert 'receipt is not an object' in body['error']
        assert 'activation may have occurred' in body['error']
        assert store.get(value['id']) == record
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
