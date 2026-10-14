import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import Mock

import pytest

from gpu_queue import admission
from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue
from gpu_queue import performance_window
from gpu_queue.operator_server import queue_snapshot
from tests.test_smoke_admission import command


def test_prepared_performance_job_requires_capable_worker(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    manifest = command(tmp_path)
    manifest['performance_window'] = 'Compare retained speech replay timing'
    plan = admission.prepare(queue, manifest)
    request = JobRequest.from_json(json.dumps(plan['job_request']))
    assert getattr(request, 'performance_window', None) == manifest['performance_window'], 'performance intent silently dropped'
    assert 'performance-window.v1' in request.required_worker_capabilities
    admission.submit_prepared(queue, plan)
    old = {'capabilities': ['structured-command.v1', 'structured-command.v2', admission.CAPABILITY]}
    assert queue.run_one({}, claimant=old) is False
    assert not list((queue.queue_dir / 'running').iterdir())


def test_flag_preserves_performance_intent(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'gpu_queue.cli', '--queue-dir', str(tmp_path / 'queue'),
        'submit-command', '--repo-root', str(tmp_path), '--cwd', str(tmp_path),
        '--route-identity', 'test/measurement', '--performance-window', 'Measure frame cost',
        '--', sys.executable, '-c', 'pass'], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    record = json.loads(next((tmp_path / 'queue' / 'pending').glob('*/request.json')).read_text())
    assert record['performance_window'] == 'Measure frame cost'
    assert 'performance-window.v1' in record['required_worker_capabilities']


def spec(root):
    return {'job_id': 'test-job', 'owner': 'test-owner', 'route': 'test/performance',
            'reason': 'Measure timing', 'receipt_path': str(root / 'performance-window.json'),
            'argv': [sys.executable, '-c', f"from pathlib import Path; Path({str(root / 'executed')!r}).write_text('ran')"],
            'timeout': 1}


@pytest.mark.parametrize('answer', [False, None, 1, 'yes'])
def test_only_actual_confirmation_launches_command(tmp_path, answer):
    value = spec(tmp_path)
    def prompt(_):
        report = json.loads(Path(value['receipt_path']).read_text())
        assert report['status'] == 'waiting' and not report['workload_started']
        assert not (tmp_path / 'executed').exists()
        return answer
    assert performance_window.run(value, prompt=prompt) == 125
    report = json.loads(Path(value['receipt_path']).read_text())
    assert report['status'] == 'cancelled' and report['workload_started'] is False
    assert not (tmp_path / 'executed').exists()


def test_prompt_time_does_not_spend_command_timeout(tmp_path, monkeypatch):
    value = spec(tmp_path)
    clock = [100.0]
    child = Mock(pid=123)
    child.wait.return_value = 0
    monkeypatch.setattr(performance_window.time, 'time', lambda: clock[0])
    monkeypatch.setattr(performance_window.subprocess, 'Popen', Mock(return_value=child))
    def prompt(_):
        clock[0] += 20
        return True
    assert performance_window.run(value, prompt=prompt) == 0
    report = json.loads(Path(value['receipt_path']).read_text())
    assert report['status'] == 'complete'
    assert report['operator_wait_seconds'] == 20
    child.wait.assert_called_once_with(timeout=1)
    assert report['quietness_proved'] is False


def test_dialog_failure_never_launches_and_has_phase_receipt(tmp_path):
    value = spec(tmp_path)
    assert performance_window.run(value, prompt=Mock(side_effect=OSError('console unavailable'))) == 125
    report = json.loads(Path(value['receipt_path']).read_text())
    assert report['status'] == 'failed' and report['phase'] == 'operator-confirmation'
    assert report['error'] == 'console unavailable' and not report['workload_started']
    assert not (tmp_path / 'executed').exists()


@pytest.mark.parametrize('response,confirmed', [(0, False), (1, True), (2, False), (3, False), (9, False)])
def test_native_response_is_exact_and_cfstrings_released(monkeypatch, response, confirmed):
    cf = Mock()
    cf.CFStringCreateWithCString.side_effect = [10, 11, 12, 13]
    def alert(*args):
        args[-1]._obj.value = response
        assert args[0:2] == (0, 2), 'No response timeout; caution-level alert'
        return 0
    cf.CFUserNotificationDisplayAlert.side_effect = alert
    monkeypatch.setattr(performance_window.sys, 'platform', 'darwin')
    monkeypatch.setattr(performance_window.ctypes, 'CDLL', lambda *_: cf)
    assert performance_window.native_dialog({'job_id': 'one', 'owner': 'owner', 'route': 'route', 'reason': 'timing'}) is confirmed
    assert cf.CFRelease.call_count == 4


def test_worker_records_native_gate_and_runs_exact_command(tmp_path, monkeypatch):
    value = spec(tmp_path)
    queue = GPUQueue(tmp_path / 'queue')
    request = JobRequest(job_type='command', input_path='', command_argv=value['argv'],
                         performance_window=value['reason'], agent_id=value['owner'], route_identity=value['route'])
    original = performance_window.runner_command
    def cpu_dialog(job_dir, request, argv, timeout):
        actual = original(job_dir, request, argv, timeout)
        return [sys.executable, '-c',
            'import json,sys;from pathlib import Path;from gpu_queue.performance_window import run;'
            'sys.exit(run(json.loads(Path(sys.argv[1]).read_text()),prompt=lambda _:True))', actual[-1]]
    monkeypatch.setattr(performance_window, 'runner_command', cpu_dialog)
    queue.submit(request)
    assert queue.run_one({})
    receipt = json.loads((queue.queue_dir / 'done' / request.job_id / 'receipt.json').read_text())
    assert receipt['performance_window']['status'] == 'complete'
    assert receipt['performance_window']['quietness_proved'] is False
    assert receipt['workload_argv'] == value['argv']
    assert receipt['performance_window']['workload_pid'] != receipt['child_pid']
    assert (tmp_path / 'executed').is_file()


def test_tampered_confirmation_cannot_claim_quiet_or_another_job(tmp_path):
    value = spec(tmp_path)
    performance_window.run(value, prompt=lambda _:True)
    request = JobRequest(job_type='command', input_path='', job_id=value['job_id'], agent_id=value['owner'],
                         route_identity=value['route'], performance_window=value['reason'])
    path = Path(value['receipt_path'])
    valid = json.loads(path.read_text())
    for changes in ({'job_id': 'other'}, {'quietness_proved': True}, {'status': 'unknown'},
                    {'status': 'complete', 'workload_started': False}):
        path.write_text(json.dumps({**valid, **changes}))
        with pytest.raises(ValueError):
            performance_window.read_receipt(path, request)


def test_console_distinguishes_waiting_confirmation_from_command_time(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    request = JobRequest(job_type='command', input_path='', performance_window='Measure frame timing')
    job = queue.submit(request)
    snapshot = queue_snapshot(queue, 'all')
    row = next(item for item in snapshot['jobs'] if item['job_id'] == request.job_id)
    assert row.get('performance_window_reason') == request.performance_window


@pytest.mark.parametrize('accepted', [False, True])
def test_worker_refusal_and_timeout_preserve_phase_and_release_group(tmp_path, monkeypatch, accepted):
    queue = GPUQueue(tmp_path / 'queue')
    marker = tmp_path / 'executed'
    code = f"from pathlib import Path;import time;Path({str(marker)!r}).write_text('ran');time.sleep(30)"
    request = JobRequest(job_type='command', input_path='', command_argv=[sys.executable, '-c', code],
                         performance_window='Measure timing', command_timeout=.1)
    original = performance_window.runner_command
    def cpu_dialog(job_dir, request, argv, timeout):
        actual = original(job_dir, request, argv, timeout)
        return [sys.executable, '-c',
            'import json,sys;from pathlib import Path;from gpu_queue.performance_window import run;'
            f'sys.exit(run(json.loads(Path(sys.argv[1]).read_text()),prompt=lambda _:{accepted!r}))', actual[-1]]
    monkeypatch.setattr(performance_window, 'runner_command', cpu_dialog)
    queue.submit(request)
    assert queue.run_one({})
    receipt = json.loads((queue.queue_dir / 'failed' / request.job_id / 'receipt.json').read_text())
    assert receipt['failure_phase'] == ('performance_window_timeout' if accepted else 'performance_window_operator-confirmation')
    assert receipt['performance_window']['workload_started'] is accepted
    if not accepted:
        assert not marker.exists()
    assert not list((queue.queue_dir / 'running').iterdir())
    assert queue._process_group_alive(receipt['child_process_group']) is False


def test_pause_still_precedes_confirmation(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / 'queue')
    request = JobRequest(job_type='command', input_path='', command_argv=[sys.executable, '-c', 'pass'],
                         performance_window='Measure timing')
    queue.submit(request)
    launch = Mock(side_effect=AssertionError('Must not launch while paused'))
    monkeypatch.setattr(performance_window, 'runner_command', launch)
    queue.pause()
    assert queue.run_one({}) is False
    launch.assert_not_called()


def test_native_api_error_and_unsupported_host_refuse(monkeypatch):
    value = {'job_id': 'one', 'owner': 'owner', 'route': 'route', 'reason': 'timing'}
    monkeypatch.setattr(performance_window.sys, 'platform', 'linux')
    with pytest.raises(RuntimeError, match='macOS console'):
        performance_window.native_dialog(value)
    monkeypatch.setattr(performance_window.sys, 'platform', 'darwin')
    cf = Mock()
    cf.CFStringCreateWithCString.side_effect = [10, 11, 12, 13]
    cf.CFUserNotificationDisplayAlert.return_value = 7
    monkeypatch.setattr(performance_window.ctypes, 'CDLL', lambda *_: cf)
    with pytest.raises(OSError):
        performance_window.native_dialog(value)
    assert cf.CFRelease.call_count == 4
