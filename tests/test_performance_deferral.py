import fcntl
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from gpu_queue import performance_window
from gpu_queue.models import JobRequest
from gpu_queue.operator_server import queue_snapshot
from gpu_queue.queue import GPUQueue
from tests.test_performance_window import spec


def test_deferral_never_launches_and_records_selected_delay(tmp_path, monkeypatch):
    value = spec(tmp_path)
    monkeypatch.setattr(performance_window.time, 'time', lambda: 1000.)
    assert performance_window.run(value, prompt=lambda _: {'defer_minutes': '5'}) == 75
    report = json.loads(Path(value['receipt_path']).read_text())
    assert report['status'] == 'deferred'
    assert report['deferred_until'] == 1300
    assert report['defer_seconds'] == 300
    assert report['workload_started'] is False
    assert not (tmp_path / 'executed').exists()


@pytest.mark.parametrize('minutes', ['0', '-1', 'nan', 'inf', '', 'tomorrow', True])
def test_invalid_delay_cannot_turn_into_confirmation(tmp_path, minutes):
    value = spec(tmp_path)
    assert performance_window.run(value, prompt=lambda _: {'defer_minutes': minutes}) == 125
    report = json.loads(Path(value['receipt_path']).read_text())
    assert report['status'] == 'failed'
    assert not (tmp_path / 'executed').exists()


def install_prompt(monkeypatch, answer):
    original = performance_window.runner_command
    def launch(job_dir, request, argv, timeout):
        actual = original(job_dir, request, argv, timeout)
        return [sys.executable, '-c',
            'import json,sys;from pathlib import Path;from gpu_queue.performance_window import run;'
            'sys.exit(run(json.loads(Path(sys.argv[1]).read_text()),prompt=lambda _:' + repr(answer) + '))', actual[-1]]
    monkeypatch.setattr(performance_window, 'runner_command', launch)
    monkeypatch.setattr(performance_window, 'native_notice', Mock())


def request(tmp_path):
    return JobRequest(job_type='command', input_path='', agent_id='test-owner',
        route_identity='test/timing', performance_window='Compare command cost',
        command_argv=[sys.executable, '-c', 'import time;time.sleep(.02)'])


def test_other_job_runs_while_same_measurement_is_deferred(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / 'queue')
    first = request(tmp_path)
    first_dir = queue.submit(first)
    original_request = (first_dir / 'request.json').read_bytes()
    install_prompt(monkeypatch, {'defer_minutes': '5'})
    assert queue.run_one({})
    state = queue.get_job(first.job_id)
    assert state.status.value == 'deferred', 'Deferral must not terminalize or keep the execution slot'
    assert state.started_at is None and state.finished_at is None
    assert (queue.queue_dir / 'deferred' / first.job_id / 'request.json').read_bytes() == original_request
    with queue.lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    second = JobRequest(job_type='command', input_path='', command_argv=[sys.executable, '-c', 'import time;time.sleep(.02)'])
    queue.submit(second)
    assert queue.run_one({})
    assert queue.get_job(second.job_id).status.value == 'done'
    assert queue.run_one({}) is False
    assert queue.get_job(first.job_id).status.value == 'deferred'
    performance_window.native_notice.assert_not_called()
    restarted = GPUQueue(queue.queue_dir)
    row = next(row for row in queue_snapshot(restarted, 'active')['jobs'] if row['job_id'] == first.job_id)
    assert row['consistent'] and row['status'] == 'deferred'
    assert row['performance_window']['workload_started'] is False
    assert restarted.cancel(first.job_id)
    assert restarted.get_job(first.job_id).status.value == 'cancelled'


def test_due_deferral_reoffers_without_duplicate_execution_and_respects_pause(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / 'queue')
    first = request(tmp_path)
    queue.submit(first)
    install_prompt(monkeypatch, {'defer_minutes': '.001'})
    assert queue.run_one({})
    state = queue.get_job(first.job_id)
    assert state.status.value == 'deferred'
    deadline = state.performance_deferral['deferred_until']
    monkeypatch.undo()
    queue = GPUQueue(queue.queue_dir)
    monkeypatch.setattr('gpu_queue.queue.time.time', lambda: deadline + 1)
    monkeypatch.setattr('gpu_queue.queue.time.monotonic', __import__('time').monotonic)
    pause = queue.pause(owner='test-owner', epoch='test-pause')
    assert queue.run_one({}) is False
    assert queue.get_job(first.job_id).status.value == 'deferred'
    queue.resume(owner='test-owner', epoch=pause['epoch'])
    install_prompt(monkeypatch, True)
    assert queue.run_one({})
    final = queue.queue_dir / 'done' / first.job_id
    assert final.is_dir()
    assert len(list((final / 'performance-deferrals').glob('*.json'))) == 1
    assert json.loads((final / 'receipt.json').read_text())['performance_window']['status'] == 'complete'
    assert queue.run_one({}) is False


def test_native_prompt_has_editable_default_and_returns_exact_text(monkeypatch):
    cf = Mock()
    cf.CFStringCreateWithCString.return_value = 10
    cf.CFUserNotificationCreate.return_value = 20
    monkeypatch.setattr(performance_window.ctypes.c_void_p, 'in_dll', lambda *_: Mock(value=100))
    monkeypatch.setattr(performance_window.ctypes, 'CDLL', lambda *_: cf)
    monkeypatch.setattr(performance_window.sys, 'platform', 'darwin')
    def receive(*args):
        args[-1]._obj.value = 2
        return 0
    cf.CFUserNotificationReceiveResponse.side_effect = receive
    cf.CFUserNotificationGetResponseValue.return_value = 30
    cf.CFStringGetLength.return_value = 4
    cf.CFStringGetMaximumSizeForEncoding.return_value = 12
    def read(_value, buffer, _size, _encoding):
        buffer.value = b'12.5'
        return True
    cf.CFStringGetCString.side_effect = read
    assert performance_window.native_dialog({'job_id': 'one', 'owner': 'owner', 'route': 'route', 'reason': 'timing'}) == {'defer_minutes': '12.5'}
    strings = [call.args[1] for call in cf.CFStringCreateWithCString.call_args_list]
    assert b'5' in strings and b'Remind me later' in strings and b'Remind me in (minutes)' in strings
    assert cf.CFRelease.call_count == 11


def test_native_invalid_edit_reoffers_instead_of_losing_job(monkeypatch):
    prompt = Mock(side_effect=[{'defer_minutes': '0'}, {'defer_minutes': '2.5'}])
    monkeypatch.setattr(performance_window, '_native_prompt', prompt)
    assert performance_window.native_dialog({}) == {'defer_minutes': '2.5'}
    assert prompt.call_args_list[0].args == ({}, None)
    assert 'positive' in prompt.call_args_list[1].args[1]


@pytest.mark.parametrize('crash_boundary', ['before-commit', 'after-status'])
def test_crashed_deferral_is_recovered_once_without_execution(tmp_path, monkeypatch, crash_boundary):
    queue = GPUQueue(tmp_path / 'queue')
    first = request(tmp_path)
    queue.submit(first)
    install_prompt(monkeypatch, {'defer_minutes': '5'})
    if crash_boundary == 'after-status':
        original = queue._move_job
        def crash(job, destination):
            if destination == 'deferred':
                raise SystemExit('simulated crash before rename')
            return original(job, destination)
        monkeypatch.setattr(queue, '_move_job', crash)
    else:
        original = performance_window.atomic_write
        def crash(path, value):
            if Path(path).name == 'performance-defer.json':
                raise SystemExit('simulated crash before commit')
            return original(path, value)
        monkeypatch.setattr(performance_window, 'atomic_write', crash)
    with pytest.raises(SystemExit, match='simulated crash'):
        queue.run_one({})
    monkeypatch.undo()
    monkeypatch.setattr(performance_window, 'native_notice', Mock())
    restarted = GPUQueue(queue.queue_dir)
    assert restarted.recover_stale() == [first.job_id]
    assert restarted.recover_stale() == []
    assert restarted.get_job(first.job_id).status.value == 'deferred'
    assert restarted.run_one({}) is False
    assert len(list((queue.queue_dir / 'deferred' / first.job_id / 'performance-deferrals').glob('*.json'))) == 1
    performance_window.native_notice.assert_not_called()


def test_changed_deferred_request_is_held_not_executed(tmp_path, monkeypatch):
    queue = GPUQueue(tmp_path / 'queue')
    first = request(tmp_path)
    queue.submit(first)
    install_prompt(monkeypatch, {'defer_minutes': '.001'})
    assert queue.run_one({})
    job = queue.queue_dir / 'deferred' / first.job_id
    record = json.loads((job / 'request.json').read_text())
    record['route_identity'] = 'changed/route'
    (job / 'request.json').write_text(json.dumps(record))
    assert queue.run_one({}) is False
    assert (job / 'deferral-error.json').is_file()
    assert not list((queue.queue_dir / 'running').iterdir())
    assert queue.get_job(first.job_id).status.value == 'deferred'
