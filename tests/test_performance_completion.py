import concurrent.futures
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from gpu_queue import performance_window
from gpu_queue.models import JobRequest, JobStatus
from gpu_queue.queue import GPUQueue


@pytest.fixture(autouse=True)
def native(monkeypatch):
    callback = Mock()
    callback.native_function = getattr(performance_window, 'native_notice', None)
    monkeypatch.setattr(performance_window, 'native_notice', callback, raising=False)
    return callback


def request():
    return JobRequest(job_type='command', input_path='', agent_id='timing-owner',
                      route_identity='timing/comparison', performance_window='Compare command cost',
                      command_argv=[sys.executable, '-c', 'pass'])


def terminal(req, **changes):
    return {'job_id': req.job_id, 'status': 'done', 'finished_at': 123.0,
            'failure_phase': None, **changes}


def cpu_prompt(monkeypatch):
    original = performance_window.runner_command
    def launch(job, req, argv, timeout):
        value = original(job, req, argv, timeout)
        return [sys.executable, '-c', 'import json,sys;from pathlib import Path;'
                'from gpu_queue.performance_window import run;'
                'sys.exit(run(json.loads(Path(sys.argv[1]).read_text()),prompt=lambda _:True))', value[-1]]
    monkeypatch.setattr(performance_window, 'runner_command', launch)


@pytest.mark.parametrize('unknown', [False, True])
def test_worker_notice_follows_receipt_cleanup_and_execution_unlock(tmp_path, monkeypatch, native, unknown):
    queue = GPUQueue(tmp_path / 'queue')
    req = request()
    cpu_prompt(monkeypatch)
    if unknown:
        monkeypatch.setattr(queue, '_quiesce_owned_child', lambda *_: False)
    queue.submit(req)
    def notice(title, message, level):
        job = queue.queue_dir / ('running' if unknown else 'done') / req.job_id
        receipt = json.loads((job / 'receipt.json').read_text())
        assert receipt['status'] == ('ownership_unknown' if unknown else 'done')
        lock = queue._try_execution_lock()
        assert lock is not None, 'notice still holds the execution mutex'
        lock.close()
        assert req.job_id in message and req.route_identity in message
        if unknown:
            assert 'uncertain' in title and 'may still be running' in message
            assert 'window has ended' not in message
        else:
            assert 'finished' in title and 'window has ended' in message
    native.side_effect = notice
    assert queue.run_one({})
    native.assert_called_once()


def test_notification_failure_does_not_fail_or_rerun_workload(tmp_path, monkeypatch, native):
    queue = GPUQueue(tmp_path / 'queue')
    req = request()
    cpu_prompt(monkeypatch)
    native.side_effect = OSError('console unavailable')
    queue.submit(req)
    assert queue.run_one({})
    job = queue.queue_dir / 'done' / req.job_id
    assert json.loads((job / 'receipt.json').read_text())['exit_code'] == 0
    notice = json.loads((job / 'performance-completion-notice.json').read_text())
    assert notice['status'] == 'failed' and notice['error'] == 'console unavailable'
    assert queue.run_one({}) is False


def test_ordinary_work_has_no_end_notice(tmp_path, native):
    queue = GPUQueue(tmp_path / 'queue')
    req = request()
    req.performance_window = None
    queue.submit(req)
    assert queue.run_one({})
    native.assert_not_called()


def test_concurrent_duplicate_notice_is_submitted_once(tmp_path, native):
    req = request()
    value = terminal(req)
    def send(_):
        performance_window.notify_completion(tmp_path, req, value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(send, range(8)))
    native.assert_called_once()
    saved = json.loads((tmp_path / 'performance-completion-notice.json').read_text())
    assert saved['status'] == 'submitted' and saved['human_observation'] == 'unverified'


def test_warning_can_be_followed_by_recovered_terminal_notice(tmp_path, native):
    req = request()
    performance_window.notify_completion(tmp_path, req,
        terminal(req, status='ownership_unknown', finished_at=None))
    performance_window.notify_completion(tmp_path, req,
        terminal(req, status='failed', failure_phase='stale_recovery'))
    assert native.call_count == 2
    assert (tmp_path / 'performance-ownership-warning.json').exists()
    assert (tmp_path / 'performance-completion-notice.json').exists()


def test_missing_terminal_evidence_never_announces_end(tmp_path, native):
    req = request()
    performance_window.notify_completion(tmp_path, req, terminal(req, finished_at=None))
    title, message, level = native.call_args.args
    assert 'uncertain' in title and 'window has ended' not in message


def test_wrong_job_receipt_cannot_notify(tmp_path, native):
    req = request()
    with pytest.raises(ValueError, match='identity'):
        performance_window.notify_completion(tmp_path, req, terminal(req, job_id='wrong-job'))
    native.assert_not_called()


def test_native_notice_uses_one_way_api_without_waiting_for_response(monkeypatch, native):
    cf = Mock()
    cf.CFStringCreateWithCString.side_effect = [10, 11, 12]
    cf.CFUserNotificationDisplayNotice.return_value = 0
    monkeypatch.setattr(performance_window.sys, 'platform', 'darwin')
    monkeypatch.setattr(performance_window.ctypes, 'CDLL', lambda *_: cf)
    native.native_function('Measurement finished', 'Processes stopped', 1)
    cf.CFUserNotificationDisplayNotice.assert_called_once_with(0, 1, None, None, None, 10, 11, 12)
    cf.CFUserNotificationDisplayAlert.assert_not_called()
    assert cf.CFRelease.call_count == 3


def test_conflicting_quiescence_phase_never_announces_end(tmp_path, native):
    req = request()
    performance_window.notify_completion(tmp_path, req,
        terminal(req, failure_phase='completion_quiescence_unresolved'))
    title, message, _ = native.call_args.args
    assert 'uncertain' in title and 'window has ended' not in message


@pytest.mark.parametrize('live', [False, True])
def test_recovery_notices_preserve_process_authority_and_unlock(tmp_path, monkeypatch, native, live):
    queue = GPUQueue(tmp_path / 'queue')
    req = request()
    job = queue._move_job(queue.submit(req), 'running')
    state = queue.get_job(req.job_id)
    state.status = JobStatus.RUNNING
    state.worker_pid = 666666
    state.child_pid = state.child_process_group = 777777
    state.child_start_identity = 'fixture-child'
    queue._write_text_atomic(job / 'status.json', state.to_json())
    monkeypatch.setattr(queue, '_pid_alive', lambda _: False)
    monkeypatch.setattr(queue, '_process_start_identity', lambda _: 'fixture-child' if live else None)
    monkeypatch.setattr(queue, '_process_group_alive', lambda _: live)
    monkeypatch.setattr('gpu_queue.queue.os.getpgid', lambda _: 777777)
    def notice(title, message, level):
        lock = queue._try_execution_lock()
        assert lock is not None
        lock.close()
        assert ('uncertain' in title) is live
    native.side_effect = notice
    assert queue.recover_stale() == ([] if live else [req.job_id])
    native.assert_called_once()
    queue.recover_stale()
    native.assert_called_once()


@pytest.mark.parametrize('failure', ['api', 'partial-string'])
def test_native_notice_failure_releases_strings_and_is_visible(monkeypatch, native, failure):
    cf = Mock()
    cf.CFStringCreateWithCString.side_effect = [10, None] if failure == 'partial-string' else [10, 11, 12]
    cf.CFUserNotificationDisplayNotice.return_value = 7
    monkeypatch.setattr(performance_window.sys, 'platform', 'darwin')
    monkeypatch.setattr(performance_window.ctypes, 'CDLL', lambda *_: cf)
    with pytest.raises((OSError, RuntimeError)):
        native.native_function('Measurement finished', 'Processes stopped', 1)
    assert cf.CFRelease.call_count == (1 if failure == 'partial-string' else 3)
