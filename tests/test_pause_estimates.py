import json
import sys
from pathlib import Path

import pytest

from gpu_queue import cooperative, dispatch, job_controls
from gpu_queue.models import JobRequest, JobState, JobStatus
from gpu_queue.queue import GPUQueue


@pytest.fixture
def attempt(tmp_path, monkeypatch):
    q = GPUQueue(tmp_path)
    request = JobRequest(job_type='command', input_path='', cooperative_checkpoint=True,
        agent_id='fixture', command_argv=[sys.executable, '-c', 'pass'])
    path = q.submit(request)
    path = path.rename(tmp_path/'running'/request.job_id)
    state = JobState.from_json((path/'status.json').read_text())
    state.status = JobStatus.RUNNING
    state.started_at = 100
    state.child_start_identity = 'synthetic-attempt'
    state.child_pid = 123
    (path/'status.json').write_text(state.to_json())
    monkeypatch.setattr(cooperative, '_context', lambda: (q, path, request, json.loads(state.to_json())))
    monkeypatch.setattr('time.time', lambda: 110)
    return q, path, request, state


def estimate(attempt):
    q, path, request, state = attempt
    return job_controls.projection(q.queue_dir, path, request, state, {})['pause_control']['estimate']


def test_producer_estimate_counts_down_without_authorizing_pause(attempt, monkeypatch):
    cooperative.report_pause_estimate(30, checkpoint='end of tile')
    monkeypatch.setattr('time.time', lambda: 120)
    result = estimate(attempt)
    assert result['status'] == 'estimated'
    assert result['remaining_seconds'] == 20
    assert result['reported_at'] == 110
    assert result['checkpoint'] == 'end of tile'
    assert result['authority'] == 'producer-estimate-only'
    assert not dispatch.policy(attempt[0].queue_dir).get('job_holds')
    monkeypatch.setattr('time.time', lambda: 141)
    assert estimate(attempt)['status'] == 'overdue'
    assert estimate(attempt)['remaining_seconds'] is None


def test_missing_and_withdrawn_estimates_are_unknown(attempt):
    assert estimate(attempt)['status'] == 'unknown'
    cooperative.report_pause_estimate(20)
    cooperative.report_pause_estimate(None)
    assert estimate(attempt)['status'] == 'unknown'


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '30'])
def test_invalid_estimate_does_not_overwrite(attempt, value):
    cooperative.report_pause_estimate(20)
    before = (attempt[1]/'pause-estimate.json').read_bytes()
    with pytest.raises(ValueError):
        cooperative.report_pause_estimate(value)
    assert (attempt[1]/'pause-estimate.json').read_bytes() == before


@pytest.mark.parametrize('field,value', [('job_id','wrong'), ('request_sha256','wrong'),
    ('started_at',99), ('child_start_identity','old'), ('child_pid',456),
    ('reported_at',200), ('seconds',True), ('seconds',float('nan'))])
def test_invalid_or_different_attempt_never_projects_estimate(attempt, field, value):
    cooperative.report_pause_estimate(20)
    path = attempt[1]/'pause-estimate.json'
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    assert estimate(attempt)['status'] == 'unknown'


def test_terminal_and_pending_attempts_do_not_reuse_estimate(attempt):
    cooperative.report_pause_estimate(20)
    for status in (JobStatus.DONE, JobStatus.PENDING):
        attempt[3].status = status
        assert estimate(attempt)['status'] == 'not-running'


def test_unmanaged_reporting_is_noop(monkeypatch):
    monkeypatch.delenv('GPU_GREENROOM_CONTEXT', raising=False)
    assert cooperative.report_pause_estimate(20) is False


def test_actual_worker_child_publishes_bound_estimate(tmp_path):
    q = GPUQueue(tmp_path/'q')
    code = "from gpu_queue.cooperative import report_pause_estimate; assert report_pause_estimate(12, checkpoint='case end')"
    r = JobRequest(job_type='command', input_path='', cooperative_checkpoint=True, agent_id='fixture',
        command_argv=[sys.executable,'-c',code],
        command_env={'PYTHONPATH':str(Path(__file__).resolve().parents[1])})
    q.submit(r)
    assert q.run_one({})
    assert q.get_job(r.job_id).status == JobStatus.DONE
    job = q.queue_dir/'done'/r.job_id
    record = json.loads((job/'pause-estimate.json').read_text())
    state = json.loads((job/'status.json').read_text())
    assert record['child_start_identity'] == state['child_start_identity']
    assert record['child_pid'] == state['child_pid']
    assert record['seconds'] == 12


def test_malformed_report_does_not_break_projection(attempt):
    (attempt[1]/'pause-estimate.json').write_text('{')
    assert estimate(attempt)['status'] == 'unknown'


def test_zero_estimate_is_not_pause_confirmation(attempt):
    cooperative.report_pause_estimate(0)
    assert estimate(attempt)['status'] == 'overdue'
    assert attempt[3].status == JobStatus.RUNNING
