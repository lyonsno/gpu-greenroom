import hashlib
import json
import subprocess
import sys
import uuid

import pytest

from gpu_queue import admission, dispatch
from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue, worker_identity


def policy():
    return {'schema': 'gpu-greenroom.dispatch-policy.v4', 'mode': 'two-class',
            'producer_hints': {'max_seconds': 30}}


def manifest(tmp_path, **extra):
    return {'schema': 'gpu-greenroom.command.v5', 'argv': [sys.executable, '-c',
            'import time; time.sleep(.05); print("hint-smoke")'],
            'repo_root': str(tmp_path), 'cwd': str(tmp_path), 'route_identity': 'new/capture',
            'runtime_hint': {'expected_seconds': 10, 'reason': 'small preview'}, **extra}


def test_first_submission_hint_qualifies_without_history_or_profile():
    config = dispatch.validate(policy())
    kind, basis = dispatch.classify({'runtime_hint': {'expected_seconds': 10}}, config)
    assert kind == 'quick'
    assert 'producer-estimated' in basis
    assert dispatch.classify({'runtime_hint': {'expected_seconds': 31}}, config)[0] == 'normal'
    assert dispatch.classify({}, config)[0] == 'normal'
    assert dispatch.classify({'runtime_hint': {'expected_seconds': 10}, 'service_class': 'normal'}, config)[0] == 'normal'


def test_manifest_roundtrip_prepared_identity_and_observed_execution(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    queue.configure_dispatch(policy(), owner='operator')
    plan = admission.prepare(queue, manifest(tmp_path))
    assert plan['job_request']['runtime_hint']['expected_seconds'] == 10
    assert 'runtime-hint.v1' in plan['job_request']['required_worker_capabilities']
    directory = admission.submit_prepared(queue, plan)
    request = JobRequest.from_json((directory / 'request.json').read_text())
    admission.assert_current(request, plan['validation'])
    old = worker_identity()
    old['capabilities'].remove('runtime-hint.v1')
    assert not queue.run_one({}, claimant=old)
    assert queue.run_one({})
    receipt = json.loads((queue.queue_dir / 'done' / request.job_id / 'receipt.json').read_text())
    assert receipt['dispatch']['service_class'] == 'quick'
    assert receipt['effective_timeout'] is None
    comparison = receipt['runtime_hint_comparison']
    assert comparison['expected_seconds'] == 10
    assert comparison['actual_seconds'] > 0
    assert comparison['successful'] is True
    assert comparison['clock'] == 'execution-wall'
    assert comparison['condition'] == 'unknown'
    assert comparison['ratio'] == comparison['actual_seconds'] / 10


@pytest.mark.parametrize('hint', [False, 10, {}, {'expected_seconds': True},
    {'expected_seconds': 0}, {'expected_seconds': -1}, {'expected_seconds': float('nan')},
    {'expected_seconds': float('inf')}, {'expected_seconds': '10'},
    {'expected_seconds': 10, 'reason': 4}])
def test_invalid_hint_rejected_before_queueing(tmp_path, hint):
    queue = GPUQueue(tmp_path / 'queue')
    with pytest.raises(ValueError, match='runtime_hint'):
        payload = admission.normalize_manifest(manifest(tmp_path, runtime_hint=hint))
        queue.submit(admission.command_request(payload))
    assert not list((queue.queue_dir / 'pending').iterdir())


def test_hint_and_policy_cannot_hide_in_older_schemas(tmp_path):
    with pytest.raises(ValueError, match='command.v5'):
        admission.normalize_manifest(manifest(tmp_path, schema='gpu-greenroom.command.v4'))
    with pytest.raises(ValueError, match='v4'):
        dispatch.validate({**policy(), 'schema': dispatch.HISTORY_SCHEMA})
    with pytest.raises(ValueError, match='producer_hints'):
        dispatch.validate({**policy(), 'producer_hints': {'max_seconds': True}})


def test_hints_keep_fifo_within_classes_and_alternate_without_starvation(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    queue.configure_dispatch(policy(), owner='operator')
    for name, hint in [('normal1', None), ('normal2', None), ('quick1', 10), ('quick2', 5)]:
        request = admission.command_request(admission.normalize_manifest(manifest(tmp_path,
            runtime_hint=None if hint is None else {'expected_seconds': hint})))
        request.job_id = name
        queue.submit(request)
    assert [p.name for p in dispatch.pending_order(queue.queue_dir)] == ['quick1', 'normal1', 'quick2', 'normal2']


def test_manual_override_and_reconfiguration_preserve_hint_policy(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    queue.configure_dispatch({'schema': dispatch.OVERRIDE_SCHEMA, 'mode': 'two-class'}, owner='operator')
    queue.configure_dispatch(policy(), owner='operator')
    assert dispatch.policy(queue.queue_dir)['schema'] == policy()['schema']
    request = admission.command_request(admission.normalize_manifest(manifest(tmp_path)))
    directory = queue.submit(request)
    queue.override_dispatch(job_id=request.job_id, service_class='normal', request_id=str(uuid.uuid4()),
        requested_by='operator', request_digest=hashlib.sha256((directory/'request.json').read_bytes()).hexdigest())
    config = dispatch.policy(queue.queue_dir)
    assert config['schema'] == policy()['schema']
    assert config['producer_hints'] == {'max_seconds': 30}
    assert dispatch.classify(json.loads(request.to_json()), config,
        request_digest=hashlib.sha256((directory/'request.json').read_bytes()).hexdigest())[0] == 'normal'


def test_cli_hint_survives_submission_and_cannot_override_manifest(tmp_path):
    queue = GPUQueue(tmp_path / 'queue')
    args = [sys.executable, '-m', 'gpu_queue.cli', '--queue-dir', str(queue.queue_dir), 'submit-command']
    result = subprocess.run(args + ['--repo-root', str(tmp_path), '--cwd', str(tmp_path),
        '--route-identity', 'first/job', '--expected-seconds', '10', '--runtime-reason', 'preview',
        '--', sys.executable, '-c', 'print(1)'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    request = json.loads(open(json.loads(result.stdout)['request_path']).read())
    assert request['runtime_hint'] == {'expected_seconds': 10, 'reason': 'preview'}
    assert request['command_timeout'] is None
    path = tmp_path / 'command.json'
    path.write_text(json.dumps(manifest(tmp_path)))
    rejected = subprocess.run(args + ['--manifest', str(path), '--expected-seconds', '3'], capture_output=True, text=True)
    assert rejected.returncode == 2
    assert 'cannot override a manifest' in rejected.stderr


def test_observed_performance_wait_and_failure_are_not_successful_compute():
    from gpu_queue.runtime_hints import comparison
    hint = {'expected_seconds': 10}
    base = {'started_at': 1, 'finished_at': 101, 'status': 'done', 'exit_code': 0, 'child_pid': 123}
    perf = {'workload_started': True, 'command_wall_seconds': 2}
    result = comparison(hint, {**base, 'performance_window': perf})
    assert result['actual_seconds'] == 2
    assert result['execution_wall_seconds'] == 100
    assert result['clock'] == 'performance-command-wall'
    waiting = comparison(hint, {**base, 'performance_window': {'workload_started': False}})
    assert waiting['actual_seconds'] is None
    assert waiting['successful'] is False
    failed = comparison(hint, {**base, 'status': 'failed', 'exit_code': 1})
    assert failed['actual_seconds'] == 100
    assert failed['successful'] is False
    assert failed['over_estimate'] is True


def test_absent_hint_preserves_old_prepared_request_digest(tmp_path):
    request = JobRequest(job_type='echo', input_path='x')
    before = json.loads(request.to_json())
    before.pop('runtime_hint', None)
    before.pop('runtime_profile', None)
    before.pop('performance_window', None)
    before.pop('submitted_at')
    assert admission.request_identity(request) == admission.digest(before)


def test_extreme_valid_hint_keeps_comparison_json_finite():
    from gpu_queue.runtime_hints import comparison
    result = comparison({'expected_seconds': 1e-320},
        {'started_at': 1, 'finished_at': 2, 'child_pid': 1, 'status': 'done', 'exit_code': 0})
    json.dumps(result, allow_nan=False)
    assert result['actual_seconds'] == 1
    assert result['ratio'] is None
    assert result['ratio_unrepresentable'] is True


def test_console_projects_hint_and_terminal_comparison(tmp_path):
    from gpu_queue.operator_server import queue_snapshot
    queue = GPUQueue(tmp_path/'queue')
    request = admission.command_request(admission.normalize_manifest(manifest(tmp_path)))
    queue.submit(request)
    pending = queue_snapshot(queue)['jobs'][0]
    assert pending['runtime_hint'] == request.runtime_hint
    assert pending.get('runtime_hint_comparison') is None
    assert queue.run_one({})
    completed = queue_snapshot(queue, 'done')['jobs'][0]
    assert completed['runtime_hint_comparison']['actual_seconds'] > 0
    assert completed['runtime_hint_comparison']['authority'] == 'producer-estimated'
    path = queue.queue_dir/'done'/request.job_id/'receipt.json'
    receipt = json.loads(path.read_text())
    receipt['job_id'] = 'wrong-job'
    path.write_text(json.dumps(receipt))
    mismatched = queue_snapshot(queue, 'done')['jobs'][0]
    assert 'does not match' in mismatched['runtime_hint_error']
    assert 'runtime_hint_comparison' not in mismatched


@pytest.mark.parametrize('field,value', [('started_at', 1), ('finished_at', 1000),
    ('workload_argv', ['different-command']), ('effective_cwd', '/different'),
    ('requested_route', 'wrong-route')])
def test_console_rejects_conflicting_observation_identity(tmp_path, field, value):
    from gpu_queue.operator_server import queue_snapshot
    queue = GPUQueue(tmp_path/'queue')
    request = admission.command_request(admission.normalize_manifest(manifest(tmp_path)))
    queue.submit(request)
    assert queue.run_one({})
    path = queue.queue_dir/'done'/request.job_id/'receipt.json'
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    row = queue_snapshot(queue, 'done')['jobs'][0]
    assert row.get('runtime_hint_error')
    assert 'runtime_hint_comparison' not in row


@pytest.mark.parametrize('flags', [['--expected-seconds','-1'],['--runtime-reason','orphan']])
@pytest.mark.parametrize('command', ['submit', 'submit-command'])
def test_invalid_cli_hint_has_durable_validation_failure(tmp_path, flags, command):
    queue = tmp_path/'queue'
    args = [sys.executable,'-m','gpu_queue.cli','--queue-dir',str(queue),command]
    args += ['echo','input'] if command == 'submit' else [
        '--repo-root',str(tmp_path),'--cwd',str(tmp_path),'--route-identity','invalid-hint']
    result = subprocess.run(args+flags,capture_output=True,text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads(result.stderr)
    assert report['failure_phase'] == 'submission-validation'
    assert list((queue/'submission-failures').glob('*.json'))
    assert not list((queue/'pending').iterdir())


@pytest.mark.parametrize('mutation', ['duration','missing'])
def test_performance_hint_requires_agreeing_command_clock(tmp_path, monkeypatch, mutation):
    from gpu_queue import performance_window
    from gpu_queue.operator_server import queue_snapshot
    original = performance_window.runner_command
    def accepted_cpu_gate(job_dir, request, argv, timeout):
        actual = original(job_dir, request, argv, timeout)
        return [sys.executable,'-c',
            'import json,sys;from pathlib import Path;from gpu_queue import performance_window as p;'
            'p.native_notice=lambda *a,**k:None;'
            'sys.exit(p.run(json.loads(Path(sys.argv[1]).read_text()),prompt=lambda _:True))',actual[-1]]
    monkeypatch.setattr(performance_window, 'runner_command', accepted_cpu_gate)
    monkeypatch.setattr(performance_window, 'native_notice', lambda *a, **k: None)
    queue = GPUQueue(tmp_path/'queue')
    request = admission.command_request(admission.normalize_manifest(
        manifest(tmp_path, performance_window='Isolated CPU fixture')))
    queue.submit(request)
    assert queue.run_one({})
    valid = queue_snapshot(queue, 'done')['jobs'][0]
    assert valid['runtime_hint_comparison']['clock'] == 'performance-command-wall'
    assert valid['runtime_hint_comparison']['actual_seconds'] == valid['performance_window']['command_wall_seconds']
    path = queue.queue_dir/'done'/request.job_id/'receipt.json'
    receipt = json.loads(path.read_text())
    if mutation == 'duration':
        receipt['performance_window']['command_wall_seconds'] = 1000
    else:
        receipt['performance_window'] = None
    path.write_text(json.dumps(receipt))
    row = queue_snapshot(queue, 'done')['jobs'][0]
    assert row.get('runtime_hint_error')
    assert 'runtime_hint_comparison' not in row
