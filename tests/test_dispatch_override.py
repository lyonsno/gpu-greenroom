import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from gpu_queue import dispatch
from gpu_queue.models import JobRequest
from gpu_queue.operator_server import make_handler, queue_snapshot
from gpu_queue.queue import GPUQueue, worker_identity
from tests.test_quick_dispatch import policy, submit


def move(queue, job_id, service_class='quick', request_id=None, digest=None):
    path = queue.queue_dir / 'pending' / job_id / 'request.json'
    return queue.override_dispatch(
        job_id=job_id, service_class=service_class,
        request_digest=digest or hashlib.sha256(path.read_bytes()).hexdigest(),
        request_id=request_id or str(uuid4()), requested_by='operator-console',
    )


def test_manual_class_is_separate_from_measured_route_eligibility():
    request = {'job_id': 'fixture', 'job_type': 'long'}
    config = {**policy(), 'schema': 'gpu-greenroom.dispatch-policy.v2',
              'job_overrides': {'fixture': {
                  'schema': 'gpu-greenroom.dispatch-override.v1',
                  'request_id': str(uuid4()), 'job_id': 'fixture',
                  'request_digest': 'a' * 64, 'service_class': 'quick',
                  'requested_by': 'operator-console', 'at': 1,
                  'authority': 'authenticated-console; caller identity unverified',
              }}, 'action_receipts': {}}
    receipt = config['job_overrides']['fixture']
    config['action_receipts'][receipt['request_id']] = receipt
    config = dispatch.validate(config)
    assert dispatch.classify(request, config, request_digest='a' * 64)[0] == 'quick'
    assert dispatch.classify({**request, 'job_id': 'another'}, config)[0] == 'normal'
    with pytest.raises(ValueError, match='digest'):
        dispatch.classify(request, config, request_digest='b' * 64)


def test_move_is_durable_idempotent_and_keeps_request_and_validation(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    first = submit(queue, 'long', 1)
    second = submit(queue, 'long', 2)
    directory = tmp_path / 'pending' / second
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    identity = str(uuid4())
    receipt = move(queue, second, request_id=identity)
    assert move(queue, second, request_id=identity) == receipt
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before
    assert GPUQueue(tmp_path)._next_pending().name == second
    state = dispatch.policy(tmp_path)
    assert state['quick_job_types'] == policy()['quick_job_types']
    assert state['quick_routes'] == {}
    assert queue.get_job(second).submitted_at == 2
    with pytest.raises(ValueError, match='reused'):
        move(queue, second, 'normal', request_id=identity)
    move(queue, second, 'normal')
    assert queue._next_pending().name == first
    assert len(dispatch.policy(tmp_path)['action_receipts']) == 2
    queue.configure_dispatch(policy(), owner='fixture')
    assert len(dispatch.policy(tmp_path)['action_receipts']) == 2


def test_overridden_dispatch_preserves_fairness_and_rejects_old_claimant(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    first = submit(queue, 'long', 1)
    second = submit(queue, 'long', 2)
    third = submit(queue, 'long', 3)
    receipt = move(queue, second)
    move(queue, third)
    config = {'long': {'cmd': [sys.executable, '-c', 'pass']}}
    old = {**worker_identity(), 'capabilities': ['structured-command.v1', dispatch.CAPABILITY]}
    assert not queue.run_one(config, claimant=old)
    assert not list((tmp_path / 'running').iterdir())
    assert queue.run_one(config)
    terminal = json.loads((tmp_path / 'done' / second / 'receipt.json').read_text())
    assert terminal['dispatch']['service_class'] == 'quick'
    assert terminal['dispatch']['operator_override'] == receipt
    assert queue._next_pending().name == first
    assert queue.run_one(config)
    assert queue._next_pending().name == third
    replay = move(queue, second, request_id=receipt['request_id'], digest=receipt['request_digest'])
    assert replay == receipt  # retry after execution returns the original action, not a new move


def test_stale_or_claimed_job_never_changes_policy(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    old = (tmp_path / 'dispatch-policy.json').read_bytes()
    with pytest.raises(ValueError, match='digest'):
        move(queue, job_id, digest='0' * 64)
    assert (tmp_path / 'dispatch-policy.json').read_bytes() == old
    digest = hashlib.sha256((tmp_path / 'pending' / job_id / 'request.json').read_bytes()).hexdigest()
    assert queue.run_one({'long': {'cmd': [sys.executable, '-c', 'pass']}})
    with pytest.raises(ValueError, match='pending'):
        move(queue, job_id, digest=digest)
    assert (tmp_path / 'dispatch-policy.json').read_bytes() == old
    with pytest.raises(ValueError):
        move(queue, '../escape', digest=digest)


def test_override_and_claim_share_one_atomic_boundary(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    digest = hashlib.sha256((tmp_path / 'pending' / job_id / 'request.json').read_bytes()).hexdigest()
    barrier = threading.Barrier(2)
    def change():
        barrier.wait()
        try:
            return move(GPUQueue(tmp_path), job_id, digest=digest)
        except ValueError as error:
            return str(error)
    def claim():
        barrier.wait()
        return GPUQueue(tmp_path).run_one({'long': {'cmd': [sys.executable, '-c', 'pass']}})
    with ThreadPoolExecutor(2) as pool:
        changed, claimed = pool.submit(change), pool.submit(claim)
        result, ran = changed.result(), claimed.result()
    assert ran
    terminal = json.loads((tmp_path / 'done' / job_id / 'receipt.json').read_text())
    if isinstance(result, dict):
        assert terminal['dispatch']['operator_override'] == result
        assert terminal['dispatch']['service_class'] == 'quick'
    else:
        assert 'pending' in result
        assert terminal['dispatch']['service_class'] == 'normal'


def test_duplicate_concurrent_drop_commits_one_action(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    identity = str(uuid4())
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: move(GPUQueue(tmp_path), job_id, request_id=identity), range(2)))
    assert results[0] == results[1]
    assert len(dispatch.policy(tmp_path)['action_receipts']) == 1


def test_submission_replay_after_move_preserves_pending_and_terminal_request(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    request = JobRequest(job_type='long', input_path='fixture')
    pending = queue.submit(request)
    assert queue.submit(request) == pending
    move(queue, request.job_id)
    before = (pending / 'request.json').read_bytes()
    assert queue.submit(request) == pending
    assert (pending / 'request.json').read_bytes() == before
    assert queue.run_one({'long': {'cmd': [sys.executable, '-c', 'pass']}})
    terminal = tmp_path / 'done' / request.job_id
    assert queue.submit(request) == terminal
    assert (terminal / 'request.json').read_bytes() == before
    (terminal / 'request.json').write_bytes(before + b'\n')
    with pytest.raises(ValueError, match='digest'):
        queue.submit(request)


def test_pause_and_live_lease_still_block_manual_quick(tmp_path):
    import os
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    move(queue, job_id)
    queue.pause(owner='fixture')
    config = {'long': {'cmd': [sys.executable, '-c', 'pass']}}
    assert not queue.run_one(config)
    queue.resume(owner='fixture', epoch=queue.pause_state()['epoch'])
    queue.claim_lease(owner='holder', agent_id='holder', repo_root='/fixture', pid=os.getpid(),
                      process_group=os.getpgrp(), effective_route='fixture/live', backend='fixture',
                      device='fixture:0', profile='fixture', supports_checkpoints=True,
                      interruptible=False, ttl_seconds=60)
    assert not queue.run_one(config)


def test_digest_change_after_override_blocks_execution_and_snapshot(tmp_path):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    move(queue, job_id)
    request_path = tmp_path / 'pending' / job_id / 'request.json'
    request_path.write_bytes(request_path.read_bytes() + b'\n')
    assert not queue.run_one({'long': {'cmd': [sys.executable, '-c', 'pass']}})
    assert 'digest' in queue_snapshot(queue)['dispatch']['error']
    assert not list((tmp_path / 'running').iterdir())


def test_actual_previous_worker_rejects_v2_without_advancing(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    previous = os.environ.get('GREENROOM_PREVIOUS_SOURCE')
    if not previous or not (Path(previous) / 'gpu_queue/queue.py').is_file():
        pytest.skip('requires an observed previous worker source for cross-version conformance')
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    move(queue, job_id)
    script = """import json,sys
from gpu_queue.queue import GPUQueue
queue=GPUQueue(sys.argv[1])
ran=queue.run_one({'long':{'cmd':[sys.executable,'-c','pass']}})
print(json.dumps({'ran':ran,'source':__import__('gpu_queue').__file__,'error':json.loads((queue.queue_dir/'dispatch-error.json').read_text())['error']}))
"""
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)], cwd=previous,
                            env={**os.environ, 'PYTHONPATH': previous}, capture_output=True, text=True, check=True)
    observed = json.loads(result.stdout)
    assert Path(observed['source']).resolve().is_relative_to(Path(previous).resolve())
    assert observed['ran'] is False
    assert 'schema' in observed['error']
    assert (tmp_path / 'pending' / job_id).is_dir()
    assert not list((tmp_path / 'running').iterdir())


@pytest.mark.parametrize('read_only', [False, True])
def test_console_endpoint_auth_and_admission_control(tmp_path, read_only):
    queue = GPUQueue(tmp_path)
    queue.configure_dispatch(policy(), owner='fixture')
    job_id = submit(queue, 'long', 1)
    row = queue_snapshot(queue)['jobs'][0]
    assert row['request_digest']
    body = {'job_id': job_id, 'service_class': 'quick', 'request_digest': row['request_digest'],
            'request_id': str(uuid4())}
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
        tmp_path, 'secret', admission_control=True, read_only=read_only))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_address[1]}/api/dispatch-class'
    try:
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(Request(url, data=json.dumps(body).encode(), method='POST'))
        assert unauthorized.value.code == 401
        authenticated = Request(url, data=json.dumps(body).encode(), method='POST',
                                headers={'Authorization': 'Bearer secret'})
        if read_only:
            with pytest.raises(HTTPError) as rejected:
                urlopen(authenticated)
            assert rejected.value.code == 403
        else:
            result = json.load(urlopen(authenticated))
            assert result['service_class'] == 'quick'
            assert queue_snapshot(queue)['jobs'][0]['dispatch_class'] == 'quick'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
