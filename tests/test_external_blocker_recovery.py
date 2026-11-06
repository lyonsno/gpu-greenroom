import fcntl
import hashlib
import http.client
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

import pytest

from gpu_queue.models import ExternalLease, LeaseStatus
from gpu_queue.operator_server import make_handler, queue_snapshot
from gpu_queue.queue import GPUQueue


def legacy_lease(queue):
    lease = ExternalLease(
        owner='page-owner', agent_id='page-agent', repo_root=str(queue.queue_dir),
        effective_route='fixture/browser-work', backend='webgpu', device='test',
        profile='interactive', supports_checkpoints=False, interruptible=False,
        lease_id='legacy-page', pid=99999999, process_group=None,
        process_start_identity='recorded-server-start',
        renewed_at=time.time()-300, ttl_seconds=120,
        lifecycle_state=LeaseStatus.OWNERSHIP_UNKNOWN,
        unknown_reason='ttl_expired', unknown_at=time.time()-180,
    )
    queue._write_lease_locked(lease)
    return lease


def recovery_request(queue, **changes):
    lease = queue._read_lease_locked()
    request = dict(
        lease_id=lease.lease_id,
        expected_record_sha256=hashlib.sha256(queue.current_lease_path.read_bytes()).hexdigest(),
        expected_start_identity=lease.process_start_identity,
        request_id='recover-one', quiescence_confirmed=True,
        reason='Operator confirms the external browser workload has stopped',
    )
    return {**request, **changes}


@pytest.fixture
def queue(tmp_path):
    q = GPUQueue(tmp_path/'queue')
    legacy_lease(q)
    return q


@pytest.fixture
def api(queue):
    servers = []
    def request(body, *, local_operator=True, read_only=False, authorized=True):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(
            queue.queue_dir, 'test-secret', local_operator=local_operator,
            read_only=read_only, admission_control=True,
        ))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        client = http.client.HTTPConnection('127.0.0.1', server.server_port)
        try:
            headers = {'Content-Type':'application/json'}
            if authorized:
                headers['Authorization'] = 'Bearer test-secret'
            client.request('POST', '/api/lease/recover', json.dumps(body), headers)
            response = client.getresponse()
            return response.status, json.loads(response.read())
        finally:
            client.close()
    yield request
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join()


def test_unpaused_expired_lease_is_an_effective_dispatch_blocker(queue):
    before = queue.current_lease_path.read_bytes()
    state = queue_snapshot(queue.queue_dir, 'active')
    assert state['paused'] is False
    assert state['execution_admission']['state'] == 'external_lease'
    blocker = state['execution_admission']['blockers'][0]
    assert blocker['lease_id'] == 'legacy-page'
    assert blocker['reason'] == 'ttl_expired'
    assert blocker['process_group'] is None
    assert blocker['recovery']['confirmation_required'] is True
    assert blocker['record_sha256'] == hashlib.sha256(before).hexdigest()
    assert queue.current_lease_path.read_bytes() == before


def test_recovery_requires_positive_settlement_not_just_dead_server(queue, api):
    before = queue.current_lease_path.read_bytes()
    status, result = api(recovery_request(queue, quiescence_confirmed=False))
    assert status == 409
    assert 'quiescence' in result['error']
    assert queue.current_lease_path.read_bytes() == before


def test_confirmed_legacy_recovery_is_receipted_and_idempotent(queue, api):
    import sys
    from gpu_queue.models import JobRequest, JobStatus
    pending = JobRequest(job_type='command', input_path='', command_argv=[sys.executable,'-c','pass'])
    queue.submit(pending)
    assert queue.run_one({}) is False
    before = queue.current_lease_path.read_bytes()
    body = recovery_request(queue)
    status, result = api(body)
    assert status == 200, result
    assert queue._read_lease_locked().lifecycle_state == LeaseStatus.RELEASED
    assert result['quiescence_authority'] == 'operator_confirmed_external_settlement'
    receipt = json.loads(Path(result['receipt_path']).read_text())
    assert receipt['phase'] == 'completed'
    assert receipt['original_record_json'].encode() == before
    assert receipt['request'] == body
    assert receipt['result'] == result
    assert queue.run_one({}) is True
    assert queue.get_job(pending.job_id).status == JobStatus.DONE
    assert api(body) == (200, result)
    conflict, _ = api({**body, 'reason':'different reason'})
    assert conflict == 409


@pytest.mark.parametrize('change', [
    {'expected_record_sha256':'0'*64},
    {'expected_start_identity':'another-process-start'},
    {'lease_id':'another-lease'},
    {'request_id':'../unsafe'},
    {'reason':''},
    {'quiescence_confirmed':1},
])
def test_recovery_rejects_stale_or_incomplete_identity(queue, api, change):
    before = queue.current_lease_path.read_bytes()
    assert api(recovery_request(queue, **change))[0] == 409
    assert queue.current_lease_path.read_bytes() == before


@pytest.mark.parametrize('options,status', [
    ({'authorized':False},401),
    ({'local_operator':False},403),
    ({'read_only':True},403),
])
def test_recovery_preserves_http_authority(queue, api, options, status):
    before = queue.current_lease_path.read_bytes()
    assert api(recovery_request(queue), **options)[0] == status
    assert queue.current_lease_path.read_bytes() == before


def test_recovery_cannot_cross_execution_mutex(queue, api):
    before = queue.current_lease_path.read_bytes()
    with queue.lock_path.open('w') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert api(recovery_request(queue))[0] == 409
    assert queue.current_lease_path.read_bytes() == before


def test_completed_recovery_retry_does_not_release_a_new_holder(queue, api):
    body = recovery_request(queue)
    status, result = api(body)
    assert status == 200
    fresh = legacy_lease(queue)
    fresh.lease_id = 'different-new-holder'
    queue._write_lease_locked(fresh)
    before = queue.current_lease_path.read_bytes()
    assert api(body) == (200, result)
    assert queue.current_lease_path.read_bytes() == before


def test_malformed_lease_is_visible_as_unknown_not_free(queue):
    queue.current_lease_path.write_text('{bad JSON')
    state = queue_snapshot(queue.queue_dir, 'active')
    assert state['execution_admission']['state'] == 'unknown'
    assert state['execution_admission']['blockers'][0]['kind'] == 'unreadable_lease'
    assert state['execution_admission']['blockers'][0]['recovery']['available'] is False


def test_ttl_expiry_does_not_authorize_automatic_release(queue):
    assert queue.lease_status().lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN
    assert queue._external_execution_blocked() is True
    with pytest.raises(RuntimeError, match='quiescence'):
        queue.recover_lease('legacy-page', expected_start_identity='recorded-server-start',
                            recovered_by='test', reason='server absent')
    assert queue._external_execution_blocked() is True


@pytest.mark.parametrize('failed_write', ['lease', 'completion'])
def test_interrupted_recovery_resumes_only_its_exact_prepared_transition(queue, api, monkeypatch, failed_write):
    from gpu_queue import lease_controls
    body = recovery_request(queue)
    before = queue.current_lease_path.read_bytes()
    with monkeypatch.context() as patch:
        if failed_write == 'lease':
            patch.setattr(queue.__class__, '_write_lease_locked',
                          lambda *args: (_ for _ in ()).throw(OSError('injected lease write failure')))
        else:
            original = lease_controls.write_durable
            def write(path, payload):
                if payload.get('phase') == 'completed':
                    raise OSError('injected completion failure')
                return original(path, payload)
            patch.setattr(lease_controls, 'write_durable', write)
        status, result = api(body)
        assert status == 409 and 'injected' in result['error']
    journal = queue.queue_dir/'operator-lease-recoveries'/'recover-one.json'
    assert json.loads(journal.read_text())['phase'] == 'prepared'
    if failed_write == 'lease':
        assert queue.current_lease_path.read_bytes() == before
    assert api(body)[0] == 200
    assert json.loads(journal.read_text())['phase'] == 'completed'
    assert queue._read_lease_locked().lifecycle_state == LeaseStatus.RELEASED


def test_interrupted_recovery_never_applies_to_a_replacement_record(queue, api, monkeypatch):
    body = recovery_request(queue)
    with monkeypatch.context() as patch:
        patch.setattr(queue.__class__, '_write_lease_locked',
                      lambda *args: (_ for _ in ()).throw(OSError('injected write failure')))
        assert api(body)[0] == 409
    fresh = legacy_lease(queue)
    fresh.lease_id = 'different-new-holder'
    queue._write_lease_locked(fresh)
    before = queue.current_lease_path.read_bytes()
    assert api(body)[0] == 409
    assert queue.current_lease_path.read_bytes() == before


def test_confirmed_recovery_preserves_existing_granted_handoff(queue, api):
    from gpu_queue.models import BumpRequest, BumpStatus
    bump = BumpRequest(requester='next-owner', agent_id='next-agent', repo_root='/fixture',
        intended_route='next-route', workload_class='capture', memory_pressure='small',
        estimated_occupancy='short', full_quiescence_required=True, reason='fixture',
        callback_address='fixture', holder_lease_id='legacy-page', status=BumpStatus.GRANTED,
        quiescence_confirmed=True)
    queue._write_bump_locked(bump)
    status, result = api(recovery_request(queue))
    assert status == 200
    lease = queue._read_lease_locked()
    assert lease.lifecycle_state == LeaseStatus.HANDOFF
    assert lease.handoff_bump_id == bump.bump_id == result['handoff_bump_id']
    assert queue._external_execution_blocked() is True


def test_concurrent_recoveries_only_commit_one_exact_transition(queue):
    from concurrent.futures import ThreadPoolExecutor
    from gpu_queue.lease_controls import recover
    base = recovery_request(queue)
    def attempt(i):
        try:
            return recover(queue, **{**base, 'request_id':f'concurrent-{i}'})
        except (ValueError, RuntimeError):
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(list((queue.queue_dir/'operator-lease-recoveries').glob('*.json'))) == 1


def test_missing_authority_fields_cannot_be_filled_by_dataclass_defaults(queue):
    value = json.loads(queue.current_lease_path.read_text())
    del value['renewed_at']
    queue.current_lease_path.write_text(json.dumps(value))
    state = queue_snapshot(queue.queue_dir, 'active')
    assert state['execution_admission']['state'] == 'unknown'
    assert state['execution_admission']['blockers'][0]['recovery']['available'] is False


@pytest.mark.parametrize('field,value', [
    ('owner', {'name':'unexpected shape'}), ('repo_root', True),
    ('effective_route', ['another shape']), ('pid', 'server'),
    ('process_start_identity', 12), ('claimed_at', float('nan')),
])
def test_retyped_lease_authority_is_unknown_not_recoverable(queue, field, value):
    record = json.loads(queue.current_lease_path.read_text())
    record[field] = value
    queue.current_lease_path.write_text(json.dumps(record))
    state = queue_snapshot(queue.queue_dir, 'active')
    assert state['execution_admission']['state'] == 'unknown'
    assert state['execution_admission']['blockers'][0]['recovery']['available'] is False


def test_doctor_separates_service_checks_from_external_dispatch_obstruction(queue, capsys):
    from types import SimpleNamespace
    from gpu_queue.cli import cmd_doctor
    cmd_doctor(SimpleNamespace(queue_dir=str(queue.queue_dir)))
    result = json.loads(capsys.readouterr().out)
    assert result['healthy'] is True
    assert result['queue']['paused'] is False
    assert result['execution_admission']['state'] == 'external_lease'
    assert result['execution_admission']['blockers'][0]['lease_id'] == 'legacy-page'


@pytest.mark.parametrize('representation', ['path', 'queue'])
def test_snapshot_preserves_malformed_expired_active_authority(queue, representation):
    record = json.loads(queue.current_lease_path.read_text())
    record['lifecycle_state'] = LeaseStatus.ACTIVE.value
    del record['pid']
    queue.current_lease_path.write_text(json.dumps(record))
    original = queue.current_lease_path.read_bytes()
    state = queue_snapshot(queue.queue_dir if representation == 'path' else queue, 'active')
    assert state['execution_admission']['state'] == 'unknown'
    assert state['execution_admission']['blockers'][0]['recovery']['available'] is False
    assert queue.current_lease_path.read_bytes() == original


@pytest.mark.parametrize('field', ['claimed_at', 'renewed_at', 'ttl_seconds'])
def test_overflowing_json_integer_is_visible_through_snapshot_and_doctor(queue, capsys, field):
    from types import SimpleNamespace
    from gpu_queue.cli import cmd_doctor
    record = json.loads(queue.current_lease_path.read_text())
    record[field] = 10**400
    queue.current_lease_path.write_text(json.dumps(record))
    for surface in (queue.queue_dir, queue):
        state = queue_snapshot(surface, 'active')['execution_admission']
        assert state['state'] == 'unknown'
        assert state['blockers'][0]['recovery']['available'] is False
    cmd_doctor(SimpleNamespace(queue_dir=str(queue.queue_dir)))
    assert json.loads(capsys.readouterr().out)['execution_admission']['state'] == 'unknown'


def test_first_recovery_persists_journal_parent_before_lease_transition(queue, monkeypatch):
    import os
    from gpu_queue import admission_control, lease_controls
    syncs = []
    original_sync = os.fsync
    original_write = queue._write_lease_locked
    def sync(fd):
        info = os.fstat(fd)
        for path in (queue.queue_dir, queue.queue_dir/'operator-lease-recoveries'):
            if path.exists() and (info.st_dev, info.st_ino) == (path.stat().st_dev, path.stat().st_ino):
                syncs.append(path)
        return original_sync(fd)
    def write(lease):
        directory = queue.queue_dir/'operator-lease-recoveries'
        assert queue.queue_dir in syncs
        assert directory in syncs
        assert syncs.index(queue.queue_dir) < syncs.index(directory)
        assert json.loads((directory/'recover-one.json').read_text())['phase'] == 'prepared'
        return original_write(lease)
    monkeypatch.setattr(admission_control.os, 'fsync', sync)
    monkeypatch.setattr(queue, '_write_lease_locked', write)
    lease_controls.recover(queue, **recovery_request(queue))


def test_doctor_reports_same_dispatch_policy_error_as_monitor(queue, capsys):
    from types import SimpleNamespace
    from gpu_queue.cli import cmd_doctor
    queue.current_lease_path.unlink()
    (queue.queue_dir/'dispatch-policy.json').write_text('{}')
    state = queue_snapshot(queue.queue_dir, 'active')['execution_admission']
    cmd_doctor(SimpleNamespace(queue_dir=str(queue.queue_dir)))
    observed = json.loads(capsys.readouterr().out)['execution_admission']
    assert observed == state
    assert observed['state'] == 'dispatch_blocked'
    assert observed['blockers'][0]['kind'] == 'dispatch_policy'
