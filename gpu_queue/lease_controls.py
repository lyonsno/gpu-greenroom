"""Recorded dispatch blockers and explicit operator settlement of stale leases."""

import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import time

from .admission_control import sync_directory, write_durable
from .models import ExternalLease, LeaseStatus


def observe(queue_dir):
    path = Path(queue_dir) / 'leases' / 'current.json'
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, None
    except OSError as error:
        return None, _unreadable(error)
    try:
        payload = json.loads(raw)
        lease = ExternalLease.from_json(raw.decode())
        if not isinstance(payload, dict) or not all(isinstance(payload.get(key), str) and payload[key].strip() for key in (
            'lease_id', 'owner', 'agent_id', 'repo_root', 'effective_route', 'lifecycle_state',
        )) or not {'renewed_at', 'ttl_seconds', 'claimed_at', 'pid', 'process_group',
                   'process_start_identity'}.issubset(payload):
            raise ValueError('external lease identity is incomplete')
        if any(value is not None and (type(value) is not int or value < 1)
               for value in (lease.pid, lease.process_group)):
            raise ValueError('external process identity is invalid')
        if lease.process_start_identity is not None and (
                not isinstance(lease.process_start_identity, str) or not lease.process_start_identity.strip()):
            raise ValueError('external process start identity is invalid')
        if type(lease.claimed_at) not in (int, float) or not math.isfinite(lease.claimed_at):
            raise ValueError('external lease acquisition time is invalid')
        if lease.lifecycle_state == LeaseStatus.RELEASED:
            return payload, None
        if (type(lease.renewed_at) not in (int, float) or not math.isfinite(lease.renewed_at)
                or type(lease.ttl_seconds) not in (int, float) or not math.isfinite(lease.ttl_seconds)
                or lease.ttl_seconds <= 0):
            raise ValueError('external lease renewal identity is invalid')
        age = max(0, time.time() - lease.renewed_at)
        unknown_reason = (lease.unknown_reason or 'ownership_unknown'
                          if lease.lifecycle_state == LeaseStatus.OWNERSHIP_UNKNOWN
                          else 'ttl_expired' if age > lease.ttl_seconds
                          else 'holder_identity_unbound' if not lease.pid or not lease.process_start_identity
                          else None)
        unknown = unknown_reason is not None
        if unknown:
            payload = {**payload, 'recorded_lifecycle_state': lease.lifecycle_state.value,
                       'lifecycle_state': 'ownership_unknown', 'unknown_reason': unknown_reason}
        return payload, {
            'kind': 'external_lease', 'lease_id': lease.lease_id,
            'owner': lease.owner, 'agent_id': lease.agent_id, 'repo_root': lease.repo_root,
            'effective_route': lease.effective_route, 'pid': lease.pid,
            'process_group': lease.process_group, 'process_start_identity': lease.process_start_identity,
            'lifecycle_state': 'ownership_unknown' if unknown else lease.lifecycle_state.value,
            'recorded_lifecycle_state': lease.lifecycle_state.value,
            'reason': unknown_reason if unknown else lease.lifecycle_state.value,
            'renewal_age_seconds': age, 'ttl_seconds': lease.ttl_seconds,
            'record_sha256': hashlib.sha256(raw).hexdigest(),
            'recovery': {
                'available': unknown, 'confirmation_required': True,
                'authority': 'operator_confirmed_external_settlement',
                'reason': ('Process group not recorded; browser settlement cannot be inferred from its server'
                           if not lease.process_group else
                           'Recorded process identity alone does not establish external GPU settlement'),
            },
        }
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError, OverflowError) as error:
        return None, _unreadable(error)


def _unreadable(error):
    return {'kind': 'unreadable_lease', 'reason': str(error),
            'recovery': {'available': False, 'confirmation_required': True}}


def admission(*, lease_blocker, paused, running_ids, dispatch_error=None):
    blockers = []
    if lease_blocker:
        blockers.append(lease_blocker)
    if paused:
        blockers.append({'kind': 'queue_pause'})
    if running_ids:
        blockers.append({'kind': 'running_work', 'job_ids': running_ids})
    if dispatch_error:
        blockers.append({'kind': 'dispatch_policy', 'reason': dispatch_error})
    state = ('unknown' if lease_blocker and lease_blocker['kind'] == 'unreadable_lease'
             else 'external_lease' if lease_blocker else 'paused' if paused
             else 'running' if running_ids else 'dispatch_blocked' if dispatch_error
             else 'no_recorded_blocker')
    return {'state': state, 'blockers': blockers,
            'authority': 'recorded coordination state; execution flock remains authoritative'}


def recover(queue, *, lease_id, expected_record_sha256, expected_start_identity,
            request_id, quiescence_confirmed, reason):
    if quiescence_confirmed is not True:
        raise ValueError('positive operator confirmation of external GPU quiescence is required')
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', request_id):
        raise ValueError('recovery request_id must contain only letters, numbers, underscores and hyphens')
    if not isinstance(expected_record_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', expected_record_sha256):
        raise ValueError('recovery requires the exact displayed record digest')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('recovery needs an operator settlement reason')
    request = dict(lease_id=lease_id, expected_record_sha256=expected_record_sha256,
                   expected_start_identity=expected_start_identity, request_id=request_id,
                   quiescence_confirmed=quiescence_confirmed, reason=reason)
    directory = queue.queue_dir / 'operator-lease-recoveries'
    receipt_path = directory / (request_id + '.json')

    def previous():
        if not receipt_path.exists():
            return None
        value = json.loads(receipt_path.read_text())
        if value['request'] != request:
            raise ValueError('recovery request_id already belongs to a different action')
        return value

    # A completed retry is a receipt read, even when another job now owns flock.
    prior = previous()
    if prior and prior['phase'] == 'completed':
        return prior['result']
    lock = queue._try_execution_lock()
    if lock is None:
        raise RuntimeError('gpu.lock is held; cannot recover external lease')
    try:
        with queue._coordination_lock():
            prior = previous()
            if prior and prior['phase'] == 'completed':
                return prior['result']
            if any((queue.queue_dir / 'running').iterdir()):
                raise RuntimeError('running workload needs separate recovery')
            raw = queue.current_lease_path.read_bytes()
            if prior:
                if raw.decode() not in (prior['original_record_json'], prior['settled_record_json']):
                    raise ValueError('lease changed during an incomplete recovery; refresh its ownership')
                settled = ExternalLease.from_json(prior['settled_record_json'])
                handoff = queue._granted_lease_handoff_locked(settled)
                if (handoff.bump_id if handoff else None) != settled.handoff_bump_id:
                    raise ValueError('handoff changed during an incomplete recovery')
            else:
                if hashlib.sha256(raw).hexdigest() != expected_record_sha256:
                    raise ValueError('lease record changed; refresh before recovery')
                lease = ExternalLease.from_json(raw.decode())
                if lease.lease_id != lease_id or lease.process_start_identity != expected_start_identity:
                    raise ValueError('lease recovery identity does not match the displayed holder')
                _, blocker = observe(queue.queue_dir)
                if not blocker or not blocker['recovery']['available']:
                    raise ValueError('only stale or ownership-unknown leases need recovery')
                handoff = queue._granted_lease_handoff_locked(lease)
                lease.lifecycle_state = LeaseStatus.HANDOFF if handoff else LeaseStatus.RELEASED
                lease.handoff_bump_id = handoff.bump_id if handoff else None
                if handoff:
                    lease.renewed_at = time.time()
                lease.released_at = time.time()
                lease.released_by = 'operator-console'
                lease.release_reason = reason
                result = {
                    'lease_id': lease_id, 'lifecycle_state': lease.lifecycle_state.value,
                    'recovered_by': 'operator-console', 'reason': reason,
                    'quiescence_authority': 'operator_confirmed_external_settlement',
                    'handoff_bump_id': lease.handoff_bump_id, 'recovered_at': lease.released_at,
                    'receipt_path': str(receipt_path.resolve()),
                }
                prior = {
                    'schema': 'gpu-greenroom.operator-lease-recovery.v1', 'phase': 'prepared',
                    'request': request, 'original_record_json': raw.decode(),
                    'original_record_sha256': expected_record_sha256,
                    'settled_record_json': lease.to_json(), 'result': result,
                }
                directory.mkdir(exist_ok=True)
                sync_directory(queue.queue_dir)
                write_durable(receipt_path, prior)
                settled = lease
            queue._write_lease_locked(settled)
            write_durable(receipt_path, {**prior, 'phase': 'completed'})
            queue._emit_event_locked('lease_recovery', lease_id, {
                'recovered_by': 'operator-console', 'request_id': request_id,
                'quiescence_authority': 'operator_confirmed_external_settlement',
            })
            return prior['result']
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
