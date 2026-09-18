"""Receipted console start-gate changes, using the worker's coordination lock."""

import json
import os
from pathlib import Path
import re
import time
import uuid

from .queue import GPUQueue, PAUSE_STATE_SCHEMA


def sync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_durable(path: Path, payload: dict):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def transition(queue: GPUQueue, *, action: str, request_id: str, owner: str, epoch=None):
    if action not in {'pause', 'resume'}:
        raise ValueError('only pause and resume are supported')
    if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', request_id):
        raise ValueError('request_id must contain only letters, numbers, underscores and hyphens')
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError('owner is required')
    if action == 'resume' and (not isinstance(epoch, str) or not epoch):
        raise ValueError('resume requires the exact observed pause epoch')
    request = {'action': action, 'request_id': request_id, 'owner': owner, 'epoch': epoch}
    with queue._coordination_lock():
        directory = queue.queue_dir / 'operator-actions'
        directory.mkdir(exist_ok=True)
        sync_directory(queue.queue_dir)
        receipt_path = directory / f'{request_id}.json'
        if receipt_path.exists():
            previous = json.loads(receipt_path.read_text())
            if previous.get('request') != request:
                raise ValueError('request_id already belongs to a different action')
            if previous.get('phase') != 'completed':
                raise RuntimeError(f'incomplete control transition: inspect {receipt_path}')
            return previous['result']
        before = queue._read_pause_state_locked()
        if action == 'resume' and before is not None and before.get('epoch') != epoch:
            raise ValueError('pause epoch changed; refresh before resuming')
        effective_epoch = before.get('epoch') if before else (uuid.uuid4().hex if action == 'pause' else epoch)
        result = {
            'action': action, 'owner': owner, 'epoch': effective_epoch,
            'effective_paused': action == 'pause',
            'idempotent': (before is not None) if action == 'pause' else (before is None),
            'previous_pause': before, 'requested_at': time.time(),
            'receipt_path': str(receipt_path.resolve()),
        }
        receipt = {
            'schema': 'gpu-greenroom.operator-action.v1', 'phase': 'prepared',
            'queue_dir': str(queue.queue_dir.resolve()), 'request': request,
            'previous_pause': before, 'result': result,
        }
        # The durable intent survives a crash between marker mutation and completion.
        # An incomplete action is never blindly replayed as successful.
        write_durable(receipt_path, receipt)
        if action == 'pause' and before is None:
            write_durable(queue.pause_path, {
                'schema': PAUSE_STATE_SCHEMA, 'status': 'effective',
                'owner': owner, 'epoch': effective_epoch,
                'contention_class': 'operator-console',
                'queue_dir': str(queue.queue_dir.resolve()),
                'requested_at': result['requested_at'],
                'effective_at': time.time(),
            })
        elif action == 'resume' and before is not None:
            queue.pause_path.unlink()
            sync_directory(queue.queue_dir)
        try:
            result['acknowledged_at'] = time.time()
            write_durable(receipt_path, {**receipt, 'phase': 'completed'})
        except OSError as error:
            raise OSError(f'action may have applied; receipt incomplete at {receipt_path}: {error}') from error
        return result
