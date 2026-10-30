"""Advisory producer forecasts; never dispatch or quiescence authority."""

import hashlib
import json
import math
import time

from .models import JobStatus

SCHEMA = 'gpu-greenroom.pause-estimate.v1'


def valid_seconds(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def project(path, request, state):
    unknown = {'status': 'unknown', 'remaining_seconds': None,
               'authority': 'producer-estimate-only'}
    if state.status != JobStatus.RUNNING:
        return {**unknown, 'status': 'not-running'}
    if not request.cooperative_checkpoint:
        return {**unknown, 'reason': 'pause unsupported'}
    try:
        value = json.loads((path/'pause-estimate.json').read_text())
        now = time.time()
        if (not isinstance(value, dict) or value.get('schema') != SCHEMA
                or value.get('job_id') != request.job_id
                or value.get('request_sha256') != hashlib.sha256((path/'request.json').read_bytes()).hexdigest()
                or not state.child_start_identity
                or value.get('child_start_identity') != state.child_start_identity
                or value.get('child_pid') != state.child_pid
                or value.get('started_at') != state.started_at
                or not valid_seconds(value.get('reported_at'))
                or state.started_at is None or not state.started_at <= value['reported_at'] <= now
                or not isinstance(value.get('checkpoint'), str)
                or (value.get('seconds') is not None and not valid_seconds(value['seconds']))
                or str(state.failure_phase).endswith('quiescence_unresolved')):
            return {**unknown, 'reason': 'invalid or different-attempt estimate'}
        if value.get('seconds') is None:
            return {**unknown, 'reason': 'producer withdrew estimate'}
        remaining = value['seconds'] - (now - value['reported_at'])
        return {**unknown, 'status': 'estimated' if remaining > 0 else 'overdue',
                'remaining_seconds': remaining if remaining > 0 else None,
                'reported_at': value['reported_at'], 'checkpoint': value['checkpoint']}
    except FileNotFoundError:
        return {**unknown, 'reason': 'producer has not reported an estimate'}
    except (OSError, ValueError, TypeError, OverflowError):
        return {**unknown, 'reason': 'estimate unavailable or malformed'}
