"""Cheap producer estimates, distinct from observed runtime history."""

import math

CAPABILITY = 'runtime-hint.v1'


def validate(hint):
    if hint is None:
        return None
    if (not isinstance(hint, dict) or type(hint.get('expected_seconds')) not in (int, float)
            or not math.isfinite(hint['expected_seconds']) or hint['expected_seconds'] <= 0
            or ('reason' in hint and not isinstance(hint['reason'], str))):
        raise ValueError('runtime_hint requires positive finite expected_seconds and optional string reason')
    return hint


def from_args(args):
    seconds = getattr(args, 'expected_seconds', None)
    reason = getattr(args, 'runtime_reason', None)
    if seconds is None:
        if reason is not None:
            raise ValueError('runtime_hint reason requires --expected-seconds')
        return None
    return validate({'expected_seconds': seconds, **({'reason': reason} if reason is not None else {})})


def comparison(hint, receipt):
    if hint is None:
        return None
    validate(hint)
    start, end = receipt.get('started_at'), receipt.get('finished_at')
    wall = (end - start if type(start) in (int, float) and type(end) in (int, float)
            and math.isfinite(start) and math.isfinite(end) and end >= start else None)
    actual = wall if receipt.get('child_pid') is not None else None
    clock = 'execution-wall'
    performance = receipt.get('performance_window')
    if performance is not None:
        clock = 'performance-command-wall'
        actual = performance.get('command_wall_seconds') if performance.get('workload_started') is True else None
    if type(actual) not in (int, float) or not math.isfinite(actual) or actual < 0:
        actual = None
    expected = hint['expected_seconds']
    return {'expected_seconds': expected, 'reason': hint.get('reason'),
            'authority': 'producer-estimated', 'actual_seconds': actual,
            'execution_wall_seconds': wall, 'clock': clock, 'condition': 'unknown',
            'successful': actual is not None and receipt.get('status') == 'done' and receipt.get('exit_code') == 0,
            'ratio': actual / expected if actual is not None else None,
            'over_estimate': actual > expected if actual is not None else None}
