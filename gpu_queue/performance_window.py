"""Native operator confirmation before performance-sensitive commands."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

CAPABILITY = 'performance-window.v1'
SCHEMA = 'gpu-greenroom.performance-window.v1'
DEFER_EXIT = 75


def delay_seconds(minutes):
    if isinstance(minutes, bool):
        raise ValueError('Reminder delay must be a positive number of minutes')
    seconds = float(minutes) * 60
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(time.time() + seconds):
        raise ValueError('Reminder delay must be a positive finite number of minutes')
    return seconds


def atomic_write(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _native_prompt(spec, error_message=None):
    if sys.platform != 'darwin':
        raise RuntimeError('Native performance confirmation requires the macOS console')
    cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFDictionaryCreateMutable.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    cf.CFDictionaryCreateMutable.restype = ctypes.c_void_p
    cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    cf.CFArrayCreate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_long, ctypes.c_void_p]
    cf.CFArrayCreate.restype = ctypes.c_void_p
    cf.CFUserNotificationCreate.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_int32), ctypes.c_void_p]
    cf.CFUserNotificationCreate.restype = ctypes.c_void_p
    cf.CFUserNotificationReceiveResponse.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.POINTER(ctypes.c_ulong)]
    cf.CFUserNotificationReceiveResponse.restype = ctypes.c_int32
    cf.CFUserNotificationGetResponseValue.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
    cf.CFUserNotificationGetResponseValue.restype = ctypes.c_void_p
    cf.CFStringGetLength.argtypes = [ctypes.c_void_p]
    cf.CFStringGetLength.restype = ctypes.c_long
    cf.CFStringGetMaximumSizeForEncoding.argtypes = [ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetMaximumSizeForEncoding.restype = ctypes.c_long
    cf.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_uint32]
    cf.CFStringGetCString.restype = ctypes.c_bool
    message = (
        'Please quit Chrome, or stop live rendering and other GPU-heavy activity before continuing. '
        'The command has not started. Keep competing activity stopped until this run finishes.\n\n'
        f"Job: {spec['job_id']}\nOwner: {spec.get('owner') or 'not recorded'}\n"
        f"Route: {spec['route']}\n{spec['reason']}\n\n"
        'Need Chrome a little longer? Remind me later lets other queued work proceed. '
        'The measurement will ask again after your chosen delay; it will not start automatically. '
        'Confirmation does not verify GPU quietness.')
    if error_message:
        message += '\n\n' + error_message
    values = []
    def keep(value):
        if not value:
            raise RuntimeError('Could not construct native performance confirmation')
        values.append(value)
        return value
    def key(name):
        return ctypes.c_void_p.in_dll(cf, 'kCFUserNotification' + name + 'Key').value
    def string(text):
        return keep(cf.CFStringCreateWithCString(None, text.encode('utf-8'), 0x08000100))
    try:
        dictionary = keep(cf.CFDictionaryCreateMutable(None, 0, None, None))
        for name, text in (
                ('AlertHeader', 'Greenroom: performance measurement'), ('AlertMessage', message),
                ('DefaultButtonTitle', 'Cancel run'), ('AlternateButtonTitle', 'Start measurement'),
                ('OtherButtonTitle', 'Remind me later'), ('TextFieldTitles', 'Remind me in (minutes)'),
                ('TextFieldValues', '5')):
            value = string(text)
            if name.startswith('TextField'):
                value = keep(cf.CFArrayCreate(None, (ctypes.c_void_p * 1)(value), 1, None))
            cf.CFDictionarySetValue(dictionary, key(name), value)
        error = ctypes.c_int32()
        notification = keep(cf.CFUserNotificationCreate(None, 0, 2, ctypes.byref(error), dictionary))
        if error.value:
            raise OSError(error.value, 'Native performance confirmation failed')
        response = ctypes.c_ulong(3)
        code = cf.CFUserNotificationReceiveResponse(notification, 0, ctypes.byref(response))
        if code:
            raise OSError(code, 'Native performance response failed')
        if response.value == 2:
            value = cf.CFUserNotificationGetResponseValue(notification, key('TextFieldValues'), 0)
            if not value:
                raise ValueError('Reminder delay was not returned by macOS')
            size = cf.CFStringGetMaximumSizeForEncoding(cf.CFStringGetLength(value), 0x08000100) + 1
            buffer = ctypes.create_string_buffer(size)
            if not cf.CFStringGetCString(value, buffer, size, 0x08000100):
                raise ValueError('Could not read reminder delay')
            return {'defer_minutes': buffer.value.decode('utf-8')}
        return response.value == 1
    finally:
        for value in reversed(values):
            cf.CFRelease(value)


def native_dialog(spec):
    error_message = None
    while True:
        answer = _native_prompt(spec, error_message)
        if isinstance(answer, dict):
            try:
                delay_seconds(answer['defer_minutes'])
            except (ValueError, TypeError, OverflowError):
                error_message = 'Please enter a positive number of minutes, for example 5 or 2.5.'
                continue
        return answer


def native_notice(title, message, level):
    if sys.platform != 'darwin':
        raise RuntimeError('Native performance notification requires the macOS console')
    cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFUserNotificationDisplayNotice.argtypes = [
        ctypes.c_double, ctypes.c_ulong, *([ctypes.c_void_p] * 6)]
    cf.CFUserNotificationDisplayNotice.restype = ctypes.c_int32
    values = []
    try:
        for text in (title, message, 'OK'):
            value = cf.CFStringCreateWithCString(None, text.encode('utf-8'), 0x08000100)
            if not value:
                raise RuntimeError('Could not construct native performance notification')
            values.append(value)
        # The one-way API returns without waiting for dismissal.
        error = cf.CFUserNotificationDisplayNotice(0, level, None, None, None, *values)
        if error:
            raise OSError(error, 'Native performance notification failed')
    finally:
        for value in values:
            cf.CFRelease(value)


def notify_completion(job_dir, request, receipt):
    if receipt.get('job_id') != request.job_id:
        raise ValueError('Performance notification job identity conflicts')
    finished = receipt.get('finished_at')
    phase = receipt.get('failure_phase')
    confirmed = (receipt.get('status') in {'done', 'failed', 'cancelled'}
                 and type(finished) in (int, float) and math.isfinite(finished) and finished > 0
                 and (phase is None or (isinstance(phase, str) and not phase.endswith('_quiescence_unresolved'))))
    if not confirmed:
        outcome, level = 'cleanup uncertain', 0
        detail = ('Some processes may still be running. Cleanup is not confirmed; '
                  'do not treat this as an end-of-window or GPU-free notice.')
    else:
        outcome = ('finished' if receipt['status'] == 'done' else
                   'cancelled' if (receipt['status'] == 'cancelled'
                                  or (receipt.get('performance_window') or {}).get('status') == 'cancelled') else
                   'interrupted' if phase in {'worker_shutdown', 'stale_recovery', 'performance_window_interrupted'}
                   else 'failed')
        level = 1 if outcome == 'finished' else 2
        detail = ("This job's processes have stopped. Its performance window has ended. "
                  'Other measurements may request their own window; this does not certify GPU quietness.')
    path = Path(job_dir) / ('performance-completion-notice.json' if confirmed else 'performance-ownership-warning.json')
    report = {'schema': 'gpu-greenroom.performance-notification.v1', 'job_id': request.job_id,
              'owner': request.agent_id, 'route': request.route_identity or request.job_type,
              'reason': request.performance_window, 'outcome': outcome, 'failure_phase': phase,
              'quiescence': 'confirmed' if confirmed else 'unknown', 'status': 'submission-unknown',
              'requested_at': time.time(), 'backend': 'CoreFoundation.CFUserNotificationDisplayNotice',
              'human_observation': 'unverified', 'quietness_proved': False}
    try:
        with path.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        return {'status': 'already-attempted', 'receipt_path': str(path)}
    message = (f"Job: {request.job_id}\nOwner: {request.agent_id or 'not recorded'}\n"
               f"Route: {report['route']}\nOutcome: {outcome}\n"
               f"{request.performance_window}\n\n{detail}")
    if phase:
        message += f'\nFailure phase: {phase}'
    try:
        native_notice('Greenroom: measurement ' + outcome, message, level)
        report['status'] = 'submitted'
    except Exception as error:
        report.update(status='failed', error=str(error))
    report['finished_at'] = time.time()
    atomic_write(path, report)
    return report


def runner_command(job_dir, request, argv, timeout):
    spec = {'schema': SCHEMA, 'job_id': request.job_id, 'owner': request.agent_id,
            'route': request.route_identity or request.job_type, 'reason': request.performance_window,
            'argv': argv, 'timeout': timeout, 'attempt_id': uuid.uuid4().hex,
            'receipt_path': str((job_dir / 'performance-window.json').resolve())}
    validation_path = job_dir / 'validation.json'
    if 'command-preflight.v1' in request.required_worker_capabilities or validation_path.exists():
        spec['validation'] = json.loads(validation_path.read_text())
        spec['request'] = json.loads(request.to_json())
    path = job_dir / 'performance-launch.json'
    atomic_write(path, spec)
    return [sys.executable, str(Path(__file__).resolve()), '--launch', str(path.resolve())]


def run(spec, *, prompt=None):
    path = Path(spec['receipt_path'])
    report = {'schema': SCHEMA, 'job_id': spec['job_id'], 'owner': spec['owner'],
              'route': spec['route'], 'reason': spec['reason'], 'status': 'waiting',
              'phase': 'operator-confirmation', 'requested_at': time.time(),
              'backend': 'CoreFoundation.CFUserNotificationCreate',
              'attempt_id': spec.get('attempt_id') or uuid.uuid4().hex,
              'quietness_proved': False, 'authority': 'operator-acknowledgement-only',
              'workload_started': False, 'workload_argv': spec['argv']}
    atomic_write(path, report)
    try:
        answer = (prompt or native_dialog)(spec)
        if isinstance(answer, dict) and 'defer_minutes' in answer:
            seconds = delay_seconds(answer['defer_minutes'])
            finished = time.time()
            report.update(status='deferred', finished_at=finished, defer_seconds=seconds,
                          deferred_until=finished + seconds, operator_wait_seconds=finished - report['requested_at'])
            atomic_write(path, report)
            return DEFER_EXIT
        if answer is not True:
            report.update(status='cancelled', finished_at=time.time())
            report['operator_wait_seconds'] = report['finished_at'] - report['requested_at']
            atomic_write(path, report)
            return 125
        report.update(status='confirmed', confirmed_at=time.time())
        report['operator_wait_seconds'] = report['confirmed_at'] - report['requested_at']
        atomic_write(path, report)
        report.update(phase='validation-invalidated')
        atomic_write(path, report)
        if 'validation' in spec:
            # The native dialog may stay open while an owner edits the source.
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from gpu_queue import admission
            from gpu_queue.models import JobRequest
            admission.assert_current(JobRequest.from_json(json.dumps(spec['request'])), spec['validation'])
        report.update(phase='launch')
        atomic_write(path, report)
        # Stay in the worker-owned process group so its Stop/shutdown still owns every child.
        ready = os.environ.get('GPU_GREENROOM_READY_FD')
        options = {'pass_fds': (int(ready),)} if ready is not None else {}
        started = time.monotonic()
        child = subprocess.Popen(spec['argv'], **options)
        report.update(status='running', phase='execution', workload_started=True,
                      workload_pid=child.pid, workload_started_at=time.time())
        atomic_write(path, report)
        try:
            code = child.wait(timeout=spec['timeout'])
        except subprocess.TimeoutExpired:
            # The worker quiesces this entire group before releasing its execution lock.
            report.update(status='failed', phase='timeout', error='Command exceeded its execution timeout',
                          finished_at=time.time(), command_wall_seconds=time.monotonic() - started, exit_code=124)
            atomic_write(path, report)
            return 124
        report.update(status='complete' if code == 0 else 'failed', exit_code=code,
                      finished_at=time.time(), command_wall_seconds=time.monotonic() - started)
        if code != 0:
            report['error'] = f'Command exited with code {code}'
        atomic_write(path, report)
        return code if code >= 0 else 128 - code
    except Exception as error:
        report.update(status='failed', error=str(error), finished_at=time.time())
        if 'confirmed_at' not in report:
            report['operator_wait_seconds'] = report['finished_at'] - report['requested_at']
        atomic_write(path, report)
        return 125


def read_receipt(path, request):
    value = json.loads(Path(path).read_text())
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA
            or value.get('job_id') != request.job_id or value.get('reason') != request.performance_window
            or value.get('owner') != request.agent_id
            or value.get('route') != (request.route_identity or request.job_type)
            or value.get('quietness_proved') is not False
            or value.get('authority') != 'operator-acknowledgement-only'
            or value.get('status') not in {'waiting', 'confirmed', 'running', 'complete', 'cancelled', 'failed', 'deferred'}):
        raise ValueError('Performance-window receipt identity or status conflicts')
    if value['status'] == 'deferred' and (
            value.get('workload_started') is not False or value.get('phase') != 'operator-confirmation'
            or not isinstance(value.get('attempt_id'), str) or len(value['attempt_id']) != 32
            or any(c not in '0123456789abcdef' for c in value['attempt_id'])
            or not all(type(value.get(key)) in (float, int) and math.isfinite(value[key]) and value[key] >= 0
                       for key in ('defer_seconds', 'finished_at', 'deferred_until', 'operator_wait_seconds'))
            or value['defer_seconds'] <= 0
            or value['deferred_until'] != value['finished_at'] + value['defer_seconds']
            or 'workload_pid' in value or 'confirmed_at' in value):
        raise ValueError('Performance deferral lacks a valid delay and unstarted-workload evidence')
    if value['status'] == 'complete' and (
            value.get('workload_started') is not True or value.get('exit_code') != 0
            or value.get('phase') != 'execution'
            or not all(type(value.get(key)) in (float, int) and math.isfinite(value[key]) and value[key] >= 0
                       for key in ('operator_wait_seconds', 'command_wall_seconds', 'confirmed_at', 'workload_started_at', 'finished_at'))):
        raise ValueError('Performance-window completion lacks confirmed execution evidence')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', required=True)
    args = parser.parse_args()
    sys.exit(run(json.loads(Path(args.launch).read_text())))
