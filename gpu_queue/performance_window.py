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

CAPABILITY = 'performance-window.v1'
SCHEMA = 'gpu-greenroom.performance-window.v1'


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


def native_dialog(spec):
    if sys.platform != 'darwin':
        raise RuntimeError('Native performance confirmation requires the macOS console')
    cf = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFUserNotificationDisplayAlert.argtypes = [
        ctypes.c_double, ctypes.c_ulong, *([ctypes.c_void_p] * 8), ctypes.POINTER(ctypes.c_ulong)]
    cf.CFUserNotificationDisplayAlert.restype = ctypes.c_int32
    message = (
        'Please quit Chrome, or stop live rendering and other GPU-heavy activity before continuing. '
        'The command has not started. Keep competing activity stopped until this run finishes.\n\n'
        f"Job: {spec['job_id']}\nOwner: {spec.get('owner') or 'not recorded'}\n"
        f"Route: {spec['route']}\n{spec['reason']}\n\n"
        'This job holds its Greenroom slot while waiting. Cancel run releases the slot. '
        'Confirmation does not verify GPU quietness.')
    values = []
    try:
        for text in ('Greenroom: performance measurement', message, 'Cancel run', 'Start measurement'):
            value = cf.CFStringCreateWithCString(None, text.encode('utf-8'), 0x08000100)
            if not value:
                raise RuntimeError('Could not construct native performance confirmation')
            values.append(value)
        response = ctypes.c_ulong(3)
        error = cf.CFUserNotificationDisplayAlert(0, 2, None, None, None, *values, None, ctypes.byref(response))
        if error:
            raise OSError(error, 'Native performance confirmation failed')
        return response.value == 1
    finally:
        for value in values:
            cf.CFRelease(value)


def runner_command(job_dir, request, argv, timeout):
    spec = {'schema': SCHEMA, 'job_id': request.job_id, 'owner': request.agent_id,
            'route': request.route_identity or request.job_type, 'reason': request.performance_window,
            'argv': argv, 'timeout': timeout,
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
              'backend': 'CoreFoundation.CFUserNotificationDisplayAlert',
              'quietness_proved': False, 'authority': 'operator-acknowledgement-only',
              'workload_started': False, 'workload_argv': spec['argv']}
    atomic_write(path, report)
    try:
        if (prompt or native_dialog)(spec) is not True:
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
            or value.get('status') not in {'waiting', 'confirmed', 'running', 'complete', 'cancelled', 'failed'}):
        raise ValueError('Performance-window receipt identity or status conflicts')
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
