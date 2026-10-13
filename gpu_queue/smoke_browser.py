"""Request one retained browser page from its live owning runner, never open a URL."""

import json
import os
from pathlib import Path
import socket
import stat
import subprocess
from uuid import UUID, uuid4

from .queue import GPUQueue


SCHEMA = 'gpu-greenroom.application-reveal.v1'
IDENTITY_FIELDS = ('schema', 'socket_path', 'session_id', 'owner_pid', 'owner_start_identity',
                   'browser_pid', 'browser_start_identity', 'page_id')


def validate(identity):
    if not isinstance(identity, dict) or identity.get('schema') != SCHEMA:
        raise ValueError('browser reveal identity is missing or unsupported')
    for key in ('socket_path', 'session_id', 'owner_start_identity', 'browser_start_identity', 'page_id'):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError('browser reveal identity lacks ' + key)
    if not Path(identity['socket_path']).is_absolute():
        raise ValueError('browser reveal socket must be absolute')
    if str(UUID(identity['session_id'])) != identity['session_id']:
        raise ValueError('browser reveal session ID must be a canonical UUID')
    if any(type(identity.get(key)) is not int or identity[key] <= 0 for key in ('owner_pid', 'browser_pid')):
        raise ValueError('browser reveal process IDs must be positive integers')
    return {key: identity[key] for key in IDENTITY_FIELDS}


def verify(identity):
    identity = validate(identity)
    for key in ('owner', 'browser'):
        if GPUQueue._process_start_identity(identity[key + '_pid']) != identity[key + '_start_identity']:
            raise ValueError('browser reveal ' + key + ' process expired or changed')
    path = Path(identity['socket_path'])
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError('browser reveal socket is unavailable: ' + str(error)) from error
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('browser reveal socket must be private and owned by the current user')
    return identity


def descendant(pid, parent):
    """Require the reveal service and browser to belong to the running job's tree."""
    seen = set()
    while type(pid) is int and pid > 1 and pid not in seen:
        if pid == parent:
            return True
        seen.add(pid)
        try:
            pid = int(subprocess.run(['ps', '-p', str(pid), '-o', 'ppid='],
                check=True, capture_output=True, text=True).stdout.strip())
        except (OSError, ValueError, subprocess.CalledProcessError):
            return False
    return False


def exchange(application, request_id, action):
    identity = verify(application.get('reveal'))
    command = {'schema': 'gpu-greenroom.application-reveal-request.v1', 'action': action,
        'call_id': str(uuid4()), 'request_id': request_id,
        **{key: application[key] for key in ('request_digest', 'job_id', 'url')}, 'reveal': identity}
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(identity['socket_path'])
            with client.makefile('rwb') as stream:
                stream.write(json.dumps(command).encode() + b'\n')
                stream.flush()
                receipt = json.loads(stream.readline())
    except (OSError, ValueError) as error:
        raise ValueError('browser reveal owner unavailable: ' + str(error)) from error
    expected = {key: value for key, value in command.items() if key != 'schema'}
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'gpu-greenroom.application-reveal-receipt.v1'
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError('browser reveal receipt identity conflicts with the displayed session')
    if receipt.get('error'):
        raise ValueError('browser reveal owner: ' + str(receipt['error']))
    if action == 'reveal' and receipt.get('foreground') is not True:
        raise ValueError('browser reveal did not verify the foreground application')
    verify(identity)
    return receipt
