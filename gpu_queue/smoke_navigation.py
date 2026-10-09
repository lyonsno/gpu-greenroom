"""Observed terminal identity; activation uses the configured terminal navigator."""

import json
import os
import shutil
import subprocess
from pathlib import Path


def observe(pane_id, pid):
    from .queue import GPUQueue
    if type(pane_id) is not int or pane_id < 0 or type(pid) is not int or pid <= 0:
        raise ValueError('terminal pane and process IDs must be integers')
    wezterm = shutil.which('wezterm') or '/Applications/WezTerm.app/Contents/MacOS/wezterm'
    try:
        listing = subprocess.run([wezterm, 'cli', '--no-auto-start', 'list', '--format', 'json'],
                                 check=True, capture_output=True, text=True)
        panes = json.loads(listing.stdout)
        if not isinstance(panes, list) or any(not isinstance(pane, dict) for pane in panes):
            raise ValueError('malformed terminal inventory')
        matches = [pane for pane in panes if pane.get('pane_id') == pane_id]
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError('terminal inventory unavailable: ' + str(error)) from error
    if len(matches) != 1 or not matches[0].get('tty_name'):
        raise ValueError('terminal pane is absent or ambiguous')
    start = GPUQueue._process_start_identity(pid)
    try:
        tty = subprocess.run(['ps', '-p', str(pid), '-o', 'tty='],
                             check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('terminal process unavailable: ' + str(error)) from error
    if not start or tty in {'', '??', '-'} or '/dev/' + tty != matches[0]['tty_name']:
        raise ValueError('terminal process does not own the observed pane tty')
    return {'pane_id': pane_id, 'pid': pid, 'tty_name': matches[0]['tty_name'],
            'process_start_identity': start}


def verify(identity):
    observed = observe(identity['pane_id'], identity['pid'])
    if any(identity.get(key) != value for key, value in observed.items()):
        raise ValueError('terminal process or pane identity is stale')
    return observed


def focus(identity):
    verify(identity)
    executable = os.environ.get('GPU_GREENROOM_TERMINAL_NAVIGATOR', '')
    if not Path(executable).is_absolute() or not os.access(executable, os.X_OK):
        raise ValueError('Configured terminal navigation is unavailable')
    result = subprocess.run([executable, 'focus-pane', '--pane-id', str(identity['pane_id']),
                             '--expected-tty', identity['tty_name'], '--json'],
                            capture_output=True, text=True)
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'Shared terminal navigation failed')
    receipt = json.loads(result.stdout)
    if not isinstance(receipt, dict):
        raise ValueError('Navigator receipt is not an object; activation may have occurred')
    if receipt.get('pane_id') != identity['pane_id']:
        raise ValueError('Shared terminal navigation returned a different pane')
    return receipt
