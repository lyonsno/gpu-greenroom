"""Publish the current application's smoke destination at its usable boundary."""

import os
from pathlib import Path

from .smoke_requests import SmokeRequests


def publish(request_id, *, phase='operator-needed', label, queue_dir=None, browser_url=None, browser_reveal=None):
    """Call in the terminal process, or from a browser application's server."""
    root = Path(queue_dir or os.environ.get('GPU_GREENROOM_DIR', '~/.local/state/gpu-greenroom')).expanduser()
    store = SmokeRequests(root / 'smoke-requests')
    if browser_url is not None:
        payload = {'url': browser_url, 'phase': phase, 'label': label}
        if browser_reveal is not None:
            payload['reveal'] = browser_reveal
        return store.publish_application(request_id, payload)
    if browser_reveal is not None:
        raise ValueError('browser reveal needs its actual browser URL')
    pane = os.environ.get('WEZTERM_PANE')
    if pane is None or not pane.isdecimal():
        raise ValueError('publish terminal session from its actual WezTerm process')
    return store.publish_session(request_id, {'pane_id': int(pane), 'pid': os.getpid(), 'phase': phase, 'label': label})
