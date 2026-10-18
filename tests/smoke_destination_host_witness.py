"""Opt-in host conformance: independent browser, real owned TTY, disposable queue.

Run from the Greenroom worktree with: uv run python
tests/smoke_destination_host_witness.py NAVIGATOR_SOURCE CHROME OUTPUT
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_queue.operator_server import make_handler
from gpu_queue.smoke_requests import SmokeRequests
from gpu_queue.smoke_navigation import observe
from gpu_queue.queue import GPUQueue
from tests.test_smoke_progress import job, request

navigator = chrome = output = queue = store = None
output_owned = False
report = {'status': 'failed', 'phase': 'setup', 'last_trustworthy_evidence': {},
          'claim': 'real application navigation and owned terminal activation; synthetic queue, no GPU or speech test'}
server = None
thread = None
identity = None
original_path = os.environ['PATH']
original_navigator = os.environ.get('GPU_GREENROOM_TERMINAL_NAVIGATOR')


def phase(value):
    report['phase'] = value
    (output / 'report.json').write_text(json.dumps(report, indent=2))


async def run():
    global server, thread, identity
    assert 'Google Chrome.app/Contents' not in str(chrome.resolve()), 'Use an independent browser'
    shim = output / 'bin' / 'navigator'
    shim.parent.mkdir()
    shim.write_text('#!/bin/sh\nexec ' + shlex.join([sys.executable, str(navigator)]) + ' "$@"\n')
    shim.chmod(0o700)
    os.environ['PATH'] = str(shim.parent) + os.pathsep + original_path
    os.environ['GPU_GREENROOM_TERMINAL_NAVIGATOR'] = str(shim.resolve())
    phase('owned-terminal-launch')
    pid_file = output / 'terminal-pid.json'
    script = 'import os,json,signal; from pathlib import Path; Path(' + repr(str(pid_file)) + ').write_text(json.dumps({"pid":os.getpid()})); print("Greenroom navigation conformance: owned CPU terminal",flush=True); signal.pause()'
    pane = int(subprocess.check_output(['wezterm', 'cli', '--no-auto-start', 'spawn', '--cwd', str(Path.cwd()), sys.executable, '-u', '-c', script], text=True))
    while not pid_file.exists():
        await asyncio.sleep(.05)
    identity = observe(pane, json.loads(pid_file.read_text())['pid'])
    report['terminal_identity'] = identity
    phase('request-publication')
    terminal = request(queue.queue_dir, 'test-job')
    terminal.update(title='Owned terminal smoke', availability='prepared', target={'kind': 'terminal'})
    job(queue.queue_dir, 'running')
    store.submit(terminal)
    store.publish_session(terminal['id'], {'pane_id': pane, 'pid': identity['pid'], 'phase': 'operator-needed', 'label': 'Owned CPU terminal; awaiting inspection'})
    handler = make_handler(queue.queue_dir, 'disposable-host-token', admission_control=True, local_operator=True)
    class Handler(handler):
        def do_GET(self):
            if self.path == '/actual-smoke':
                body = b'<html><body><h1>Actual smoke application</h1><p>Distinct destination reached.</p></body></html>'
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    report['route'] = base
    browser_request = request(queue.queue_dir)
    browser_request.update(id='6ae8e19c-9441-44d2-82e0-d651b431b978', title='Actual browser smoke', availability='prepared',
                           target={'kind': 'browser', 'url': base + '/actual-smoke'})
    store.submit(browser_request)
    diagnostic = request(queue.queue_dir)
    diagnostic.update(id='d10d06c9-9caa-4cd5-9f99-b1db06e4e3a9', title='Synthetic diagnostic', purpose='diagnostic')
    store.submit(diagnostic)
    phase('native-browser-destinations')
    result = await asyncio.to_thread(subprocess.run, ['node', str(Path(__file__).with_name('operator_browser_witness.mjs')),
        str(chrome), base, 'destinations', terminal['id'], report['greenroom_revision'] + '+' + report['greenroom_diff_sha256'], str(output)],
        capture_output=True, text=True)
    (output / 'browser-stdout.log').write_text(result.stdout)
    (output / 'browser-stderr.log').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError('native browser witness failed: ' + result.stderr)
    report['browser_witness'] = json.loads(result.stdout.strip().splitlines()[-1])
    phase('stale-terminal-refusal')
    os.kill(identity['pid'], signal.SIGTERM)
    while GPUQueue._process_start_identity(identity['pid']) == identity['process_start_identity']:
        await asyncio.sleep(.05)
    display = store.display(store.get(terminal['id']))
    assert display['phase'] == 'unknown' and display.get('terminal_verified') is not True
    try:
        urlopen(Request(base + '/api/smoke-requests/' + terminal['id'] + '/focus',
            data=json.dumps({'request_digest': store.get(terminal['id'])['request_digest'],
                             'destination_digest': store.destination(store.get(terminal['id']))['identity_digest']}).encode(),
            headers={'Authorization': 'Bearer disposable-host-token'}))
    except HTTPError as error:
        assert error.code == 400
        report['stale_refusal'] = {'status': error.code, 'body': error.read().decode()}
    else:
        raise AssertionError('dead terminal focus was not refused')
    report.update(status='passed', phase='complete')


try:
    if len(sys.argv) != 4:
        raise ValueError('Usage: smoke_destination_host_witness.py NAVIGATOR_SOURCE CHROME NEW_OUTPUT')
    navigator, chrome, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=False)
    output_owned = True
    report.update(browser=str(chrome), greenroom=str(Path.cwd()), navigator_source=str(navigator))
    phase('source-discovery')
    for key, root in [('greenroom', Path.cwd()), ('navigator', navigator.parent)]:
        report[key + '_revision'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        report[key + '_diff_sha256'] = hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=root)).hexdigest()
        report['last_trustworthy_evidence'][key + '_revision'] = report[key + '_revision']
        phase('source-discovery')
    phase('queue-setup')
    queue = GPUQueue(output / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    asyncio.run(run())
except Exception as error:
    report['error'] = f'{type(error).__name__}: {error}'
finally:
    os.environ['PATH'] = original_path
    if original_navigator is None:
        os.environ.pop('GPU_GREENROOM_TERMINAL_NAVIGATOR', None)
    else:
        os.environ['GPU_GREENROOM_TERMINAL_NAVIGATOR'] = original_navigator
    if identity and GPUQueue._process_start_identity(identity['pid']) == identity['process_start_identity']:
        os.kill(identity['pid'], signal.SIGTERM)
    if server:
        server.shutdown()
        server.server_close()
        thread.join()
    if output_owned:
        (output / 'report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
sys.exit(0 if report['status'] == 'passed' else 1)
