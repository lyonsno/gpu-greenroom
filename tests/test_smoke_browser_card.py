import json
import re
import subprocess
import os
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

import pytest

from gpu_queue.operator_server import PAGE


def render(item):
    match = re.search(r'function smokeDestination\(item\)\{(.*?)\n\}', PAGE, re.S)
    if match:
        function = match.group(0)
    else:
        block = PAGE.split('    const target=item.destination||')[1].split('    const phase=')[0]
        function = 'function smokeDestination(item){const q=item.request;const target=item.destination||' + block + 'return destination;}'
    script = '''const location=new URL('http://127.0.0.1:8766/#smoke-fixture');
const document={body:{dataset:{readOnly:'false'}}};
const esc=x=>String(x??'');
''' + function + '\nconsole.log(smokeDestination(' + json.dumps(item) + '));'
    return subprocess.run(['node', '-e', script], check=True, capture_output=True, text=True).stdout.strip()


def card(kind='context', **display):
    return {'request': {'id': 'fixture', 'url': 'http://127.0.0.1:8766/#smoke-fixture'},
        'request_digest': 'request', 'destination': {'kind': kind, 'url': 'http://127.0.0.1:8766/#smoke-fixture',
        'identity_digest': 'destination'}, 'display': display}


def test_self_card_is_not_advertised_as_an_application_entry():
    assert render(card(phase='awaiting-start')) == ''
    value = card(phase='waiting-gpu')
    value['destination']['url'] = 'http://127.0.0.1:8766/?smoke=fixture#smoke-fixture'
    assert render(value) == ''


def test_running_browser_focus_is_an_action_not_a_new_tab():
    value = card('browser', job_id='job', job_state='running', application_verified=True)
    value['destination']['url'] = 'http://127.0.0.1:61946/log'
    output = render(value)
    assert 'smokeFocus' in output and 'Show application' in output
    assert 'data-destination="destination"' in output
    assert 'href=' not in output and 'disabled' not in output


def test_url_only_running_browser_has_no_false_reveal_action():
    value = card('browser', job_id='job', job_state='running', application_reported=True)
    value['destination']['url'] = 'http://127.0.0.1:61946/log'
    output = render(value)
    assert 'href=' not in output
    assert 'disabled' in output


def test_ended_browser_does_not_offer_a_dead_application():
    assert render(card('browser', job_id='job', job_state='done', run_available=True)) == ''


def test_explicit_static_page_remains_a_link_not_a_managed_session():
    value = card('browser')
    value['destination']['url'] = 'http://127.0.0.1:61946/retained-recording'
    assert 'href="http://127.0.0.1:61946/retained-recording"' in render(value)


@pytest.mark.skipif(not os.environ.get('GREENROOM_CHROME'), reason='independent browser not configured')
def test_actual_monitor_browser_button_routes_reveal_without_another_page(tmp_path, monkeypatch):
    from tests.test_smoke_browser_entry import revealed_store
    from gpu_queue.operator_server import make_handler
    root = tmp_path / 'queue'
    root.mkdir()
    store, value, _, _, owner = revealed_store(root, monkeypatch)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(root, 'fixture', admission_control=True, local_operator=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(['node', str(Path(__file__).with_name('operator_browser_witness.mjs')),
            os.environ['GREENROOM_CHROME'], f'http://127.0.0.1:{server.server_port}', 'browser-entry',
            value['id'], 'owned-browser-entry-candidate', str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report['nativeForeground'] == 'synthetic owner receipt; not claimed'
        assert [call['action'] for call in owner.calls].count('reveal') == 2
        print('GREENROOM_BROWSER_ENTRY ' + json.dumps(report))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        owner.close()
