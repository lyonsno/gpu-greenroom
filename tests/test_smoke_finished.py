import json
import os
from pathlib import Path
from http.server import ThreadingHTTPServer
import subprocess
import sys
import threading
import wave
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest

from gpu_queue.operator_server import PAGE, make_handler
from gpu_queue.queue import GPUQueue
from gpu_queue.smoke_requests import SmokeRequests
from tests.test_smoke_admission import command, smoke


def finished(root):
    queue = GPUQueue(root / 'queue')
    store = SmokeRequests(queue.queue_dir / 'smoke-requests')
    value = smoke(root, command(root))
    value['target'] = {'kind': 'terminal'}
    record, _ = store.submit(value)
    store.start(value['id'], record['request_digest'])
    assert queue.run_one({})
    record = store.get(value['id'])
    return queue, store, record


def successor(root, store, **changes):
    value = smoke(root, {**command(root), 'output_dir': str(root / 'next-output')})
    value.update(id=str(uuid4()), target={'kind': 'terminal'}, **changes)
    return store.submit(value)[0]


def test_finished_card_exposes_end_time_and_native_outcome_not_participation(tmp_path):
    queue, store, record = finished(tmp_path)
    display = store.display(record)
    native = json.loads((queue.queue_dir / 'done' / record['activation']['job_id'] / 'status.json').read_text())
    assert display.get('finished_at') == native['finished_at'], 'Ended card omits its end time'
    assert display.get('exit_code') == 0
    assert display['label'] == 'Last session ended; your experience has not been recorded'
    assert display.get('run_available') is True
    assert display.get('terminal_verified') is not True


def test_retained_run_is_bound_to_exact_job_and_preserves_full_logs(tmp_path):
    queue, store, record = finished(tmp_path)
    details = store.run_details(record['request']['id'])
    assert details['job_id'] == record['activation']['job_id']
    assert details['exit_code'] == 0 and details['native_state'] == 'done'
    assert 'native child' in details['stdout']
    assert details['participation'] == 'not-recorded'
    assert details['request_digest'] == record['request_digest']
    path = queue.queue_dir / 'done' / details['job_id']
    (path / 'status.json').write_text(json.dumps({'job_id': 'other', 'status': 'done'}))
    with pytest.raises(ValueError, match='identity'):
        store.run_details(record['request']['id'])


def test_missed_session_response_is_explicit_and_idempotent(tmp_path):
    _, store, record = finished(tmp_path)
    identity = record['request']['id']
    response = store.respond(identity, 'It ended before I could try it.', participation='not-tried')
    assert response['response']['participation'] == 'not-tried'
    assert store.respond(identity, 'It ended before I could try it.', participation='not-tried') == response
    with pytest.raises(ValueError, match='different response'):
        store.respond(identity, 'It ended before I could try it.', participation='tried')


def test_repeat_request_is_receipted_without_reusing_or_enqueuing_a_job(tmp_path):
    queue, store, record = finished(tmp_path)
    identity = record['request']['id']
    result = store.request_repeat(identity, record['request_digest'])
    assert store.request_repeat(identity, record['request_digest']) == result
    assert result['repeat_request']['job_id'] == record['activation']['job_id']
    assert result['repeat_request']['gpu_execution_authority'] is False
    assert not list((queue.queue_dir / 'pending').iterdir())
    assert store.get(identity)['request_digest'] == record['request_digest']
    with pytest.raises(ValueError, match='changed'):
        store.request_repeat(identity, 'wrong')


def test_successor_is_fresh_prepared_work_and_start_remains_operator_owned(tmp_path):
    queue, store, record = finished(tmp_path)
    following = successor(tmp_path, store)
    configured = store.configure(record['request']['id'], {'next_request_id': following['request']['id']})
    display = store.display(configured)
    assert display['next_session']['id'] == following['request']['id']
    assert display['next_session']['phase'] == 'awaiting-start'
    assert not list((queue.queue_dir / 'pending').iterdir())
    store.start(following['request']['id'], display['next_session']['request_digest'])
    assert store.display(configured)['next_session']['phase'] == 'waiting-gpu'
    assert record['activation']['job_id'] != following['prepared']['job_request']['job_id']


@pytest.mark.parametrize('bad', ['self', 'owner', 'output', 'stale-input'])
def test_invalid_successor_cannot_be_presented_as_startable(tmp_path, bad):
    _, store, record = finished(tmp_path)
    if bad == 'self':
        following = record
    else:
        manifest = command(tmp_path)
        manifest['output_dir'] = str(tmp_path / 'next-output') if bad != 'output' else record['prepared']['job_request']['output_dir']
        if bad == 'stale-input':
            script = tmp_path / 'next.py'
            script.write_text("print('first')")
            manifest['argv'] = [sys.executable, str(script)]
        value = smoke(tmp_path, manifest)
        value['id'] = str(uuid4())
        if bad == 'owner':
            value['source']['agent_id'] = manifest['agent_id'] = 'other-owner'
        following = store.submit(value)[0]
        if bad == 'stale-input':
            script.write_text("print('changed')")
    with pytest.raises(ValueError):
        store.configure(record['request']['id'], {'next_request_id': following['request']['id']})
    assert 'next_request_id' not in store.get(record['request']['id']).get('presentation', {})


def test_retained_audio_is_explicit_and_cannot_escape_native_output_root(tmp_path):
    _, store, record = finished(tmp_path)
    root = tmp_path / 'output'
    root.mkdir(exist_ok=True)
    (root / 'playback.wav').write_bytes(b'RIFF retained audio')
    configured = store.configure(record['request']['id'], {'review_artifacts': [
        {'path': 'playback.wav', 'label': 'Actual playback'}]})
    assert configured['request_digest'] == record['request_digest']
    path, content_type = store.review_artifact(record['request']['id'], 0)
    assert path == root / 'playback.wav' and content_type == 'audio/wav'
    external = tmp_path / 'secret.wav'
    external.write_bytes(b'private')
    (root / 'escape.wav').symlink_to(external)
    for path in ['../secret.wav', str(external), 'escape.wav']:
        with pytest.raises(ValueError):
            store.configure(record['request']['id'], {'review_artifacts': [{'path': path, 'label': 'Bad'}]})
    assert store.get(record['request']['id']) == configured
    (root / 'playback.wav').unlink()
    (root / 'playback.wav').symlink_to(external)
    with pytest.raises(ValueError):
        store.review_artifact(record['request']['id'], 0)


def test_finished_run_http_is_authenticated_and_has_no_user_selected_path(tmp_path):
    _, store, record = finished(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(store.directory.parent, 'secret', admission_control=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/api/smoke-requests/{record["request"]["id"]}/run'
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(url)
        assert error.value.code == 401
        result = json.load(urlopen(Request(url, headers={'Authorization': 'Bearer secret'})))
        assert result['job_id'] == record['activation']['job_id'] and 'native child' in result['stdout']
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + '/../../secret', headers={'Authorization': 'Bearer secret'}))
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_missing_log_is_explicit_not_blank_success(tmp_path):
    queue, store, record = finished(tmp_path)
    (queue.queue_dir / 'done' / record['activation']['job_id'] / 'stdout.log').unlink()
    run = store.run_details(record['request']['id'])
    assert run['stdout'] is None
    assert 'stdout.log:' in run['errors'][0]


def test_successor_validation_is_rechecked_after_configuration(tmp_path):
    _, store, record = finished(tmp_path)
    script = tmp_path / 'next.py'
    script.write_text("print('first')")
    value = smoke(tmp_path, {**command(tmp_path), 'output_dir': str(tmp_path / 'next-output'),
                            'argv': [sys.executable, str(script)]})
    value['id'] = str(uuid4())
    store.submit(value)
    configured = store.configure(record['request']['id'], {'next_request_id': value['id']})
    script.write_text("print('changed')")
    display = store.display(configured)
    assert 'next_session' not in display
    assert 'changed' in display['next_session_error']


def test_reader_identity_describes_loaded_code_not_later_source_edits(tmp_path, monkeypatch):
    store = SmokeRequests(tmp_path / 'smoke-requests')
    original = store.snapshot()['reader_sha256']
    monkeypatch.setattr(Path, 'read_bytes', lambda path: b'later source not loaded by this process')
    assert store.snapshot()['reader_sha256'] == original


@pytest.mark.parametrize('change', ['replace', 'withdraw', 'participation'])
def test_loaded_run_refresh_tracks_published_audio_and_participation(tmp_path, change):
    _, store, record = finished(tmp_path)
    identity = record['request']['id']
    output = Path(record['prepared']['job_request']['output_dir'])
    output.mkdir(exist_ok=True)
    audio = output / 'playback.wav'
    audio.write_bytes(b'RIFF first recording')
    store.configure(identity, {'review_artifacts': [{'path': 'playback.wav', 'label': 'FIRST recording'}]})
    old_item = store.snapshot()['items'][0]
    old_run = store.run_details(identity)
    if change == 'replace':
        audio.write_bytes(b'RIFF replacement recording')
        store.configure(identity, {'review_artifacts': [{'path': 'playback.wav', 'label': 'SECOND recording'}]})
    elif change == 'withdraw':
        store.configure(identity, {'review_artifacts': []})
    store.respond(identity, 'I tried it', participation='tried')
    script = PAGE.split('<script>')[-1].split('</script>')[0].split('async function loadSmoke')[0]
    harness = r'''
const host={innerHTML:'',querySelectorAll:()=>[],before:()=>{}};
const views={innerHTML:'',querySelectorAll:()=>[]};
globalThis.location={hash:'',href:'http://127.0.0.1:8766/',origin:'http://127.0.0.1:8766',pathname:'/'};
globalThis.sessionStorage={getItem:()=>'',setItem:()=>{}};
globalThis.document={body:{dataset:{}},activeElement:null,querySelector:s=>s==='#smokeRequestList'?host:s==='#smokeViews'?views:{textContent:''},querySelectorAll:()=>[]};
const player={pause:()=>{},src:'blob:old'};
'''
    code = harness + script + 'smokeFilter="history";'
    code += f'const identity={json.dumps(identity)},oldItem={json.dumps(old_item)};'
    code += f'smokeRuns.set(identity,{json.dumps(old_run)});smokeAudio.set(identity+":0",player);'
    code += 'if(typeof smokeRunVersions!=="undefined")smokeRunVersions.set(identity,smokeRunVersion(oldItem));'
    code += 'if(typeof smokeAudioBindings!=="undefined")smokeAudioBindings.set(identity+":0",smokeAudioBinding(oldItem,0));'
    code += f'renderSmoke({json.dumps(store.snapshot())});'
    code += 'console.log(JSON.stringify({oldLabel:host.innerHTML.includes("FIRST recording"),oldParticipation:host.innerHTML.includes("Participation: not-recorded"),playerRetained:smokeAudio.get(identity+":0")===player}));'
    result = subprocess.run(['node', '-e', code], capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual['oldParticipation'] is False
    assert actual['oldLabel'] is (change == 'participation')
    assert actual['playerRetained'] is (change == 'participation')


def test_finished_ui_has_no_live_directions_or_dead_terminal_action():
    # Execute the actual page renderer; synthetic data tests presentation, not host identity.
    script = PAGE.split('<script>')[-1].split('</script>')[0].split('async function loadSmoke')[0]
    script += r'''
const payload={schema:'gpu-greenroom.smoke-request-list.v1',errors:[],items:[{
 request:{id:'one',title:'Finished',prompt:'Start session. Press Enter to talk.',source:{}},
 status:'operator-needed',request_digest:'digest',destination:{kind:'terminal',pane_id:59},
 display:{phase:'awaiting-response',job_state:'done',section:'active',finished_at:1791590454,exit_code:0,run_available:true}}]};
renderSmoke(payload);
console.log(host.innerHTML); console.log(views.innerHTML);
'''
    harness = r'''
const host={innerHTML:'',querySelectorAll:()=>[],before:()=>{}};
const views={innerHTML:'',querySelectorAll:()=>[]};
globalThis.location={hash:'',href:'http://127.0.0.1:8766/',origin:'http://127.0.0.1:8766',pathname:'/'};
globalThis.sessionStorage={getItem:()=>'',setItem:()=>{}};
globalThis.document={body:{dataset:{}},activeElement:null,querySelector:s=>s==='#smokeRequestList'?host:s==='#smokeViews'?views:{textContent:''},querySelectorAll:()=>[]};
'''
    result = subprocess.run(['node', '-e', harness + script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Start session. Press Enter' not in result.stdout
    assert 'Open terminal' not in result.stdout
    assert 'View last run' in result.stdout
    assert 'Did you get to try this session?' in result.stdout
    assert 'Request another session' in result.stdout
    assert 'Needs attention (1)' in result.stdout


@pytest.mark.skipif(not os.environ.get('GREENROOM_CHROME'), reason='independent browser not configured')
def test_browser_finished_session_actions_use_real_api_and_preserve_audio(tmp_path):
    queue, store, record = finished(tmp_path)
    output = Path(record['prepared']['job_request']['output_dir'])
    output.mkdir(exist_ok=True)
    with wave.open(str(output / 'playback.wav'), 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b'\0\0' * 2400)
    store.configure(record['request']['id'], {'review_artifacts': [{'path': 'playback.wav', 'label': 'Synthetic recorded audio'}]})
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(queue.queue_dir, 'fixture-secret', admission_control=True, local_operator=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(['node', str(Path(__file__).with_name('operator_browser_witness.mjs')),
            os.environ['GREENROOM_CHROME'], f'http://127.0.0.1:{server.server_port}', 'finished-session',
            record['request']['id'], 'working-source/CPU-native-fixture', str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        final = store.get(record['request']['id'])
        assert final['response']['participation'] == 'not-tried'
        assert final['response']['text'] == 'Synthetic missed-session fixture'
        assert final['repeat_request']['gpu_execution_authority'] is False
        assert not list((queue.queue_dir / 'pending').iterdir())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
