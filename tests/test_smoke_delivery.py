from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import subprocess
import threading
from http.server import ThreadingHTTPServer
from uuid import uuid4

import pytest

from gpu_queue.operator_server import PAGE, queue_snapshot, make_handler
from tests.test_smoke_admission import command, smoke
from tests.test_smoke_finished import finished


def relay(store, adapter):
    from gpu_queue.smoke_delivery import SmokeDelivery
    return SmokeDelivery(store, adapter=adapter)


def test_saved_feedback_has_honest_delivery_state_before_relay(tmp_path):
    _, store, record = finished(tmp_path)
    store.respond(record['request']['id'], 'Exact feedback.', participation='tried')
    item = store.snapshot()['items'][0]
    assert item.get('delivery', {}).get('response', {}).get('status') == 'saved', (
        'Saved feedback must not be presented as returned to its owner')


def test_two_columns_keep_fast_work_out_of_operator_column():
    assert 'id="automaticQueue"' in PAGE, 'Automatic work needs its own column'
    assert 'id="operatorQueue"' in PAGE, 'Operator smokes need their own column'
    assert 'id="quickQueue"' in PAGE, 'Quick automatic work needs its own optional visual column'
    assert "responded:'Response returned'" not in PAGE


def test_monitor_uses_effective_quick_policy_and_projected_queue_position(tmp_path):
    queue, _, _ = finished(tmp_path)
    config = {'schema': 'gpu-greenroom.dispatch-policy.v1', 'mode': 'two-class',
              'quick_routes': {'live-smoke/test': 'observed CPU timing'}, 'quick_job_types': {}}
    (queue.queue_dir / 'dispatch-policy.json').write_text(json.dumps(config))
    job = queue_snapshot(queue)['jobs'][0]
    assert job.get('dispatch_class') == 'normal', 'Later policy changes must not rewrite native dispatch identity'
    assert job.get('dispatch_source') == 'native-dispatch-receipt'
    from gpu_queue import admission
    plan = admission.prepare(queue, {**command(tmp_path), 'output_dir': str(tmp_path / 'pending-output')})
    admission.submit_prepared(queue, plan)
    pending = next(item for item in queue_snapshot(queue)['jobs'] if item['status'] == 'pending')
    assert pending['dispatch_class'] == 'quick'
    assert pending['dispatch_basis'] == 'observed CPU timing'
    assert pending['queue_position'] == 1


def test_feedback_and_repeat_have_distinct_idempotent_source_linked_returns(tmp_path):
    _, store, record = finished(tmp_path)
    calls = []
    def adapter(event, operation):
        calls.append((event, operation))
        return {'event_id': event['event_id'], 'status': 'queued', 'receipt': 'peer receipt'}
    delivery = relay(store, adapter)
    delivery.activate()
    store.respond(record['request']['id'], '  Exact\nfeedback.  ', participation='tried')
    store.request_repeat(record['request']['id'], record['request_digest'])
    delivery.run_once()
    delivery.run_once()
    delivered = [event for event, operation in calls if operation == 'deliver']
    assert {event['kind'] for event in delivered} == {'response', 'repeat'}
    assert len(delivered) == 2
    response = next(event for event in delivered if event['kind'] == 'response')
    assert response['payload']['text'] == '  Exact\nfeedback.  '
    assert response['request_digest'] == record['request_digest']
    assert response['source'] == record['request']['source']
    assert response['record_path'] == str(store.path(record['request']['id']).resolve())
    assert store.snapshot()['items'][0]['delivery']['response']['status'] == 'queued'


def test_queued_return_refreshes_to_receiver_consumed_without_resubmitting(tmp_path):
    _, store, record = finished(tmp_path)
    calls = []
    def adapter(event, operation):
        calls.append(operation)
        return {'event_id': event['event_id'], 'status': 'queued' if operation == 'deliver' else 'consumed'}
    delivery = relay(store, adapter)
    delivery.activate()
    store.respond(record['request']['id'], 'Good.')
    delivery.run_once()
    delivery.run_once()
    assert calls == ['deliver', 'status']
    assert store.snapshot()['items'][0]['delivery']['response']['status'] == 'consumed'


@pytest.mark.parametrize('ack', [None, {'event_id': 'other', 'status': 'consumed'}, {'status': 'accepted'}])
def test_blank_or_wrong_adapter_receipt_cannot_claim_feedback_delivered(tmp_path, ack):
    _, store, record = finished(tmp_path)
    delivery = relay(store, lambda event, operation: ack)
    delivery.activate()
    store.respond(record['request']['id'], 'Feedback.')
    delivery.run_once()
    state = store.snapshot()['items'][0]['delivery']['response']
    assert state['status'] == 'uncertain'
    assert state['error']


def test_cutover_does_not_spam_historical_responses_or_diagnostics(tmp_path):
    _, store, record = finished(tmp_path)
    store.respond(record['request']['id'], 'Previously saved.')
    calls = []
    delivery = relay(store, lambda *args: calls.append(args))
    delivery.activate()
    delivery.run_once()
    assert not calls
    assert store.snapshot()['items'][0]['delivery']['response']['status'] == 'saved-before-activation'


def test_concurrent_relays_claim_only_one_delivery_and_recover_uncertain_by_status(tmp_path):
    _, store, record = finished(tmp_path)
    calls = []
    def adapter(event, operation):
        calls.append(operation)
        if operation == 'deliver':
            raise RuntimeError('lost acknowledgement after send')
        return {'event_id': event['event_id'], 'status': 'queued'}
    delivery = relay(store, adapter)
    delivery.activate()
    store.respond(record['request']['id'], 'Feedback.')
    with ThreadPoolExecutor(2) as executor:
        list(executor.map(lambda _: delivery.run_once(), range(2)))
    relay(store, adapter).run_once()
    assert calls.count('deliver') == 1
    assert 'status' in calls
    assert store.snapshot()['items'][0]['delivery']['response']['status'] == 'queued'


def test_loading_stale_and_prepared_sessions_cannot_generate_usable_notification(tmp_path, monkeypatch):
    _, store, record = finished(tmp_path)
    calls = []
    delivery = relay(store, lambda *args: calls.append(args))
    delivery.activate()
    monkeypatch.setattr(store, 'display', lambda _: {'phase': 'loading', 'terminal_verified': True})
    delivery.run_once()
    monkeypatch.setattr(store, 'display', lambda _: {'phase': 'operator-needed', 'terminal_verified': False})
    delivery.run_once()
    monkeypatch.setattr(store, 'display', lambda _: {'phase': 'awaiting-start'})
    delivery.run_once()
    assert not calls


def test_missing_adapter_is_definitely_unsent_and_explicitly_retryable(tmp_path):
    _, store, record = finished(tmp_path)
    delivery = relay(store, str(tmp_path / 'missing-adapter'))
    delivery.activate()
    store.respond(record['request']['id'], 'Preserve this feedback.')
    delivery.run_once()
    event = delivery.events(store.get(record['request']['id']))[0]
    assert delivery.summary(store.get(record['request']['id']))['response']['status'] == 'failed'
    delivery.retry(store.get(record['request']['id']), 'response', event['event_id'])
    calls = []
    delivery.adapter = lambda value, op: calls.append(op) or {'event_id': value['event_id'], 'status': 'queued'}
    delivery.run_once()
    assert calls == ['deliver']


@pytest.mark.parametrize('malformation', ['json', 'missing-status', 'wrong-event', 'unknown-status'])
def test_bad_receipt_is_preserved_without_stalling_another_owner(tmp_path, malformation):
    _, store, record = finished(tmp_path)
    other = smoke(tmp_path, {**command(tmp_path), 'output_dir': str(tmp_path / 'other')})
    other.pop('operator_command')
    other['id'] = str(uuid4())
    other['source']['agent_id'] = 'another-owner'
    store.submit(other)
    calls = []
    delivery = relay(store, lambda event, op: calls.append(event['request_id']) or {'event_id': event['event_id'], 'status': 'queued'})
    delivery.activate()
    store.respond(record['request']['id'], 'First feedback.')
    store.respond(other['id'], 'Second feedback.')
    first = delivery.events(store.get(record['request']['id']))[0]
    path = delivery.directory / (first['event_id'] + '.json')
    value = {'event': first, 'event_id': first['event_id']}
    if malformation == 'wrong-event':
        value['event'] = {**first, 'request_digest': 'wrong'}
        value['status'] = 'queued'
    if malformation == 'unknown-status':
        value['status'] = 'made-up'
    raw = '{broken' if malformation == 'json' else json.dumps(value)
    path.write_text(raw)
    delivery.run_once()
    delivery.run_once()
    assert record['request']['id'] not in calls
    assert other['id'] in calls
    assert path.read_text() == raw
    assert delivery.summary(store.get(record['request']['id']))['response']['status'] == 'unverified'
    assert delivery.summary(store.get(other['id']))['response']['status'] == 'queued'
    assert (delivery.directory / (first['event_id'] + '.error.json')).is_file()
    with pytest.raises(ValueError):
        delivery.retry(store.get(record['request']['id']), 'response', first['event_id'])


@pytest.mark.parametrize('mode', ['full', 'admission', 'readonly'])
@pytest.mark.parametrize('state,expected', [('pending', 'Cancel'), ('running', 'Force stop')])
@pytest.mark.parametrize('consistent', [True, False])
def test_linked_smoke_keeps_only_its_authorized_native_actions(tmp_path, mode, state, expected, consistent):
    import re
    script = re.search(r'<script>(.*?)</script>', PAGE, re.S).group(1)
    script = script[:script.index("document.querySelector('#pause').onclick")]
    fixture = """
const nodes=new Map();
function node(key){if(!nodes.has(key))nodes.set(key,{innerHTML:'',style:{},dataset:{},querySelectorAll:()=>[],before:()=>{}});return nodes.get(key)}
global.document={body:{dataset:{readOnly:MODE==='readonly'?'true':'false'}},activeElement:null,querySelector:node,querySelectorAll:()=>[],createElement:()=>node('created'),getElementById:()=>null};
global.location={href:'http://127.0.0.1:8766/',search:'',hash:'',origin:'http://127.0.0.1:8766',pathname:'/'};
""".replace('MODE', json.dumps(mode))
    action = """
render({jobs:[{job_id:'owned-job',status:STATE,consistent:CONSISTENT,submitted_at:1}],paused:false,read_only:MODE==='readonly',admission_control:MODE==='admission'});
renderSmoke({schema:'gpu-greenroom.smoke-request-list.v1',errors:[],items:[{request:{id:'smoke',source:{},title:'Owned smoke'},status:'operator-needed',display:{job_id:'owned-job',job_state:STATE,phase:'waiting-gpu'},destination:{kind:'context'}}]});
console.log(JSON.stringify({automatic:node('#jobs').innerHTML,operator:node('#smokeRequestList').innerHTML}));
""".replace('MODE', json.dumps(mode)).replace('STATE', json.dumps(state)).replace('CONSISTENT', json.dumps(consistent))
    result = subprocess.run(['node', '-e', fixture + script + action], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    markup = json.loads(result.stdout)
    assert 'owned-job' not in markup['automatic']
    if mode == 'full' and consistent:
        assert expected in markup['operator']
    else:
        assert "act('" not in markup['operator']


def test_terminal_notification_is_bound_and_not_duplicated_on_phase_changes(tmp_path, monkeypatch):
    _, store, record = finished(tmp_path)
    calls = []
    delivery = relay(store, lambda event, op: calls.append(event) or {'event_id': event['event_id'], 'status': 'notified'})
    delivery.activate()
    record['session'] = {'schema': 'gpu-greenroom.smoke-session.v1', 'request_digest': record['request_digest'],
        'job_id': record['activation']['job_id'], 'pane_id': 9, 'pid': 1, 'tty_name': '/dev/ttys-test',
        'process_start_identity': 'fixture', 'phase': 'operator-needed', 'label': 'Fixture terminal',
        'observed_at': '2099-01-01T00:00:00+00:00'}
    store._write(record)
    monkeypatch.setattr(store, 'display', lambda _: {'phase': 'operator-needed', 'terminal_verified': True, 'job_state': 'running', 'job_id': 'job'})
    monkeypatch.setattr(store, 'destination', lambda _: {'kind': 'terminal', 'identity_digest': 'bound-digest', 'pane_id': 9})
    delivery.run_once()
    monkeypatch.setattr(store, 'display', lambda _: {'phase': 'interactive', 'terminal_verified': True, 'job_state': 'running', 'job_id': 'job'})
    delivery.run_once()
    assert len(calls) == 1
    assert calls[0]['payload']['destination']['identity_digest'] == 'bound-digest'
    assert calls[0]['kind'] == 'usable'


@pytest.mark.skipif(not os.environ.get('GREENROOM_CHROME'), reason='independent browser not configured')
def test_parallel_columns_in_independent_desktop_and_mobile_browser(tmp_path):
    from gpu_queue import admission
    queue, store, record = finished(tmp_path)
    value = smoke(tmp_path, {**command(tmp_path), 'output_dir': str(tmp_path / 'next')})
    value['id'] = str(uuid4())
    value['title'] = 'Prepared operator session'
    store.submit(value)
    config = {'schema': 'gpu-greenroom.dispatch-policy.v1', 'mode': 'two-class',
        'quick_routes': {'automatic/finite': 'Observed finite CPU fixture'}, 'quick_job_types': {}}
    (queue.queue_dir / 'dispatch-policy.json').write_text(json.dumps(config))
    plan = admission.prepare(queue, {**command(tmp_path), 'route_identity': 'automatic/finite',
        'output_dir': str(tmp_path / 'automatic'), 'service_class': 'quick'})
    admission.submit_prepared(queue, plan)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(queue.queue_dir, 'fixture', admission_control=True, local_operator=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(['node', str(Path(__file__).with_name('operator_browser_witness.mjs')),
            os.environ['GREENROOM_CHROME'], f'http://127.0.0.1:{server.server_port}', 'columns',
            record['request']['id'], 'prepared-smoke-composition-local-diff', str(tmp_path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)['browser'].startswith('Chrome/')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
