import json

import pytest

from gpu_queue.smoke_requests import SmokeRequestConflict, SmokeRequests


def request(root, job_id=None):
    value = {'schema': 'gpu-greenroom.interactive-smoke.v1', 'kind': 'interactive-smoke',
             'id': '9c0a03f6-b2d8-43f4-b7b7-5e848e733661',
             'source': {'agent_id': 'example-owner', 'repo_root': str(root)},
             'title': 'Inspect capture', 'prompt': 'Inspect the capture and respond.',
             'url': 'http://127.0.0.1:8766/', 'availability': 'preparation-needed',
             'availability_note': 'Preparing capture'}
    if job_id:
        value['job_id'] = job_id
    return value


def job(root, state, owner='example-owner'):
    path = root/state/'test-job'
    path.mkdir(parents=True)
    (path/'request.json').write_text(json.dumps({'job_id': 'test-job', 'agent_id': owner}))
    (path/'status.json').write_text(json.dumps({'job_id': 'test-job', 'status': state,
                                             'job_type':'command','input_path':'','output_dir':str(root/'outputs/test-job'),
                                             'submitted_at': 10, 'started_at': 20 if state!='pending' else None}))
    return path


def test_smoke_progress_is_receipted_idempotent_and_cas_bound(tmp_path):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path)
    original, _ = store.submit(value)
    update = {'expected_revision': 0, 'availability': 'preparation-needed',
              'label': 'Capturing views', 'completed': 3, 'total': 8, 'unit': 'captures'}
    changed = store.update(value['id'], update)
    assert changed['request_digest'] == original['request_digest']
    assert changed['progress']['revision'] == 1
    assert store.update(value['id'], update) == changed
    with pytest.raises(SmokeRequestConflict):
        store.update(value['id'], {**update, 'completed': 4})
    assert store.snapshot()['items'][0]['display']['phase'] == 'preparing'


@pytest.mark.parametrize('bad', [{'completed': True, 'total': 8, 'unit': 'captures'},
                               {'completed': 9, 'total': 8, 'unit': 'captures'},
                               {'completed': 3}, {'expected_revision': True}])
def test_smoke_rejects_false_or_partial_progress(tmp_path, bad):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path)
    original, _ = store.submit(value)
    with pytest.raises(ValueError):
        store.update(value['id'], {'expected_revision': 0, 'availability': 'preparation-needed',
                                   'label': 'Capture', **bad})
    assert store.get(value['id']) == original


def test_smoke_uses_exact_job_state_and_requires_prepared_before_response(tmp_path):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path, 'test-job')
    store.submit(value)
    path = job(tmp_path, 'pending')
    assert store.snapshot()['items'][0]['display']['phase'] == 'waiting-gpu'
    with pytest.raises(SmokeRequestConflict):
        store.respond(value['id'], 'Inspected')
    (tmp_path/'running').mkdir(exist_ok=True)
    path.rename(tmp_path/'running'/'test-job')
    path = tmp_path/'running'/'test-job'
    (path/'status.json').write_text(json.dumps({'job_id':'test-job','status':'running','submitted_at':10,'started_at':20}))
    assert store.snapshot()['items'][0]['display']['phase'] == 'running'


def test_wrong_job_owner_and_missing_job_are_not_progress(tmp_path):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path, 'test-job')
    store.submit(value)
    assert store.snapshot()['items'][0]['display']['phase'] == 'unknown'
    job(tmp_path, 'running', owner='other-owner')
    assert store.snapshot()['items'][0]['display']['phase'] == 'unknown'
    assert 'owner' in store.snapshot()['items'][0]['display']['error']
    store.update(value['id'], {'expected_revision':0,'availability':'preparation-needed',
                              'label':'Capturing','completed':3,'total':8,'unit':'captures'})
    assert store.snapshot()['items'][0]['display']['progress'] is None


def test_terminal_job_and_malformed_job_stay_explicit(tmp_path):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path, 'test-job')
    store.submit(value)
    path = job(tmp_path, 'failed')
    assert store.snapshot()['items'][0]['display']['phase']=='failed'
    (path/'status.json').write_text('[]')
    assert store.snapshot()['items'][0]['display']['phase']=='unknown'


def test_missing_queue_is_not_a_healthy_empty_smoke_source(tmp_path):
    root = tmp_path/'absent-queue'
    with pytest.raises(ValueError,match='queue'):
        SmokeRequests(root/'smoke-requests').snapshot()
    assert not root.exists()
    root.mkdir()
    assert SmokeRequests(root/'smoke-requests').snapshot()['items']==[]


def test_completed_counts_do_not_replace_operator_response(tmp_path):
    store = SmokeRequests(tmp_path/'smoke-requests')
    value = request(tmp_path)
    store.submit(value)
    store.update(value['id'], {'expected_revision':0,'availability':'prepared',
                              'label':'Capture complete','completed':8,'total':8,'unit':'captures'})
    assert store.snapshot()['items'][0]['display']['phase'] == 'operator-needed'
    store.respond(value['id'], 'Looks good')
    assert store.snapshot()['items'][0]['display']['phase'] == 'responded'
    with pytest.raises(SmokeRequestConflict):
        store.update(value['id'], {'expected_revision':1,'availability':'preparation-needed','label':'Restart'})
