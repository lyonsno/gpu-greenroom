import copy
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

import pytest

from gpu_queue.models import JobRequest
from gpu_queue.queue import GPUQueue


def test_history_refresh_cli_is_read_only_and_produces_index(tmp_path):
    queue = GPUQueue(tmp_path/'queue')
    result = subprocess.run([sys.executable, '-m', 'gpu_queue.cli', '--queue-dir', str(queue.queue_dir),
                             'history', 'refresh', '--index', str(tmp_path/'history.sqlite')],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['observations'] == 0
    assert not list((queue.queue_dir/'pending').iterdir())


def recipe(tmp_path):
    script = tmp_path/'work.py'
    script.write_text("import pathlib,sys,time; time.sleep(.05); pathlib.Path(sys.argv[1]).mkdir(exist_ok=True)\n")
    profile = {'schema': 'gpu-greenroom.runtime-profile.v1', 'name': 'fixture/case',
               'compatibility': 'v1', 'backend': 'python-cpu', 'device': 'cpu', 'host': platform.node(),
               'config': {'units': 1}, 'output_argv_indices': [2],
               'bindings': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in (script,Path(sys.executable))]}
    request = JobRequest(job_type='command', input_path='', agent_id='fixture-owner', repo_root=str(tmp_path),
                         command_cwd=str(tmp_path), command_argv=[sys.executable, str(script), str(tmp_path/'out')],
                         output_dir=str(tmp_path/'out'))
    return request, profile


def test_profile_identity_keeps_cost_settings_and_source_but_not_output_name(tmp_path):
    from gpu_queue.runtime_history import identity
    request, profile = recipe(tmp_path)
    key = identity(request, profile)
    assert identity(request,{**profile,'future_additive_label':'friendly'}) == key
    request.output_dir = str(tmp_path/'other')
    request.command_argv[2] = request.output_dir
    request.route_identity = 'arbitrary-run-023'
    assert identity(request, profile) == key
    request.performance_window = 'operator acknowledgement'
    assert identity(request, profile) != key
    request.performance_window = None
    for field, value in [('backend', 'fallback'), ('device', 'gpu'), ('compatibility', 'v2'), ('config', {'units': 2})]:
        changed = {**profile, field: value}
        assert identity(request, changed) != key
    request.command_argv.append('--more-work')
    assert identity(request, profile) != key
    Path(profile['bindings'][0]['path']).write_text('changed')
    with pytest.raises(ValueError, match='binding'):
        identity(request, profile)


def test_recorded_profile_history_drives_quick_and_invalidates_on_failure_or_staleness(tmp_path):
    from gpu_queue import runtime_history as history
    request, profile = recipe(tmp_path)
    queue = GPUQueue(tmp_path/'queue')
    index = tmp_path/'history.sqlite'
    queue.configure_dispatch({'schema': 'gpu-greenroom.dispatch-policy.v3', 'mode': 'two-class',
                              'quick_routes': {}, 'quick_job_types': {}, 'runtime_history': {
                                  'index': str(index), 'profiles': {'fixture/case': {
                                      'profile_digest': history.definition_digest(profile),
                                      'max_seconds': 30, 'minimum_samples': 3,
                                      'evidence': 'explicit CPU conformance fixture; not GPU calibration'}}}}, owner='fixture-operator')
    history.refresh(queue.queue_dir, index)
    for i in range(3):
        sample = copy.deepcopy(request)
        sample.job_id = f'sample{i}'
        sample.runtime_profile = profile
        queue.submit(sample)
        assert queue.run_one({})
        assert (queue.queue_dir/'done'/sample.job_id).exists()
    forecast = history.estimate(queue.queue_dir, index, request, profile, minimum_samples=3, max_seconds=30)
    assert forecast['eligible'] is True, forecast
    assert forecast['samples'] == 3
    next_request = copy.deepcopy(request)
    next_request.job_id = 'new'
    next_request.runtime_profile = profile
    queue.submit(next_request)
    from gpu_queue.dispatch import classify, policy
    value = json.loads((queue.queue_dir/'pending/new/request.json').read_text())
    assert classify(value, policy(queue.queue_dir), queue_dir=queue.queue_dir)[0] == 'quick'
    # A newly written terminal record by another/older writer makes the cache stale.
    foreign = queue.queue_dir/'failed/foreign'
    foreign.mkdir()
    assert not history.estimate(queue.queue_dir,index,request,profile,minimum_samples=3,max_seconds=30)['eligible']
    history.refresh(queue.queue_dir,index)
    assert history.estimate(queue.queue_dir,index,request,profile,minimum_samples=3,max_seconds=30)['eligible']
    # Altering a cited receipt must not preserve a cached eligibility claim.
    path = queue.queue_dir/'done/sample0/receipt.json'
    value = json.loads(path.read_text())
    value.update(status='failed',exit_code=1,failure_phase='execution')
    path.write_text(json.dumps(value))
    assert not history.estimate(queue.queue_dir,index,request,profile,minimum_samples=3,max_seconds=30)['eligible']


@pytest.mark.parametrize('field,value', [('finished_at', 0), ('finished_at', None), ('exit_code', 1),
                                       ('status', 'failed'), ('job_id', 'other')])
def test_normalizer_does_not_turn_bad_receipts_into_short_successes(tmp_path,field,value):
    from gpu_queue.runtime_history import normalize
    from gpu_queue.records import JobRecord
    receipt = dict(job_id='x',status='done',exit_code=0,started_at=10,finished_at=20,
                   effective_argv=['program'])
    receipt[field] = value
    observation = normalize(JobRecord('x','done',request={'job_id':'x'},receipt=receipt))
    assert observation['successful'] is False


def test_failed_prediction_and_unknown_fields_remain_visible(tmp_path):
    from gpu_queue.runtime_history import normalize
    from gpu_queue.records import JobRecord
    receipt = dict(job_id='x',status='done',exit_code=0,started_at=10,finished_at=20,
                   future_addition={'a':1},effective_argv=['program'])
    row = normalize(JobRecord('x','done',request={'job_id':'x'},receipt=receipt))
    assert row['seconds'] == 10
    assert row['profile_key'] is None
    assert row['authority'] == 'legacy-command-wall; source compatibility unverified'


def test_finite_unit_adapter_is_available_without_changing_direct_callers():
    result = subprocess.run([sys.executable,'-m','gpu_queue.cli','history','--help'],capture_output=True,text=True)
    assert result.returncode == 0
    from gpu_queue.cooperative import run_units
    results = run_units([1,2],perform=lambda unit: {'square':unit*unit},
                        quiesce=lambda: True,continuation=lambda checkpoint: None,
                        on_submitted=lambda request: None)
    assert results == [{'square':1},{'square':4}]


def test_profiled_quick_runs_between_real_batch_units_under_one_mutex(tmp_path):
    import os
    import threading
    import time
    from gpu_queue import runtime_history as history
    request,profile = recipe(tmp_path)
    queue = GPUQueue(tmp_path/'queue')
    index = tmp_path/'history.sqlite'
    queue.configure_dispatch({'schema':'gpu-greenroom.dispatch-policy.v3','mode':'two-class',
        'quick_routes':{},'quick_job_types':{},'runtime_history':{'index':str(index),'profiles':{
            profile['name']:{'minimum_samples':3,'max_seconds':30,'profile_digest':history.definition_digest(profile),
                             'evidence':'native CPU conformance'}}}},owner='fixture')
    history.refresh(queue.queue_dir,index)
    for i in range(3):
        sample = copy.deepcopy(request)
        sample.job_id = 'training'+str(i)
        sample.runtime_profile = profile
        queue.submit(sample)
        queue.run_one({})
    script = tmp_path/'batch.py'
    fifo = tmp_path/'continue'
    os.mkfifo(fifo)
    script.write_text('''import json,os,pathlib,sys
from gpu_queue.cooperative import run_units
from gpu_queue.models import JobRequest
def perform(unit):
    pathlib.Path(str(unit)+'.result').write_text(str(unit*unit))
    if unit==1:
        with open(sys.argv[1]) as stream: stream.read()
    return {'unit':unit,'artifact':str(pathlib.Path(str(unit)+'.result').absolute())}
def successor(path):
    return JobRequest(job_type='command',input_path='',agent_id='fixture-batch',repo_root=os.getcwd(),
        command_cwd=os.getcwd(),route_identity='fixture/batch',command_argv=[sys.executable,__file__,*sys.argv[1:]],
        command_env={'PYTHONPATH':sys.argv[2]},cooperative_checkpoint=True)
results=run_units([1,2],perform=perform,quiesce=lambda:True,continuation=successor,
    on_submitted=lambda request:pathlib.Path('registered-successor').write_text(request.job_id))
pathlib.Path('batch-results.json').write_text(json.dumps(results))
''')
    root = str(Path(__file__).resolve().parent.parent)
    batch = JobRequest(job_type='command',input_path='',agent_id='fixture-batch',repo_root=str(tmp_path),
        command_cwd=str(tmp_path),route_identity='fixture/batch',command_argv=[sys.executable,str(script),str(fifo),root],
        command_env={'PYTHONPATH':root},cooperative_checkpoint=True)
    queue.submit(batch)
    worker = threading.Thread(target=queue.run_one,args=({},))
    worker.start()
    while worker.is_alive() and (not (tmp_path/'1.result').exists() or (tmp_path/'1.result').read_text() != '1'):
        time.sleep(.01)
    try:
        assert (tmp_path/'1.result').read_text() == '1'
        quick = copy.deepcopy(request)
        quick.job_id = 'quick'
        quick.runtime_profile = profile
        queue.submit(quick)
    finally:
        try:
            fd = os.open(fifo,os.O_WRONLY|os.O_NONBLOCK)
            os.write(fd,b'continue')
            os.close(fd)
        except OSError:
            pass
        worker.join()
    parent = queue.queue_dir/'done'/batch.job_id
    assert parent.is_dir(), list((queue.queue_dir/'failed').iterdir())
    handoff = json.loads((parent/'checkpoint-handoff.json').read_text())
    assert handoff['next_job_id'] == (tmp_path/'registered-successor').read_text()
    assert not (tmp_path/'2.result').exists()
    assert queue.run_one({})
    receipt = json.loads((queue.queue_dir/'done/quick/receipt.json').read_text())
    assert receipt['dispatch']['service_class'] == 'quick'
    assert not (tmp_path/'2.result').exists()
    assert queue.run_one({})
    assert (tmp_path/'2.result').read_text() == '4'
    assert [r['unit'] for r in json.loads((tmp_path/'batch-results.json').read_text())] == [1,2]
    assert not list((queue.queue_dir/'running').iterdir())


def test_profile_field_cannot_be_silently_dropped_by_legacy_manifest_cli(tmp_path):
    from gpu_queue.admission import normalize_manifest
    request,profile = recipe(tmp_path)
    manifest = {'schema':'gpu-greenroom.command.v2','argv':request.command_argv,
        'repo_root':request.repo_root,'cwd':request.command_cwd,'route_identity':'fixture',
        'runtime_profile':profile,'output_dir':request.output_dir}
    with pytest.raises(ValueError,match='requires command.v4'):
        normalize_manifest(manifest)
    assert normalize_manifest({**manifest,'schema':'gpu-greenroom.command.v4'})['runtime_profile'] == profile


def test_unobserved_or_failed_profile_is_not_a_successful_duration(tmp_path):
    from gpu_queue.runtime_history import normalize,identity
    from gpu_queue.records import JobRecord
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    record = JobRecord(request.job_id,'failed',request=json.loads(request.to_json()),receipt={
        'job_id':request.job_id,'status':'failed','exit_code':1,'started_at':10,'finished_at':11})
    row = normalize(record)
    assert row['profile_key'] == identity(request,profile)
    assert row['successful'] is False
    assert 'profile_observation_missing_or_unverified' in row['reasons']


def test_old_worker_cannot_dispatch_profile_policy_and_manual_move_preserves_it(tmp_path):
    import uuid
    from gpu_queue.queue import worker_identity
    from gpu_queue import runtime_history
    queue = GPUQueue(tmp_path/'queue')
    index = tmp_path/'history.sqlite'
    queue.configure_dispatch({'schema':'gpu-greenroom.dispatch-policy.v3','mode':'two-class',
        'quick_routes':{},'quick_job_types':{},'runtime_history':{'index':str(index),'profiles':{}}},owner='fixture')
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    directory = queue.submit(request)
    old = worker_identity()
    old['capabilities'] = [c for c in old['capabilities'] if c != runtime_history.CAPABILITY]
    assert not queue.run_one({},claimant=old)
    queue.override_dispatch(job_id=request.job_id,service_class='quick',
        request_digest=hashlib.sha256((directory/'request.json').read_bytes()).hexdigest(),
        request_id=str(uuid.uuid4()),requested_by='fixture-console')
    from gpu_queue.dispatch import policy
    config = policy(queue.queue_dir)
    assert config['schema'] == 'gpu-greenroom.dispatch-policy.v3'
    assert config['runtime_history']['index'] == str(index)


def test_finite_unit_resume_rejects_changed_case_list(monkeypatch):
    from gpu_queue import cooperative
    monkeypatch.setattr(cooperative,'load_checkpoint',lambda:{'unit_list_sha256':'wrong','next_unit':1,'results':[{}]})
    with pytest.raises(ValueError,match='does not match'):
        cooperative.run_units([1,2],perform=lambda unit:{},quiesce=lambda:True,
            continuation=lambda checkpoint:None,on_submitted=lambda request:None)


def test_failed_or_slow_observation_demotes_and_full_refresh_is_idempotent(tmp_path):
    from gpu_queue import runtime_history as history
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    queue = GPUQueue(tmp_path/'queue')
    queue.submit(request)
    queue.run_one({})
    index = tmp_path/'history.sqlite'
    assert history.refresh(queue.queue_dir,index)['updated'] == 1
    assert history.refresh(queue.queue_dir,index)['updated'] == 0
    assert history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)['eligible']
    path = queue.queue_dir/'done'/request.job_id/'receipt.json'
    original = json.loads(path.read_text())
    for changed in ({**original,'finished_at':original['started_at']+100},
                    {**original,'exit_code':1,'status':'failed','failure_phase':'execution'}):
        path.write_text(json.dumps(changed))
        status_path = path.parent/'status.json'
        state = json.loads(status_path.read_text())
        state['finished_at'] = changed['finished_at']
        status_path.write_text(json.dumps(state))
        history.refresh(queue.queue_dir,index)
        assert not history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)['eligible']


def test_incremental_completion_updates_only_own_receipt_and_cannot_launder_stale_index(tmp_path):
    from gpu_queue import runtime_history as history
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    queue = GPUQueue(tmp_path/'queue')
    index = tmp_path/'history.sqlite'
    history.refresh(queue.queue_dir,index)
    before = history.generation(queue.queue_dir)
    queue.submit(request)
    queue.run_one({})
    directory = queue.queue_dir/'done'/request.job_id
    history.record_completion(queue.queue_dir,index,directory,before)
    assert history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)['eligible']
    (queue.queue_dir/'failed/other').mkdir()
    history.record_completion(queue.queue_dir,index,directory,history.generation(queue.queue_dir))
    assert not history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)['eligible']


def test_performance_operator_wait_is_not_substituted_for_command_clock():
    from gpu_queue import runtime_history
    from gpu_queue.records import JobRecord
    receipt = {'job_id':'x','status':'done','exit_code':0,'started_at':10,'finished_at':100,
        'performance_window':{'job_id':'x','status':'complete','workload_started':True,'command_wall_seconds':1}}
    row = runtime_history.normalize(JobRecord('x','done',request={'job_id':'x'},receipt=receipt))
    assert row['seconds'] == 90
    assert row['command_wall_seconds'] == 1
    receipt['performance_window']['job_id'] = 'wrong'
    assert runtime_history.normalize(JobRecord('x','done',request={'job_id':'x'},receipt=receipt))['command_wall_seconds'] is None


def test_observed_installed_receipts_preserve_clocks_and_legacy_authority():
    from gpu_queue.runtime_history import normalize
    from gpu_queue.records import JobRecord
    fixture = json.loads((Path(__file__).parent/'fixtures/runtime-history-observed.json').read_text())
    rows = [normalize(JobRecord(**value)) for value in fixture['records']]
    assert rows[0]['seconds'] == pytest.approx(14.131963014602661)
    assert rows[1]['seconds'] == pytest.approx(13.93107295036316)
    assert rows[1]['command_wall_seconds'] == pytest.approx(.15263054200113402)
    assert rows[2]['successful'] is False
    assert all(row['profile_key'] is None for row in rows)


def manifest_for(request,profile):
    return {'schema':'gpu-greenroom.command.v4','agent_id':request.agent_id,
        'argv':request.command_argv,'cwd':request.command_cwd,'repo_root':request.repo_root,
        'route_identity':'fixture/manifest','output_dir':request.output_dir,'runtime_profile':profile}


def test_prepared_profile_keeps_valid_identity_through_submission_and_claim(tmp_path):
    from gpu_queue import admission,runtime_history
    request,profile = recipe(tmp_path)
    queue = GPUQueue(tmp_path/'queue')
    plan = admission.prepare(queue,manifest_for(request,profile))
    assert runtime_history.CAPABILITY in plan['job_request']['required_worker_capabilities']
    directory = admission.submit_prepared(queue,plan)
    submitted = JobRequest.from_json((directory/'request.json').read_text())
    admission.assert_current(submitted,plan['validation'])
    assert queue.run_one({})
    assert (queue.queue_dir/'done'/directory.name).is_dir()


@pytest.mark.parametrize('setting,value',[('timeout',120),('performance_window','CPU fixture, no operator dialog launched'),
                                         ('cooperative_checkpoint',True)])
def test_history_cli_and_direct_query_use_identical_manifest_settings(tmp_path,setting,value):
    from gpu_queue import runtime_history as history
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    queue = GPUQueue(tmp_path/'queue')
    queue.submit(request)
    queue.run_one({})
    index = tmp_path/'history.sqlite'
    history.refresh(queue.queue_dir,index)
    manifest = {**manifest_for(request,profile),setting:value}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    setattr(request,{'timeout':'command_timeout'}.get(setting,setting),value)
    direct = history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)
    cli = subprocess.run([sys.executable,'-m','gpu_queue.cli','--queue-dir',str(queue.queue_dir),
        'history','estimate','--index',str(index),'--manifest',str(path),'--minimum-samples','1'],capture_output=True,text=True)
    assert cli.returncode == 0, cli.stderr
    returned = json.loads(cli.stdout)
    assert returned['profile_key'] == direct['profile_key']
    assert returned['eligible'] is direct['eligible'] is False


def test_profiled_cli_with_application_preflight_submits_and_runs(tmp_path):
    request,profile = recipe(tmp_path)
    preflight = tmp_path/'preflight.py'
    preflight.write_text('''import json,os,pathlib,sys
value=json.load(sys.stdin)
pathlib.Path(os.environ['GPU_GREENROOM_VALIDATION_OUTPUT']).write_text(json.dumps({
 'schema':'gpu-greenroom.preflight-result.v1','request_digest':value['request_digest'],
 'mode':'cpu-only','valid':True}))
''')
    manifest = {**manifest_for(request,profile),'preflight':{'argv':[sys.executable,str(preflight)],'inputs':[str(preflight)]}}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    queue = GPUQueue(tmp_path/'queue')
    result = subprocess.run([sys.executable,'-m','gpu_queue.cli','--queue-dir',str(queue.queue_dir),
        'submit-command','--manifest',str(path)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    returned = json.loads(result.stdout)
    assert returned['status'] == 'submitted'
    assert queue.run_one({})
    assert (queue.queue_dir/'done'/returned['job_id']).is_dir()


def test_history_cli_recognizes_matching_nondefault_timeout_history(tmp_path):
    from gpu_queue import runtime_history as history
    request,profile = recipe(tmp_path)
    request.runtime_profile = profile
    request.command_timeout = 120
    queue = GPUQueue(tmp_path/'queue')
    queue.submit(request)
    queue.run_one({})
    index = tmp_path/'history.sqlite'
    history.refresh(queue.queue_dir,index)
    manifest = {**manifest_for(request,profile),'timeout':120}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest))
    result = subprocess.run([sys.executable,'-m','gpu_queue.cli','--queue-dir',str(queue.queue_dir),
        'history','estimate','--index',str(index),'--manifest',str(path),'--minimum-samples','1'],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    returned = json.loads(result.stdout)
    direct = history.estimate(queue.queue_dir,index,request,profile,minimum_samples=1,max_seconds=30)
    assert returned['profile_key'] == direct['profile_key']
    assert returned['eligible'] is direct['eligible'] is True
