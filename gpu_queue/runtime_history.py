"""Rebuildable command-wall history; estimates never authorize execution alone."""
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3
import statistics
import time

from . import admission
from .records import load_job_records

CAPABILITY = 'runtime-profile.v1'
PROFILE_SCHEMA = 'gpu-greenroom.runtime-profile.v1'


def definition_digest(profile):
    fields = ('schema','name','compatibility','backend','device','host','bindings','output_argv_indices')
    return admission.digest({k:profile[k] for k in fields if k in profile})


def _num(value):
    return value if type(value) in (int,float) and math.isfinite(value) else None


def identity(request, profile, *, check_files=True):
    if not isinstance(profile,dict) or profile.get('schema') != PROFILE_SCHEMA:
        raise ValueError('unsupported runtime profile')
    for name in ('name','compatibility','backend','device','host'):
        if not isinstance(profile.get(name),str) or not profile[name].strip():
            raise ValueError('runtime profile requires explicit ' + name)
    if not isinstance(profile.get('config'),dict) or not isinstance(profile.get('bindings'),list) or not profile['bindings']:
        raise ValueError('runtime profile requires config and source bindings')
    if profile['host'] != platform.node():
        raise ValueError('runtime profile host differs from effective worker host')
    bindings = {}
    for item in profile['bindings']:
        if not isinstance(item,dict) or not isinstance(item.get('path'),str) or not Path(item['path']).is_absolute():
            raise ValueError('invalid runtime profile binding')
        if not isinstance(item.get('sha256'),str) or len(item['sha256']) != 64:
            raise ValueError('invalid runtime profile binding hash')
        if item['path'] in bindings:
            raise ValueError('duplicate runtime profile binding')
        bindings[item['path']] = item['sha256']
        if check_files and admission.file_binding(item['path'])['sha256'] != item['sha256']:
            raise ValueError('runtime profile source binding changed')
    argv = list(request.command_argv or [])
    if not argv or not request.command_cwd:
        raise ValueError('runtime profiles require structured commands with explicit cwd')
    if check_files:
        required = [argv[0],*admission.entry_script(argv, request.command_cwd)]
        if any(path not in bindings for path in required):
            raise ValueError('executable and entry script need runtime profile bindings')
    indices = profile.get('output_argv_indices',[])
    if not isinstance(indices,list) or any(type(i) is not int or i <= 0 or i >= len(argv) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('invalid output argv indices')
    for i in indices:
        output = Path(request.output_dir).resolve()
        path = Path(argv[i])
        if not path.is_absolute() or not path.resolve().is_relative_to(output):
            raise ValueError('ignored output argument must be inside explicit output_dir')
        argv[i] = {'output_relative': str(path.resolve().relative_to(output))}
    # A run label is not a cost input. Every other argv/setting remains exact;
    # imports/weight-package compatibility is the producer's declared contract.
    fields = ('schema','name','compatibility','backend','device','host','config','bindings','output_argv_indices')
    value = {'profile':{k:profile[k] for k in fields if k in profile}, 'argv': argv, 'cwd': request.command_cwd,
             'env': request.command_env or {}, 'input_path': request.input_path,
             'performance_window':request.performance_window is not None,
             'command_timeout':request.command_timeout,'cooperative_checkpoint':request.cooperative_checkpoint,
             'params': {k:v for k,v in request.params.items() if k != admission.DIGEST_PARAM}}
    return admission.digest(value)


def capture(request, argv, cwd):
    if request.runtime_profile is None:
        return None
    if argv != request.command_argv or cwd != request.command_cwd:
        raise ValueError('effective command does not match runtime profile')
    return {'schema':'gpu-greenroom.runtime-observation.v1',
            'key':identity(request,request.runtime_profile), 'profile':request.runtime_profile,
            'effective_host':platform.node(), 'effective_argv':argv, 'effective_cwd':cwd,
            'authority':'worker-observed command envelope; backend/device are producer-declared',
            'source_unchanged':False}


def normalize(record):
    req, state, receipt = record.request, record.state, record.receipt
    start, end = _num(receipt.get('started_at')), _num(receipt.get('finished_at'))
    reasons = []
    if record.unreadable:
        reasons.append('unreadable_metadata')
    if record.state_dir != 'done' or receipt.get('status') != 'done' or receipt.get('exit_code') != 0 or receipt.get('failure_phase'):
        reasons.append('not_successful_completion')
    if start is None or end is None or end <= start:
        reasons.append('missing_zero_or_invalid_clock')
    for doc in (req,state,receipt):
        if doc.get('job_id') and doc['job_id'] != record.job_id:
            reasons.append('conflicting_job_identity')
    for field in ('started_at','finished_at'):
        if state.get(field) is not None and state[field] != receipt.get(field):
            reasons.append('conflicting_clock')
    if state.get('status') and state['status'] != record.state_dir:
        reasons.append('conflicting_state')
    observed = receipt.get('runtime_profile_observation')
    key = None
    if req.get('runtime_profile') is not None:
        try:
            from .models import JobRequest
            key = identity(JobRequest.from_json(json.dumps(req)),req['runtime_profile'],check_files=False)
        except (ValueError,KeyError,TypeError,OSError):
            pass
    verified = False
    if isinstance(observed,dict) and observed.get('schema') == 'gpu-greenroom.runtime-observation.v1':
        try:
            from .models import JobRequest
            request = JobRequest.from_json(json.dumps(req))
            expected = identity(request,observed['profile'],check_files=False)
            if (observed.get('key') == expected and observed.get('source_unchanged') is True
                    and observed.get('profile') == req.get('runtime_profile')
                    and observed.get('effective_argv') == (receipt.get('workload_argv') or receipt.get('effective_argv'))
                    and observed.get('effective_cwd') == receipt.get('effective_cwd')
                    and observed.get('effective_host') == observed['profile']['host']):
                key = expected
                verified = True
        except (ValueError,KeyError,TypeError,OSError):
            pass
    if req.get('runtime_profile') is not None and not verified:
        reasons.append('profile_observation_missing_or_unverified')
    performance = receipt.get('performance_window')
    command_seconds = None
    if (isinstance(performance,dict) and performance.get('job_id') == record.job_id
            and performance.get('status') == 'complete' and performance.get('workload_started') is True):
        value = _num(performance.get('command_wall_seconds'))
        if value is not None and value > 0:
            command_seconds = value
    return {'job_id':record.job_id,'status':record.state_dir,'successful':not reasons,
            'seconds':end-start if not reasons else None, 'started_at':start,'finished_at':end,
            'queue_wait_seconds':start-req['submitted_at'] if start is not None and _num(req.get('submitted_at')) is not None
                                   and start >= req['submitted_at'] else None,
            'profile_key':key,'reasons':reasons,
            'authority':'source-bound command wall; conditions unobserved' if key else
                        'legacy-command-wall; source compatibility unverified',
            'command_wall_seconds':command_seconds}


def generation(queue_dir):
    root = Path(queue_dir)
    return {state: (root/state).stat().st_mtime_ns if (root/state).is_dir() else None
            for state in ('done','failed','cancelled')}


def _connect(index):
    path = Path(index)
    path.parent.mkdir(parents=True,exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS observations (job_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, profile_key TEXT, payload TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS profile_lookup ON observations(profile_key)')
    return db


def _store(db, record):
    paths = [record.record_dir/name for name in ('request.json','status.json','receipt.json')]
    digests = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
    # The record reader and this fingerprint pass are distinct reads. A writer
    # moving between them must not attach fresh digests to an older success.
    for name,field in (('request.json','request'),('status.json','state'),('receipt.json','receipt')):
        path = record.record_dir/name
        try:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digests.get(str(path)) or json.loads(raw) != getattr(record,field):
                record.unreadable.append('changed_during_read:'+name)
        except FileNotFoundError:
            if getattr(record,field):
                record.unreadable.append('changed_during_read:'+name)
        except (ValueError,OSError):
            record.unreadable.append('unreadable:'+name)
    fingerprint = admission.digest({'files':digests,'state':record.state_dir,'unreadable':record.unreadable})
    row = {**normalize(record),'files':digests}
    db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?)',
               (record.job_id,fingerprint,row['profile_key'],json.dumps(row)))
    return fingerprint


def refresh(queue_dir, index):
    root = Path(queue_dir).resolve()
    before = generation(root)
    records = load_job_records(root)
    with closing(_connect(index)) as db, db:
        previous = db.execute("SELECT value FROM meta WHERE name='queue'").fetchone()
        if previous and json.loads(previous[0]) != str(root):
            raise ValueError('runtime history index belongs to another queue')
        old = dict(db.execute('SELECT job_id,fingerprint FROM observations'))
        changed = 0
        for record in records.values():
            if not record.terminal:
                continue
            paths = [record.record_dir/name for name in ('request.json','status.json','receipt.json')]
            digests = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}
            fingerprint = admission.digest({'files':digests,'state':record.state_dir,'unreadable':record.unreadable})
            if old.pop(record.job_id,None) == fingerprint:
                continue
            _store(db,record)
            changed += 1
        for job_id in old:
            db.execute('DELETE FROM observations WHERE job_id=?',(job_id,))
        after = generation(root)
        metadata = {'queue':str(root),'generation':before if before == after else None,'refreshed_at':time.time()}
        db.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)',[(k,json.dumps(v)) for k,v in metadata.items()])
        count = db.execute('SELECT count(*) FROM observations').fetchone()[0]
    return {'observations':count,'updated':changed,'removed':len(old),'current':before == after,
            'queue_dir':str(root),'index':str(Path(index).resolve())}


def record_completion(queue_dir,index,directory,previous_generation):
    """One receipt update; a previously stale cache needs explicit full refresh."""
    from .records import _read_record
    root,directory = Path(queue_dir).resolve(),Path(directory)
    if directory.parent.name not in ('done','failed','cancelled'):
        return
    before = generation(root)
    with closing(_connect(index)) as db, db:
        meta = {k:json.loads(v) for k,v in db.execute('SELECT name,value FROM meta')}
        if meta.get('queue') != str(root):
            raise ValueError('runtime history requires explicit refresh for this queue')
        _store(db,_read_record(directory,directory.parent.name))
        current = before if meta.get('generation') == previous_generation and generation(root) == before else None
        db.executemany('INSERT OR REPLACE INTO meta VALUES (?,?)',
                       [('generation',json.dumps(current)),('refreshed_at',json.dumps(time.time()))])


def estimate(queue_dir,index,request,profile,*,minimum_samples=3,max_seconds=30):
    result = {'schema':'gpu-greenroom.runtime-estimate.v1','eligible':False,'samples':0,
              'authority':'empirical command-wall screen; not calibrated confidence or execution authority',
              'conditions':'unobserved; foreground contention may invalidate duration expectation'}
    try:
        if type(minimum_samples) is not int or minimum_samples < 1 or _num(max_seconds) is None or max_seconds <= 0:
            raise ValueError('positive threshold and sample count required')
        key = identity(request,profile)
        result['profile_key'] = key
        path = Path(index).resolve()
        db = sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)
        try:
            metadata = {k:json.loads(v) for k,v in db.execute('SELECT name,value FROM meta')}
            if metadata.get('queue') != str(Path(queue_dir).resolve()) or metadata.get('generation') != generation(queue_dir):
                raise ValueError('history is stale or belongs to another queue; run history refresh')
            rows = [json.loads(row[0]) for row in db.execute('SELECT payload FROM observations WHERE profile_key=?',(key,))]
        finally:
            db.close()
        comparable = [r for r in rows if r['profile_key'] == key]
        for row in comparable:
            for file, expected in row['files'].items():
                if hashlib.sha256(Path(file).read_bytes()).hexdigest() != expected:
                    raise ValueError('cited observation changed; run history refresh')
        if generation(queue_dir) != metadata['generation']:
            raise ValueError('history changed during query; retry after refresh')
        successes = [r for r in comparable if r['successful']]
        result.update(samples=len(successes),excluded=len(comparable)-len(successes),
                      evidence=[r['job_id'] for r in comparable],refreshed_at=metadata['refreshed_at'])
        values = [r['seconds'] for r in successes]
        if values:
            result.update(median_seconds=statistics.median(values),minimum_seconds=min(values),maximum_seconds=max(values))
        result['eligible'] = len(values) >= minimum_samples and not result['excluded'] and max(values,default=float('inf')) <= max_seconds
        result['reason'] = ('observed compatible jobs satisfy configured screen' if result['eligible'] else
                            'insufficient compatible successes, failed observations, or observed duration exceeds threshold')
    except (ValueError,KeyError,TypeError,OSError,sqlite3.Error) as error:
        result['reason'] = str(error)
    return result
