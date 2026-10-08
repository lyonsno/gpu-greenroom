"""CPU command admission and input-bound preflight, without GPU execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from . import dispatch, gc
from .models import JobRequest

CAPABILITY = 'command-preflight.v1'
SCHEMA = 'gpu-greenroom.command-validation.v1'
PLAN_SCHEMA = 'gpu-greenroom.prepared-command.v1'
DIGEST_PARAM = '_greenroom_validation_digest'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()).hexdigest()


def request_identity(request):
    value = asdict(request)
    value.pop('submitted_at')
    value['params'].pop(DIGEST_PARAM, None)
    return digest(value)


def same_request(left, right):
    return (request_identity(left) == request_identity(right)
            and left.params.get(DIGEST_PARAM) == right.params.get(DIGEST_PARAM))


def file_binding(path):
    path = Path(path).expanduser().absolute()
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(data)
    return {'path': str(path), 'sha256': sha.hexdigest(), 'size': path.stat().st_size}


def executable(argv, cwd, env):
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) and '\0' not in v for v in argv):
        raise ValueError('command argv must be a non-empty list of strings without NUL bytes')
    if not argv[0]:
        raise ValueError('command executable must be nonblank')
    name = argv[0]
    if '/' in name:
        path = Path(name) if Path(name).is_absolute() else Path(cwd) / name
        resolved = str(path.absolute())
    else:
        resolved = shutil.which(name, path=env.get('PATH', os.environ.get('PATH')))
    if not resolved or not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f'command executable is missing or not executable: {name}')
    return resolved


def entry_script(argv, cwd):
    name = Path(argv[0]).name
    python = re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', name) is not None
    node = name in {'node', 'nodejs'}
    shell = name in {'sh', 'bash', 'zsh'}
    if not (python or node or shell):
        return []
    index = 1
    simple_flags = {'-u', '-B', '-E', '-I', '-O', '-OO', '-q', '-s', '-S', '-v', '-b', '-bb'} if python else {'--no-warnings'}
    while index < len(argv):
        value = argv[index]
        if value in ({'-c'} if python or shell else {'-e', '--eval', '-p', '--print'}):
            if index + 1 >= len(argv):
                raise ValueError('inline interpreter invocation is missing its source')
            return []
        if python and value == '-m':
            raise ValueError('prevalidated module entry is unsupported; use a direct script and declare imported inputs')
        if value == '--':
            index += 1
            break
        if value in simple_flags or node and value.startswith('--input-type='):
            index += 1
            continue
        if python and value in {'-W', '-X'}:
            index += 2
            continue
        if value.startswith('-'):
            raise ValueError('unsupported prevalidated interpreter option; use a direct script or inline wrapper')
        break
    if index >= len(argv):
        raise ValueError('prevalidated interpreter invocation requires a script or inline source')
    path = Path(argv[index]).expanduser()
    if not path.is_absolute():
        path = Path(cwd) / path
    if not path.is_file():
        raise ValueError(f'entry script is missing: {path}')
    return [str(path.absolute())]


def normalize_manifest(value):
    if not isinstance(value, dict):
        raise ValueError('command manifest must be a JSON object')
    payload = deepcopy(value)
    unsupported = {'job_id', 'params', 'required_worker_capabilities', 'stdin'} & payload.keys()
    if unsupported:
        raise ValueError('unsupported command authority fields: ' + ', '.join(sorted(unsupported)))
    schema = payload.get('schema')
    if schema not in {'gpu-greenroom.command.v1', 'gpu-greenroom.command.v2', 'gpu-greenroom.command.v3'}:
        raise ValueError('command manifest schema must be gpu-greenroom.command.v1, v2 or v3')
    preflight = payload.get('preflight')
    if 'preflight' in payload and schema != 'gpu-greenroom.command.v3':
        raise ValueError('preflight requires gpu-greenroom.command.v3')
    service_class = payload.get('service_class')
    if service_class is not None and (schema == 'gpu-greenroom.command.v1' or service_class not in {'normal', 'quick'}):
        raise ValueError('service_class requires command.v2 or v3 and must be normal or quick')
    cooperative = payload.get('cooperative_checkpoint', False)
    if type(cooperative) is not bool or cooperative and schema == 'gpu-greenroom.command.v1':
        raise ValueError('cooperative_checkpoint requires command.v2 or v3 and a boolean')
    if cooperative and schema == 'gpu-greenroom.command.v3':
        raise ValueError('command.v3 cooperative continuation requires successor validation support; use the separate command.v2 checkpoint contract')
    owner = payload.get('agent_id')
    if owner is not None and (not isinstance(owner, str) or not owner.strip()):
        raise ValueError('agent_id must be a non-empty string when supplied')
    for key in ('repo_root', 'cwd', 'route_identity'):
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            raise ValueError(f'command manifest {key} must be a non-empty string')
    for key in ('repo_root', 'cwd'):
        payload[key] = str(Path(payload[key]).expanduser().resolve())
        if not Path(payload[key]).is_dir():
            raise ValueError(f'command {key} is not a directory: {payload[key]}')
    env = payload.get('env')
    if env is None:
        env = {}
    if not isinstance(env, dict) or not all(isinstance(k, str) and k and '=' not in k and '\0' not in k
                                           and isinstance(v, str) and '\0' not in v for k, v in env.items()):
        raise ValueError('command manifest env must map valid environment keys to strings')
    payload['env'] = env
    timeout = payload.get('timeout')
    if timeout is not None and (type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('command manifest timeout must be null or a positive finite number')
    output = payload.get('output_dir')
    if output is None:
        output = ''
    if not isinstance(output, str) or '\0' in output:
        raise ValueError('command manifest output_dir must be a string without NUL bytes')
    if output and Path(output).exists() and not Path(output).is_dir():
        raise ValueError('command output_dir must not be an existing non-directory')
    if payload.get('output_class') is not None and payload['output_class'] not in gc.CLASSES:
        raise ValueError('invalid command output_class')
    if not isinstance(payload.get('argv'), list):
        raise ValueError('command manifest argv must be a non-empty list of strings')
    payload['argv'] = list(payload['argv'])
    payload['argv'][0:1] = [executable(payload['argv'], payload['cwd'], env)]
    if preflight is not None:
        if not isinstance(preflight, dict) or not isinstance(preflight.get('inputs'), list):
            raise ValueError('preflight must provide argv and an explicit inputs list')
        if not all(isinstance(path, str) and Path(path).is_absolute() for path in preflight['inputs']):
            raise ValueError('preflight inputs must be absolute file paths')
        if not isinstance(preflight.get('argv'), list):
            raise ValueError('preflight argv must be a non-empty list of strings')
        preflight['argv'] = list(preflight['argv'])
        preflight['argv'][0:1] = [executable(preflight['argv'], payload['cwd'], env)]
        for path in preflight['inputs']:
            if not Path(path).is_file():
                raise ValueError(f'preflight input is missing or not a file: {path}')
    return {**payload, 'output_dir': output, 'timeout': timeout,
            'service_class': service_class, 'cooperative_checkpoint': cooperative, 'preflight': preflight}


def prepare(queue, manifest):
    try:
        payload = normalize_manifest(manifest)
    except (OSError, ValueError, TypeError) as error:
        directory = Path(tempfile.mkdtemp(prefix='rejected-', dir=_report_root(queue)))
        receipt = directory / 'validation.json'
        dispatch.atomic_write(receipt, {'schema': SCHEMA, 'valid': False, 'phase': 'input-normalization',
            'validated_at': time.time(), 'gpu_execution_authority': False, 'error': str(error)})
        raise ValueError(f'{error}; validation receipt: {receipt}') from error
    if payload['schema'] != 'gpu-greenroom.command.v3':
        raise ValueError('prepared commands require gpu-greenroom.command.v3')
    capabilities = ['structured-command.v1', 'structured-command.v2', CAPABILITY]
    if payload['service_class'] == 'quick':
        capabilities.append(dispatch.CAPABILITY)
    if payload['cooperative_checkpoint']:
        capabilities.append('checkpoint-continuation.v1')
    request = JobRequest(job_type='command', input_path='', agent_id=payload.get('agent_id'),
                         repo_root=payload['repo_root'], command_cwd=payload['cwd'],
                         command_argv=payload['argv'], command_env=payload['env'],
                         command_timeout=payload['timeout'], output_dir=payload['output_dir'],
                         output_class=payload.get('output_class'), route_identity=payload['route_identity'],
                         service_class=payload['service_class'], cooperative_checkpoint=payload['cooperative_checkpoint'],
                         required_worker_capabilities=capabilities)
    if not request.output_dir:
        request.output_dir = str(queue.queue_dir / 'outputs' / request.job_id)
    dispatch.classify(json.loads(request.to_json()), dispatch.policy(queue.queue_dir))
    report = validate(queue, request, payload['preflight'])
    request.params[DIGEST_PARAM] = digest(report)
    return {'schema': PLAN_SCHEMA, 'prepared_at': time.time(),
            'job_request': json.loads(request.to_json()), 'validation': report}


def validate(queue, request, preflight):
    directory = Path(tempfile.mkdtemp(prefix=request.job_id + '-', dir=_report_root(queue)))
    report = {'schema': SCHEMA, 'valid': False, 'phase': 'input-binding', 'validated_at': time.time(),
              'request_digest': request_identity(request), 'bound_scope': 'command-and-declared-inputs',
              'gpu_execution_authority': False, 'bindings': [], 'report_path': str(directory / 'validation.json')}
    try:
        paths = [request.command_argv[0]]
        paths += entry_script(request.command_argv, request.command_cwd)
        if preflight:
            paths += [preflight['argv'][0], *preflight['inputs']]
            paths += entry_script(preflight['argv'], request.command_cwd)
        report['bindings'] = [file_binding(path) for path in dict.fromkeys(paths)]
        if preflight:
            report['phase'] = 'application-preflight'
            result_path = directory / 'result.json'
            env = {k: v for k, v in os.environ.items() if not k.startswith('GPU_GREENROOM_')}
            env.update(request.command_env or {})
            env = {k: v for k, v in env.items() if not k.startswith('GPU_GREENROOM_')}
            env.update(GPU_GREENROOM_VALIDATION_MODE='cpu-only', GPU_GREENROOM_VALIDATION_OUTPUT=str(result_path))
            stdin = json.dumps({'schema': 'gpu-greenroom.preflight-input.v1',
                                'request_digest': report['request_digest'], 'request': json.loads(request.to_json())})
            with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
                result = subprocess.run(preflight['argv'], input=stdin.encode(), cwd=request.command_cwd,
                                        env=env, stdout=stdout, stderr=stderr)
            report['preflight'] = {'argv': preflight['argv'], 'cwd': request.command_cwd,
                                   'exit_code': result.returncode, 'result_path': str(result_path),
                                   'stdout_path': str(directory / 'stdout.log'), 'stderr_path': str(directory / 'stderr.log'),
                                   'mode_authority': 'declared-cpu-contract-not-hardware-observation'}
            response = json.loads(result_path.read_text())
            if not isinstance(response, dict) or response.get('schema') != 'gpu-greenroom.preflight-result.v1':
                raise ValueError('preflight did not return a typed validation result')
            if response.get('request_digest') != report['request_digest'] or response.get('mode') != 'cpu-only':
                raise ValueError('preflight result has wrong command identity or mode')
            report['preflight']['result'] = response
            if result.returncode != 0 or response.get('valid') is not True:
                raise ValueError(response.get('error') or 'application preflight failed')
        for previous in report['bindings']:
            if file_binding(previous['path']) != previous:
                raise ValueError('validation input changed during preflight')
        report.update(valid=True, phase='complete')
    except (OSError, ValueError, TypeError) as error:
        report['error'] = str(error)
        dispatch.atomic_write(directory / 'validation.json', report)
        raise ValueError(f"{error}; validation receipt: {directory / 'validation.json'}") from error
    dispatch.atomic_write(directory / 'validation.json', report)
    return report


def _report_root(queue):
    directory = queue.queue_dir / 'validation-reports'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    return directory


def assert_current(request, report):
    if not isinstance(report, dict) or report.get('schema') != SCHEMA or report.get('valid') is not True:
        raise ValueError('command validation is missing or invalid')
    if request.params.get(DIGEST_PARAM) != digest(report) or report.get('request_digest') != request_identity(request):
        raise ValueError('validated command identity changed')
    if not isinstance(report.get('bindings'), list) or not report['bindings']:
        raise ValueError('validated command input bindings are missing')
    for previous in report['bindings']:
        try:
            current = file_binding(previous['path'])
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise ValueError('validated input changed or disappeared') from error
        if current != previous:
            raise ValueError(f"validated input changed: {previous['path']}")


def submit_prepared(queue, plan):
    if not isinstance(plan, dict) or plan.get('schema') != PLAN_SCHEMA:
        raise ValueError('invalid prepared-command plan')
    request = JobRequest.from_json(json.dumps(plan['job_request']))
    assert_current(request, plan.get('validation'))
    request.submitted_at = time.time()
    return queue.submit(request, validation=plan['validation'])
