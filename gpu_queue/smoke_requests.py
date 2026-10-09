"""Greenroom-owned interactive-smoke request and operator-response records."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from uuid import UUID
from . import admission, dispatch
from .models import JobRequest


SCHEMA = "gpu-greenroom.interactive-smoke.v1"


class SmokeRequestConflict(ValueError):
    """A request identity or state conflicts with an existing record."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("smoke request id must be a canonical UUID")
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("smoke request id must be a canonical UUID") from error
    if str(parsed) != value:
        raise ValueError("smoke request id must be a canonical UUID")
    return value


def _digest(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _timestamp(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return value


def _target(value):
    if not isinstance(value, dict) or value.get('kind') not in {'browser', 'terminal'}:
        raise ValueError('target must be a browser application or terminal')
    if value['kind'] == 'terminal':
        if set(value) != {'kind'}:
            raise ValueError('terminal identity is published separately through the local CLI')
        return {'kind': 'terminal'}
    url = value.get('url')
    if not isinstance(url, str):
        raise ValueError('browser target URL is required')
    parsed = urlsplit(url)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username
            or parsed.password or any(ord(char) < 32 for char in url)):
        raise ValueError('browser target URL must be HTTP(S), without credentials')
    return {'kind': 'browser', 'url': url}


def validate_request(value: object) -> dict:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"request schema must be {SCHEMA}")
    if value.get("kind") != "interactive-smoke":
        raise ValueError("only explicit interactive-smoke requests are supported")
    _canonical_id(value.get("id"))
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must identify the requesting agent")
    agent_id = source.get("agent_id")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("source agent_id must be a non-empty string")
    repo_root = source.get("repo_root")
    if not isinstance(repo_root, str) or not Path(repo_root).is_absolute():
        raise ValueError("source repo_root must be absolute")
    for field in ("title", "prompt", "availability_note"):
        field_value = value.get(field)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"missing {field}")
    if 'url' in value or 'target' not in value:
        _target({'kind': 'browser', 'url': value.get('url')})
    if value.get("availability") not in {"prepared", "preparation-needed", "unavailable"}:
        raise ValueError("invalid reported availability")
    job_id = value.get("job_id")
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()):
        raise ValueError("job_id must be a non-empty string when supplied")
    if job_id is not None and (not all(c.isalnum() or c in '_-' for c in job_id)):
        raise ValueError("job_id must be a single safe queue identity")
    if 'target' in value:
        _target(value['target'])
    if value.get('purpose', 'operator') not in {'operator', 'diagnostic'}:
        raise ValueError('purpose must be operator or diagnostic')
    return deepcopy(value)


def _progress(value):
    if not isinstance(value, dict):
        raise ValueError('progress must be an object')
    if value.get('availability') not in {'prepared', 'preparation-needed', 'unavailable'}:
        raise ValueError('invalid progress availability')
    if not isinstance(value.get('label'), str) or not value['label'].strip():
        raise ValueError('progress label must be nonblank')
    metrics = ('completed', 'total', 'unit')
    if any(key in value for key in metrics):
        if not all(key in value for key in metrics):
            raise ValueError('progress counts require completed, total and unit')
        if (type(value['completed']) is not int or type(value['total']) is not int
                or not 0 <= value['completed'] <= value['total'] or value['total'] <= 0
                or not isinstance(value['unit'], str) or not value['unit'].strip()):
            raise ValueError('invalid progress counts')
    if 'current_job_id' in value and (not isinstance(value['current_job_id'],str) or not value['current_job_id']
            or not all(c.isalnum() or c in '_-' for c in value['current_job_id'])):
        raise ValueError('current_job_id must be a single safe queue identity')
    return {key: value[key] for key in ('availability', 'label', *metrics, 'current_job_id') if key in value}


class SmokeRequests:
    """One canonical JSON record per request, protected across threads/processes."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().absolute()

    def path(self, identity: str) -> Path:
        return self.directory / f"{_canonical_id(identity)}.json"

    @contextmanager
    def _locked(self, identity: str):
        lock_path = self.path(identity).with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, record: dict) -> dict:
        destination = self.path(record["request"]["id"])
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp_name = tempfile.mkstemp(prefix=f".{destination.stem}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, destination)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return record

    def get(self, identity: str) -> dict:
        request_id = _canonical_id(identity)
        record = json.loads(self.path(request_id).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("smoke request record must be an object")
        request = validate_request(record.get("request"))
        if record.get("schema") != SCHEMA or request.get("id") != request_id:
            raise ValueError("smoke request schema or identity mismatch")
        if record.get("request_digest") != _digest(request):
            raise ValueError("smoke request digest mismatch")
        _timestamp(record.get("created_at"), "created_at")
        if record.get("status") not in {"operator-needed", "responded"}:
            raise ValueError("unknown smoke request status")
        response = record.get("response")
        if record["status"] == "operator-needed" and response is not None:
            raise ValueError("operator-needed request cannot contain a response")
        if record["status"] == "responded":
            if not isinstance(response, dict) or response.get("request_digest") != record["request_digest"]:
                raise ValueError("response is not bound to this request")
            if not isinstance(response.get("text"), str) or not response["text"].strip():
                raise ValueError("response text is missing")
            if response.get("actor") != {"kind": "unverified-caller", "id": None}:
                raise ValueError("response actor attribution must remain unverified")
            _timestamp(response.get("responded_at"), "responded_at")
        if 'progress' in record:
            _progress(record['progress'])
            if type(record['progress'].get('revision')) is not int or record['progress']['revision'] < 1:
                raise ValueError('invalid progress revision')
            _timestamp(record['progress'].get('updated_at'), 'updated_at')
        if 'presentation' in record:
            presentation = record['presentation']
            if not isinstance(presentation, dict) or presentation.get('purpose', 'operator') not in {'operator', 'diagnostic'}:
                raise ValueError('invalid smoke presentation')
            if 'target' in presentation:
                _target(presentation['target'])
        if 'session' in record:
            session = record['session']
            if (not isinstance(session, dict) or session.get('schema') != 'gpu-greenroom.smoke-session.v1'
                    or session.get('request_digest') != record['request_digest']
                    or session.get('phase') not in {'loading', 'operator-needed', 'interactive'}
                    or type(session.get('pane_id')) is not int or session['pane_id'] < 0
                    or type(session.get('pid')) is not int or session['pid'] <= 0
                    or not isinstance(session.get('tty_name'), str) or not session['tty_name'].startswith('/dev/')
                    or not isinstance(session.get('process_start_identity'), str) or not session['process_start_identity']
                    or not isinstance(session.get('label'), str) or not session['label'].strip()
                    or not isinstance(session.get('job_id'), str) or not session['job_id']
                    or not all(c.isalnum() or c in '_-' for c in session['job_id'])):
                raise ValueError('invalid terminal session record')
            _timestamp(session.get('observed_at'), 'session observed_at')
        if 'operator_command' in request:
            plan = record.get('prepared')
            if not isinstance(plan, dict) or plan.get('schema') != admission.PLAN_SCHEMA:
                raise ValueError('prepared operator command is missing')
            if plan.get('manifest_digest') != admission.digest(request['operator_command']):
                raise ValueError('prepared operator command manifest changed')
            job = JobRequest.from_json(json.dumps(plan['job_request']))
            validation = plan.get('validation')
            if (job.agent_id != request['source']['agent_id'] or job.repo_root != str(Path(request['source']['repo_root']).resolve())
                    or not isinstance(validation, dict) or validation.get('valid') is not True
                    or validation.get('request_digest') != admission.request_identity(job)
                    or job.params.get(admission.DIGEST_PARAM) != admission.digest(validation)):
                raise ValueError('prepared command owner or validation identity conflicts')
        return record

    def configure(self, identity, payload):
        if not isinstance(payload, dict) or not payload or set(payload) - {'target', 'purpose'}:
            raise ValueError('configuration accepts target and purpose only')
        if 'target' in payload:
            _target(payload['target'])
        if 'purpose' in payload and payload['purpose'] not in {'operator', 'diagnostic'}:
            raise ValueError('purpose must be operator or diagnostic')
        with self._locked(identity):
            record = self.get(identity)
            record['presentation'] = {**record.get('presentation', {}), **deepcopy(payload)}
            if 'target' in payload and payload['target']['kind'] != 'terminal':
                record.pop('session', None)
            return self._write(record)

    def _running_job(self, record):
        job_id = (record.get('activation') or {}).get('job_id') or (record.get('progress') or {}).get('current_job_id') or record['request'].get('job_id')
        if not job_id:
            raise ValueError('terminal session needs a running linked job')
        if not isinstance(job_id, str) or not job_id or not all(c.isalnum() or c in '_-' for c in job_id):
            raise ValueError('invalid running terminal job identity')
        root = self.directory.parent
        paths = [root / state / job_id for state in ('pending', 'running', 'done', 'failed', 'cancelled')
                 if (root / state / job_id).is_dir()]
        if len(paths) != 1 or paths[0].parent.name != 'running':
            raise ValueError('terminal session needs an unambiguous running linked job')
        path = paths[0]
        try:
            job = json.loads((path / 'request.json').read_text())
            state = json.loads((path / 'status.json').read_text())
        except FileNotFoundError as error:
            raise ValueError('terminal session needs a running linked job') from error
        if (not isinstance(job, dict) or not isinstance(state, dict)
                or job.get('job_id') != job_id or job.get('agent_id') != record['request']['source']['agent_id']
                or state.get('job_id') != job_id or state.get('status') != 'running'):
            raise ValueError('running terminal job owner or identity conflicts')
        return job_id

    def publish_session(self, identity, payload):
        from . import smoke_navigation
        if (not isinstance(payload, dict) or set(payload) - {'pane_id', 'pid', 'phase', 'label'}
                or payload.get('phase') not in {'loading', 'operator-needed', 'interactive'}
                or not isinstance(payload.get('label'), str) or not payload['label'].strip()):
            raise ValueError('session needs pane, process, phase and label')
        with self._locked(identity):
            record = self.get(identity)
            if record['status'] == 'responded':
                raise SmokeRequestConflict('smoke already has a response')
            job_id = self._running_job(record)
            observed = smoke_navigation.observe(payload.get('pane_id'), payload.get('pid'))
            record['session'] = {**observed, 'schema': 'gpu-greenroom.smoke-session.v1',
                                 'request_digest': record['request_digest'], 'job_id': job_id,
                                 'phase': payload['phase'], 'label': payload['label'], 'observed_at': _now()}
            record['presentation'] = {**record.get('presentation', {}), 'target': {'kind': 'terminal'}}
            return self._write(record)

    def destination(self, record):
        target = record.get('presentation', {}).get('target', record['request'].get('target'))
        if target is None:
            return {'kind': 'context', 'url': record['request']['url']}
        target = _target(target)
        if target['kind'] == 'terminal':
            session = record.get('session')
            if session:
                identity = {key: session[key] for key in ('request_digest', 'job_id', 'pane_id',
                            'pid', 'tty_name', 'process_start_identity')}
                return {**target, 'pane_id': session['pane_id'], 'identity_digest': _digest(identity)}
        return target

    def _verify_session(self, record):
        from . import smoke_navigation
        session = record.get('session')
        if (not isinstance(session, dict) or session.get('schema') != 'gpu-greenroom.smoke-session.v1'
                or session.get('request_digest') != record['request_digest']
                or session.get('phase') not in {'loading', 'operator-needed', 'interactive'}
                or session.get('job_id') != self._running_job(record)):
            raise ValueError('terminal session is missing or conflicts with the running job')
        smoke_navigation.verify(session)
        return session

    def focus(self, identity, request_digest, destination_digest):
        from . import smoke_navigation
        with self._locked(identity):
            record = self.get(identity)
            if record['status'] == 'responded':
                raise SmokeRequestConflict('smoke already has a response')
            destination = self.destination(record)
            if request_digest != record['request_digest'] or destination['kind'] != 'terminal':
                raise ValueError('terminal request identity changed')
            if not destination_digest or destination_digest != destination.get('identity_digest'):
                raise ValueError('terminal destination changed; refresh before opening')
            session = self._verify_session(record)
            return smoke_navigation.focus(session)

    @staticmethod
    def public_record(record):
        result = deepcopy(record)
        plan = result.pop('prepared', None)
        if 'operator_command' in result['request']:
            command = result['request']['operator_command']
            result['request']['operator_command'] = {'schema': command.get('schema'),
                'route_identity': command.get('route_identity'), 'agent_id': command.get('agent_id')}
            if plan:
                result['prepared_job_id'] = plan['job_request']['job_id']
        return result

    def start(self, identity, request_digest):
        from .queue import GPUQueue
        with self._locked(identity):
            record = self.get(identity)
            if request_digest != record['request_digest']:
                raise SmokeRequestConflict('smoke request changed; reread before starting')
            if record['status'] == 'responded' or 'prepared' not in record:
                raise SmokeRequestConflict('smoke has no pending operator-start command')
            if record.get('activation'):
                return record
            plan = record['prepared']
            planned = JobRequest.from_json(json.dumps(plan['job_request']))
            queue = GPUQueue(self.directory.parent)
            existing = [queue.queue_dir / state / planned.job_id
                        for state in ('pending', 'running', 'done', 'failed', 'cancelled')
                        if (queue.queue_dir / state / planned.job_id).is_dir()]
            if existing:
                if len(existing) != 1:
                    raise SmokeRequestConflict('prepared job identity is ambiguous')
                actual = JobRequest.from_json((existing[0] / 'request.json').read_text())
                if not admission.same_request(actual, planned):
                    raise SmokeRequestConflict('prepared job identity belongs to different work')
                path = existing[0]
            else:
                try:
                    path = admission.submit_prepared(queue, plan)
                except (ValueError, OSError) as error:
                    record['activation_error'] = {'phase': 'validation', 'error': str(error), 'observed_at': _now()}
                    self._write(record)
                    raise
            record.pop('activation_error', None)
            record['activation'] = {'schema': 'gpu-greenroom.operator-start.v1',
                'request_digest': request_digest, 'job_id': planned.job_id,
                'requested_at': _now(), 'job_path': str(path),
                'actor': {'kind': 'unverified-caller', 'id': None}}
            return self._write(record)

    def update(self, identity, value):
        payload = _progress(value)
        revision = value.get('expected_revision')
        if type(revision) is not int or revision < 0:
            raise ValueError('expected_revision must be a nonnegative integer')
        with self._locked(identity):
            record = self.get(identity)
            if record['status'] == 'responded':
                raise SmokeRequestConflict('smoke already has a response')
            previous = record.get('progress')
            if previous and payload == _progress(previous):
                return record
            actual = previous['revision'] if previous else 0
            if revision != actual:
                raise SmokeRequestConflict('progress revision changed; reread before updating')
            record['progress'] = {**payload, 'revision': actual + 1, 'updated_at': _now()}
            return self._write(record)

    def display(self, record):
        request = record['request']
        progress = record.get('progress')
        availability = progress['availability'] if progress else request['availability']
        result = {'phase': 'responded' if record['status'] == 'responded' else
                  {'prepared': 'operator-needed', 'preparation-needed': 'preparing',
                   'unavailable': 'blocked'}[availability],
                  'label': progress['label'] if progress else request['availability_note'],
                  'progress': progress, 'source_authority': 'caller-declared',
                  'created_at': record['created_at'], 'job_id': (progress or {}).get('current_job_id', request.get('job_id'))}
        result['section'] = 'history' if (record['status'] == 'responded' or
            record.get('presentation', {}).get('purpose', request.get('purpose')) == 'diagnostic') else 'active'
        if 'prepared' in record:
            if record.get('activation'):
                result['job_id'] = record['activation']['job_id']
                result['operator_start_at'] = record['activation']['requested_at']
            elif record.get('activation_error'):
                return {**result, 'phase': 'blocked', 'progress': None,
                        'error': record['activation_error']['error'], 'job_id': None}
            else:
                return {**result, 'phase': 'awaiting-start', 'job_id': None,
                        'label': 'Prepared; GPU not acquired'}
        if not result['job_id'] or record['status'] == 'responded':
            return result
        try:
            root, job_id = self.directory.parent, result['job_id']
            matches = [root/state/job_id for state in ('pending','running','done','failed','cancelled')
                       if (root/state/job_id).is_dir()]
            if len(matches) != 1:
                raise ValueError('linked job is missing or ambiguous')
            path = matches[0]
            state = json.loads((path/'status.json').read_text())
            job = json.loads((path/'request.json').read_text())
            if not isinstance(state, dict) or not isinstance(job, dict):
                raise ValueError('linked job records are malformed')
            if state.get('job_id') != job_id or state.get('status') != path.parent.name:
                raise ValueError('linked job identity or state conflicts with containment')
            if job.get('job_id') != job_id or job.get('agent_id') != request['source']['agent_id']:
                raise ValueError('linked job owner or identity conflicts with smoke request')
            stage = path.parent.name
            if stage != 'done':
                result['phase'] = {'pending':'waiting-gpu','running':'running',
                                   'failed':'failed','cancelled':'cancelled'}[stage]
            result['job_state'] = stage
            if stage in {'failed', 'cancelled'}:
                result.update(section='history', error=state.get('error_message'),
                              failure_phase=state.get('failure_phase'), exit_code=state.get('exit_code'),
                              progress=None, label='Execution failed' if stage == 'failed' else 'Job cancelled')
            if stage == 'running' and self.destination(record)['kind'] == 'terminal' and record.get('session'):
                session = self._verify_session(record)
                result.update(phase=session['phase'], label=session['label'], terminal_verified=True,
                              phase_reported_at=session['observed_at'])
            if stage == 'done' and self.destination(record)['kind'] == 'terminal':
                result.update(phase='awaiting-response', label='Terminal session ended; awaiting your response', progress=None)
            result['started_at'] = state.get('started_at')
            if stage == 'pending':
                result['queue_position'] = dispatch.pending_order(root).index(path)+1
        except (OSError, ValueError, TypeError, KeyError) as error:
            result.update(phase='unknown', progress=None, error=str(error))
        return result

    def snapshot(self):
        if not self.directory.parent.is_dir():
            raise ValueError('Greenroom queue root is unavailable')
        records, errors = self.scan()
        return {'schema': 'gpu-greenroom.smoke-request-list.v1', 'observed_at': _now(),
                'reader_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'queue_dir': str(self.directory.parent.resolve()),
                'items': [{**self.public_record(record), 'display': self.display(record),
                           'destination': self.destination(record)} for record in records],
                'errors': errors}

    def submit(self, value: object) -> tuple[dict, bool]:
        request = validate_request(value)
        with self._locked(request["id"]):
            try:
                existing = self.get(request["id"])
            except FileNotFoundError:
                existing = None
            if existing is not None:
                if existing["request_digest"] != _digest(request):
                    raise SmokeRequestConflict("request id already belongs to a different request")
                return existing, False
            record = {
                "schema": SCHEMA,
                "request": request,
                "request_digest": _digest(request),
                "created_at": _now(),
                "status": "operator-needed",
                "response": None,
            }
            if 'operator_command' in request:
                if request.get('job_id'):
                    raise ValueError('operator_command cannot also name an existing job')
                from .queue import GPUQueue
                plan = admission.prepare(GPUQueue(self.directory.parent), request['operator_command'])
                job = JobRequest.from_json(json.dumps(plan['job_request']))
                if (job.agent_id != request['source']['agent_id']
                        or job.repo_root != str(Path(request['source']['repo_root']).resolve())):
                    raise ValueError('operator command owner/repo must match the smoke source')
                plan['manifest_digest'] = admission.digest(request['operator_command'])
                record['prepared'] = plan
            return self._write(record), True

    def respond(self, identity: str, text: object) -> dict:
        request_id = _canonical_id(identity)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("response text must be a non-empty string")
        with self._locked(request_id):
            record = self.get(request_id)
            if record["status"] == "responded":
                if record["response"]["text"] != text:
                    raise SmokeRequestConflict("smoke request already has a different response")
                return record
            if self.display(record)['phase'] not in {'operator-needed', 'interactive', 'awaiting-response'}:
                raise SmokeRequestConflict('smoke is not available for operator response')
            record["response"] = {
                "request_digest": record["request_digest"],
                "text": text,
                "actor": {"kind": "unverified-caller", "id": None},
                "responded_at": _now(),
            }
            record["status"] = "responded"
            return self._write(record)

    def scan(self) -> tuple[list[dict], list[str]]:
        if not self.directory.exists():
            return [], []
        records, errors = [], []
        for path in sorted(self.directory.glob("*.json")):
            try:
                records.append(self.get(path.stem))
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
                errors.append(f"{path.name}: {error}")
        records.sort(key=lambda item: (item["status"] != "operator-needed", item["created_at"]))
        return records, errors
