"""Greenroom-owned interactive-smoke request and operator-response records."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from uuid import UUID


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
    for field in ("title", "prompt", "url", "availability_note"):
        field_value = value.get(field)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"missing {field}")
    url = urlsplit(value["url"])
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or url.username
        or url.password
        or any(ord(char) < 32 for char in value["url"])
    ):
        raise ValueError("smoke URL must be HTTP(S), without credentials or control characters")
    if value.get("availability") not in {"prepared", "preparation-needed", "unavailable"}:
        raise ValueError("invalid reported availability")
    job_id = value.get("job_id")
    if job_id is not None and (not isinstance(job_id, str) or not job_id.strip()):
        raise ValueError("job_id must be a non-empty string when supplied")
    return deepcopy(value)


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
        return record

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
