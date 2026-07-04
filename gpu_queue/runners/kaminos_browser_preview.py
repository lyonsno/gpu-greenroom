"""Kaminos browser WebGPU preview runner.

The initial Greenroom-owned mode is deliberately fixture/no-model. It exercises
queue custody, receipt identity, result persistence, and Kaminos ingestion
without launching Chrome/WebGPU or touching ML memory.
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import http.client
import json
import mimetypes
import os
import secrets
import socket
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


RESULT_SCHEMA = "kaminos.webgpu-route-result.v0"
RECEIPT_SCHEMA = "kaminos.webgpu-route-receipt.v0"
DEFAULT_ROUTE_ID = "moge.depth-normal.webgpu-local.v0"
MODEL_ID = "Ruicheng/moge-2-vitl-normal"


_PNG_1X1_TRANSPARENT = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB"
    "/6XgmWQAAAAASUVORK5CYII="
)
_PNG_1X1_BLACK = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR42mMAAAAAAQAB"
    "DQottAAAAABJRU5ErkJggg=="
)
_PNG_1X1_WHITE = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAAC0lEQVR42mP8/x8AAwMB"
    "/6XgmWQAAAAASUVORK5CYII="
)
_COMMON_CHROME_PATHS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat().replace("+00:00", "Z")


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _safe_filename(route_id: str, request_id: str) -> str:
    raw_name = f"{route_id}__{request_id}"
    safe_name = "".join(
        char if char.isalnum() or char in ".-_" else "-"
        for char in raw_name
    ).strip(".-_")
    return f"{safe_name or uuid.uuid4().hex}.json"


def _parse_json_object(text: str | None) -> dict[str, Any]:
    if not text or text.strip() in {"", "{}"}:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {
            "kind": "unparsed",
            "raw": text,
            "parseError": "source identity was not valid JSON",
        }
    return value if isinstance(value, dict) else {"kind": "non-object", "value": value}


def _source_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _build_browser_route_row(
    *,
    source_path: Path,
    route_id: str,
    request_id: str,
    job_id: str,
    source_identity: dict[str, Any],
    module_base_url: str,
) -> dict[str, Any]:
    identity = dict(source_identity or {})
    identity.setdefault("kind", "file")
    identity.setdefault("label", source_path.name)
    identity.setdefault("role", "source-image")
    identity["url"] = _source_data_url(source_path)
    input_source_kind = identity.get("kind") or "file"
    route_config = {
        "source": "gpu-greenroom.kaminos-browser-preview-runner",
        "producer": "greenroom-isolated-browser",
        "evidenceMode": "browser",
        "sourceImageIdentity": identity,
        "inputSourceKind": input_source_kind,
        "sourceSelectionMode": input_source_kind,
        "moduleBaseUrl": module_base_url or None,
        "greenroomJobId": job_id,
    }
    return {
        "schema": "kaminos.route-provider-row.v0",
        "provider": "gpu-greenroom",
        "job_id": job_id,
        "route_job": {
            "schema": "kaminos.route-job.v0",
            "id": job_id,
            "routeId": route_id,
            "executor": {
                "kind": "browser-webgpu",
                "id": "gpu-greenroom-isolated-browser",
                "backendKind": "webgpu-local",
                "workerModule": "webgpu-inference-kit/routes/moge-worker.js",
            },
            "intent": "preview",
            "priorityClass": "preview",
            "status": "running",
            "metadata": {
                "routeConfig": route_config,
                "sourceImage": identity,
            },
        },
        "routeConfig": route_config,
        "requestId": request_id,
    }


def _find_chrome(chrome_path: str = "") -> str:
    candidates = [
        chrome_path,
        os.environ.get("GPU_GREENROOM_CHROME_PATH", ""),
        *_COMMON_CHROME_PATHS,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser())
    raise FileNotFoundError("Chrome/Chromium not found; pass --chrome-path or set GPU_GREENROOM_CHROME_PATH")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_kaminos_server(args: argparse.Namespace, output_dir: Path, result_dir: Path) -> tuple[subprocess.Popen, str]:
    kaminos_root = Path(args.kaminos_root or os.path.expanduser("~/dev/kaminos")).expanduser()
    serve_py = kaminos_root / "serve.py"
    if not serve_py.exists():
        raise FileNotFoundError(f"Kaminos serve.py not found: {serve_py}")
    port = _free_port()
    env = {
        **os.environ,
        "KAMINOS_BROWSER_WEBGPU_ROUTE_RESULTS_DIR": str(result_dir),
    }
    if args.module_base_url:
        env["KAMINOS_MOGE_WEBGPU_MODULE_BASE_URL"] = args.module_base_url
    stdout = (output_dir / "kaminos-server.stdout.log").open("w")
    stderr = (output_dir / "kaminos-server.stderr.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, str(serve_py), str(port)],
        cwd=str(kaminos_root),
        env=env,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    base_url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Kaminos server exited early with code {proc.returncode}")
        try:
            _http_json(base_url, "/api/runtime-config")
            return proc, base_url
        except Exception:
            time.sleep(0.1)
    raise TimeoutError(f"Kaminos server did not answer at {base_url}")


def _http_json(base_url: str, path: str) -> Any:
    parsed = urlparse(base_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=5)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        data = response.read()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} for {path}: {data[:200]!r}")
        return json.loads(data.decode("utf-8"))
    finally:
        conn.close()


class _CdpConnection:
    def __init__(self, websocket_url: str):
        parsed = urlparse(websocket_url)
        if parsed.scheme != "ws":
            raise ValueError(f"Only ws:// CDP URLs are supported: {websocket_url}")
        self._host = parsed.hostname or "127.0.0.1"
        self._port = parsed.port or 80
        self._path = parsed.path
        if parsed.query:
            self._path = f"{self._path}?{parsed.query}"
        self._socket = socket.create_connection((self._host, self._port), timeout=10)
        self._next_id = 1
        self._handshake()

    def close(self) -> None:
        try:
            self._socket.close()
        except OSError:
            pass

    def _handshake(self) -> None:
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {self._path} HTTP/1.1\r\n"
            f"Host: {self._host}:{self._port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response:
            response += self._socket.recv(4096)
        if b" 101 " not in response.split(b"\r\n", 1)[0]:
            raise RuntimeError(f"CDP WebSocket handshake failed: {response[:200]!r}")

    def send(self, method: str, params: dict[str, Any] | None = None, timeout_s: float = 30) -> dict[str, Any]:
        message_id = self._next_id
        self._next_id += 1
        self._send_json({"id": message_id, "method": method, "params": params or {}})
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            payload = self._recv_json(timeout_s=max(0.1, deadline - time.time()))
            if payload.get("id") != message_id:
                continue
            if "error" in payload:
                raise RuntimeError(f"CDP {method} failed: {payload['error']}")
            return payload.get("result") or {}
        raise TimeoutError(f"Timed out waiting for CDP method {method}")

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        mask = secrets.token_bytes(4)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
        self._socket.sendall(bytes(header) + mask + masked)

    def _recv_json(self, timeout_s: float) -> dict[str, Any]:
        self._socket.settimeout(timeout_s)
        first = self._read_exact(2)
        opcode = first[0] & 0x0F
        length = first[1] & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        masked = bool(first[1] & 0x80)
        mask = self._read_exact(4) if masked else b""
        data = self._read_exact(length)
        if masked:
            data = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
        if opcode == 0x8:
            raise RuntimeError("CDP WebSocket closed")
        if opcode != 0x1:
            return self._recv_json(timeout_s)
        return json.loads(data.decode("utf-8"))

    def _read_exact(self, length: int) -> bytes:
        chunks = []
        remaining = length
        while remaining:
            chunk = self._socket.recv(remaining)
            if not chunk:
                raise RuntimeError("CDP socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


def _start_chrome(args: argparse.Namespace, output_dir: Path, url: str) -> tuple[subprocess.Popen, str, Path]:
    chrome_path = _find_chrome(args.chrome_path)
    profile_dir = Path(tempfile.mkdtemp(prefix="greenroom-chrome-profile-", dir=str(output_dir)))
    proc = subprocess.Popen(
        [
            chrome_path,
            "--headless=new",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            url,
        ],
        stdout=(output_dir / "chrome.stdout.log").open("w"),
        stderr=(output_dir / "chrome.stderr.log").open("w"),
        start_new_session=True,
    )
    active_port = profile_dir / "DevToolsActivePort"
    deadline = time.time() + 15
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Chrome exited early with code {proc.returncode}")
        if active_port.exists():
            lines = active_port.read_text().splitlines()
            if len(lines) >= 2:
                port = int(lines[0])
                return proc, f"http://127.0.0.1:{port}", profile_dir
        time.sleep(0.1)
    raise TimeoutError("Chrome did not write DevToolsActivePort")


def _browser_tabs(devtools_url: str) -> list[dict[str, Any]]:
    return _http_json(devtools_url, "/json/list")


def _runtime_evaluate(
    cdp: _CdpConnection,
    expression: str,
    *,
    timeout_s: float,
    await_promise: bool = True,
    return_by_value: bool = True,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return cdp.send(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "awaitPromise": await_promise,
                    "returnByValue": return_by_value,
                    "timeout": int(timeout_s * 1000),
                },
                timeout_s=max(1.0, deadline - time.time()),
            )
        except RuntimeError as error:
            last_error = error
            if "Execution context was destroyed" not in str(error):
                raise
            time.sleep(0.2)
    if last_error:
        raise last_error
    raise TimeoutError("Timed out waiting for Runtime.evaluate")


def _run_browser_producer(args: argparse.Namespace, source_path: Path, output_dir: Path, result_dir: Path) -> dict[str, Any]:
    server_proc: subprocess.Popen | None = None
    chrome_proc: subprocess.Popen | None = None
    cdp: _CdpConnection | None = None
    try:
        if args.kaminos_url:
            kaminos_url = args.kaminos_url
        else:
            server_proc, kaminos_url = _start_kaminos_server(args, output_dir, result_dir)
        request_id = args.request_id or args.job_id or f"req:{uuid.uuid4().hex}"
        source_identity = _parse_json_object(args.source_identity_json)
        row = _build_browser_route_row(
            source_path=source_path,
            route_id=args.route_id or DEFAULT_ROUTE_ID,
            request_id=request_id,
            job_id=args.job_id or request_id,
            source_identity=source_identity,
            module_base_url=args.module_base_url or "",
        )
        chrome_proc, devtools_url, _profile_dir = _start_chrome(args, output_dir, kaminos_url)
        deadline = time.time() + float(args.browser_timeout_s)
        target = None
        while time.time() < deadline and target is None:
            for tab in _browser_tabs(devtools_url):
                if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                    target = tab
                    break
            if target is None:
                time.sleep(0.1)
        if target is None:
            raise TimeoutError("No Chrome page target became available")
        cdp = _CdpConnection(target["webSocketDebuggerUrl"])
        cdp.send("Runtime.enable")
        _runtime_evaluate(
            cdp,
            "(async () => {"
            "while (document.readyState !== 'complete') {"
            "  await new Promise(resolve => setTimeout(resolve, 100));"
            "}"
            "return document.readyState;"
            "})()",
            timeout_s=min(30.0, float(args.browser_timeout_s)),
        )
        expression = (
            "(async () => {"
            "const row = " + json.dumps(row) + ";"
            "const options = {requestId: " + json.dumps(request_id) + ", requireRealInput: true};"
            "const waitUntil = Date.now() + 30000;"
            "while (!window.kaminosBuildBrowserWebGpuPreviewRouteResult) {"
            "  if (Date.now() > waitUntil) throw new Error('Kaminos browser producer did not load');"
            "  await new Promise(resolve => setTimeout(resolve, 100));"
            "}"
            "return await window.kaminosBuildBrowserWebGpuPreviewRouteResult(row, options);"
            "})()"
        )
        evaluated = _runtime_evaluate(
            cdp,
            expression,
            timeout_s=float(args.browser_timeout_s) + 5,
        )
        if evaluated.get("exceptionDetails"):
            raise RuntimeError(f"Browser producer exception: {evaluated['exceptionDetails']}")
        result = (evaluated.get("result") or {}).get("value")
        if not isinstance(result, dict):
            raise RuntimeError(f"Browser producer returned non-object result: {evaluated!r}")
        return result
    finally:
        if cdp:
            cdp.close()
        for proc in (chrome_proc, server_proc):
            if proc and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def _artifact(role: str, request_id: str, png_b64: str, shape: list[int]) -> dict[str, Any]:
    payload = base64.b64decode(png_b64)
    return {
        "role": role,
        "artifactId": f"{role}:{request_id}",
        "sha256": _sha256_bytes(payload),
        "shape": shape,
        "status": "real",
        "previewDataUrl": f"data:image/png;base64,{png_b64}",
        "mediaType": "image/png",
    }


def _build_result(args: argparse.Namespace, source_path: Path) -> dict[str, Any]:
    route_id = args.route_id or DEFAULT_ROUTE_ID
    request_id = args.request_id or args.job_id or f"req:{uuid.uuid4().hex}"
    created_at = _now_iso()
    source_identity = _parse_json_object(args.source_identity_json)
    input_sha = source_identity.get("sha256") if isinstance(source_identity.get("sha256"), str) else _sha256_path(source_path)

    outputs = [
        _artifact("depth", request_id, _PNG_1X1_TRANSPARENT, [1, 1]),
        _artifact("normal", request_id, _PNG_1X1_BLACK, [3, 1, 1]),
        _artifact("pointmap", request_id, _PNG_1X1_WHITE, [3, 1, 1]),
    ]
    stages = [
        {"name": "fixture-no-model", "ms": 0.0},
        {"name": "output-write", "ms": 0.0},
    ]
    scheduler = {
        "schema": "kaminos.webgpu-route-scheduler.v0",
        "requestedScheduler": {"mode": "greenroom-fixture"},
        "effectiveScheduler": {
            "mode": "greenroom-fixture",
            "unsupportedFields": [],
        },
        "verificationState": "fixture",
    }
    backpressure = {
        "schema": "kaminos.webgpu-route-backpressure.v0",
        "requestedBudget": "greenroom-worker",
        "effectiveBudget": "greenroom-worker",
        "memoryExclusivity": "greenroom-serialized-gpu-queue",
        "warmCacheState": "not-applicable",
        "frameTail": {
            "sampleWindowMs": 0,
            "longFrameCount": 0,
            "maxFrameGapMs": 0,
            "p95FrameGapMs": 0,
            "p99FrameGapMs": 0,
        },
    }
    runtime_profile = {
        "schema": "kaminos.webgpu-runtime-profile.v0",
        "routeId": route_id,
        "runtimeLabel": "gpu-greenroom-fixture-no-model",
        "backend": {
            "kind": "webgpu-local",
            "runtime": "browser",
            "adapterName": "fixture/no-model",
            "browser": "not-launched",
            "features": [],
            "requestedFeatures": [],
            "limits": {},
            "timestampQuery": "not-requested",
        },
        "kernel": {
            "kitVersion": args.kit_version or "unknown",
            "profile": "greenroom-fixture-no-model",
            "commit": "unresolved",
        },
        "profile": {
            "schema": "kaminos.webgpu-staged-profile.v0",
            "route": "greenroom-fixture-no-model",
            "timingSource": "greenroom-runner",
            "requiredStages": ["fixture-no-model", "output-write"],
            "stages": stages,
            "stageNames": [stage["name"] for stage in stages],
            "totalMs": 0.0,
        },
        "evidence": {
            "mode": "fallback",
            "source": "gpu-greenroom.kaminos-browser-preview-runner",
            "fallbackReason": "fixture/no-model Greenroom runner; Chrome/WebGPU was not launched",
            "classification": "fallback-fixture-no-model",
        },
        "requiredStages": ["fixture-no-model", "output-write"],
        "timingSource": "greenroom-runner",
        "createdAt": created_at,
    }
    source_artifact = {
        "role": "source-image",
        "artifactId": f"image:{request_id}",
        "sha256": input_sha,
        "shape": source_identity.get("shape") if isinstance(source_identity.get("shape"), list) else [0, 0, 3],
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "requestedRouteId": route_id,
        "effectiveRouteId": route_id,
        "status": "fallback",
        "fallbackReason": runtime_profile["evidence"]["fallbackReason"],
        "backend": runtime_profile["backend"],
        "model": {
            "id": MODEL_ID,
            "revision": "fixture-no-model",
            "weightsHash": "sha256:fixture-no-model",
            "dtype": "fixture",
        },
        "kernel": runtime_profile["kernel"],
        "inputs": [source_artifact],
        "outputs": outputs,
        "timings": {
            "source": "greenroom-runner",
            "totalMs": 0.0,
            "stages": stages,
        },
        "runtime": {
            "runtimeProfile": runtime_profile,
            "scheduler": scheduler,
            "backpressure": backpressure,
        },
        "createdAt": created_at,
    }
    return {
        "schema": RESULT_SCHEMA,
        "requestId": request_id,
        "routeId": route_id,
        "status": "fallback",
        "request": {
            "schema": "kaminos.webgpu-route-request.v0",
            "requestId": request_id,
            "routeId": route_id,
            "backendKind": "webgpu-local",
            "inputs": [source_artifact],
            "outputs": [
                {"role": output["role"], "artifactId": output["artifactId"], "shape": output["shape"]}
                for output in outputs
            ],
            "routeConfig": {
                "source": "gpu-greenroom.kaminos-browser-preview-runner",
                "sourceImageIdentity": source_identity,
                "inputSourceKind": source_identity.get("kind") or "file",
                "mode": args.mode,
                "moduleBaseUrl": args.module_base_url,
                "greenroomJobId": args.job_id,
            },
        },
        "receipt": receipt,
        "createdAt": created_at,
    }


def _write_result_and_sidecar(
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
    output_dir: Path,
    result_dir: Path,
) -> Path:
    result_path = result_dir / _safe_filename(result["routeId"], result["requestId"])
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    tmp_path = result_path.with_name(f".{result_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(body, encoding="utf-8")
    os.replace(tmp_path, result_path)

    request = result.get("request") if isinstance(result.get("request"), dict) else {}
    route_config = request.get("routeConfig") if isinstance(request.get("routeConfig"), dict) else {}
    sidecar = {
        "schema": "gpu-greenroom.kaminos-browser-preview-runner.v0",
        "status": "done",
        "mode": args.mode,
        "route_id": result["routeId"],
        "request_id": result["requestId"],
        "job_id": args.job_id,
        "result_path": str(result_path),
        "source_image_identity": route_config.get("sourceImageIdentity"),
        "created_at": result["createdAt"],
    }
    sidecar_path = output_dir / "greenroom-browser-webgpu-preview.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def run(args: argparse.Namespace) -> Path:
    source_path = Path(args.input_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {source_path}")

    output_dir = Path(args.output_dir).expanduser()
    result_dir = Path(args.result_dir).expanduser() if args.result_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "fixture":
        result = _build_result(args, source_path)
    elif args.mode == "browser":
        result = _run_browser_producer(args, source_path, output_dir, result_dir)
    else:
        raise ValueError(f"Unsupported Kaminos browser preview runner mode: {args.mode}")
    return _write_result_and_sidecar(args=args, result=result, output_dir=output_dir, result_dir=result_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Kaminos browser WebGPU preview under GPU Greenroom custody.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-dir", default="")
    parser.add_argument("--request-id", default="")
    parser.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--source-identity-json", default="{}")
    parser.add_argument("--module-base-url", default="")
    parser.add_argument("--kit-version", default="unknown")
    parser.add_argument("--mode", default="fixture", choices=["fixture", "browser"])
    parser.add_argument("--kaminos-root", default="")
    parser.add_argument("--kaminos-url", default="")
    parser.add_argument("--chrome-path", default="")
    parser.add_argument("--browser-timeout-s", default="120")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_path = run(args)
    print(json.dumps({
        "schema": "gpu-greenroom.kaminos-browser-preview-runner.stdout.v0",
        "status": "done",
        "mode": args.mode,
        "route_id": args.route_id or DEFAULT_ROUTE_ID,
        "request_id": args.request_id or args.job_id,
        "result_path": str(result_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
