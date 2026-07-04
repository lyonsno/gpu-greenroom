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
import json
import os
import uuid
from pathlib import Path
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


def run(args: argparse.Namespace) -> Path:
    if args.mode != "fixture":
        raise ValueError(f"Unsupported Kaminos browser preview runner mode: {args.mode}")
    source_path = Path(args.input_path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {source_path}")

    output_dir = Path(args.output_dir).expanduser()
    result_dir = Path(args.result_dir).expanduser() if args.result_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    result = _build_result(args, source_path)
    result_path = result_dir / _safe_filename(result["routeId"], result["requestId"])
    body = json.dumps(result, indent=2, sort_keys=True) + "\n"
    tmp_path = result_path.with_name(f".{result_path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(body, encoding="utf-8")
    os.replace(tmp_path, result_path)

    sidecar = {
        "schema": "gpu-greenroom.kaminos-browser-preview-runner.v0",
        "status": "done",
        "mode": args.mode,
        "route_id": result["routeId"],
        "request_id": result["requestId"],
        "job_id": args.job_id,
        "result_path": str(result_path),
        "source_image_identity": result["request"]["routeConfig"]["sourceImageIdentity"],
        "created_at": result["createdAt"],
    }
    sidecar_path = output_dir / "greenroom-browser-webgpu-preview.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


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
    parser.add_argument("--mode", default="fixture", choices=["fixture"])
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
