#!/usr/bin/env python3
"""Build a disposable, synthetic queue and serve the operator console over it.

Used to produce the console screenshot in the README. Every row comes from
the real CLI and a real worker; only the workloads are fake (short Python
one-liners and one sleep). Nothing here touches the default queue directory.

    python docs/console_fixture.py /tmp/greenroom-demo 8767

Prints the console URL, then keeps the worker and server running until
interrupted.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

WRITE_OUTPUT = (
    "import pathlib, sys; out = pathlib.Path(sys.argv[1]); "
    "out.mkdir(parents=True, exist_ok=True); (out / sys.argv[2]).write_text('demo')"
)


def cli(queue_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), *args],
        check=check, capture_output=True, text=True,
    )


def submit(queue_dir: Path, agent: str, repo: str, route: str, name: str, argv: list[str]) -> None:
    cli(
        queue_dir, "submit-command",
        "--agent-id", agent,
        "--repo-root", repo, "--cwd", repo,
        "--route-identity", route,
        "--output-dir", str(queue_dir / "outputs" / name),
        "--", *[part.replace("{out}", str(queue_dir / "outputs" / name)) for part in argv],
    )


def main() -> None:
    queue_dir = Path(sys.argv[1]).resolve()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8767
    if queue_dir.exists():
        shutil.rmtree(queue_dir)
    queue_dir.mkdir(parents=True)
    # submit-command requires repo_root and cwd to exist; fake checkouts live
    # under the demo queue so the whole fixture is one deletable directory.
    repos = {}
    for key, name in (("trellis", "trellis2mlx"), ("moge", "moge-mlx"), ("sf3d", "sf3d-webgpu"),
                      ("kaminos", "kaminos"), ("mflux", "mlx-ideogram4")):
        path = queue_dir / "repos" / name
        path.mkdir(parents=True)
        repos[key] = str(path)
    ok = [sys.executable, "-c", WRITE_OUTPUT]

    # A lease that was claimed and released earlier shows on the console.
    cli(queue_dir, "lease", "claim", "--owner", "render-holder", "--agent-id", "render-holder",
        "--repo-root", repos["kaminos"], "--pid", str(os.getpid()),
        "--effective-route", "chrome --enable-unsafe-webgpu kiln.html", "--backend", "webgpu",
        "--device", "metal", "--profile", "interactive-render", "--supports-checkpoints",
        "--ttl-seconds", "300")
    lease_id = None
    for line in cli(queue_dir, "lease", "status").stdout.splitlines():
        if '"lease_id"' in line:
            lease_id = line.split('"')[3]
    cli(queue_dir, "lease", "release", lease_id, "--released-by", "render-holder",
        "--reason", "capture checkpoint complete")

    # FIFO order is the story: two finish, one fails, one is long, three wait.
    submit(queue_dir, "trellis-batch", repos["trellis"], "trellis2mlx skull.png --resolution 512 --seed 42",
           "skull-trellis", [*ok, "{out}", "output.glb"])
    submit(queue_dir, "depth-probe", repos["moge"], "moge depth --image plate.png --dtype float16",
           "plate-depth", [*ok, "{out}", "depth.npz"])
    submit(queue_dir, "sf3d-sweep", repos["sf3d"], "sf3d skull.png --texture-resolution 2048",
           "skull-sf3d", [sys.executable, "-c", "raise SystemExit(3)"])
    submit(queue_dir, "kiln-witness", repos["kaminos"], "blender render skull.glb --views 8",
           "skull-views", ["sleep", "600"])
    submit(queue_dir, "mflux-edit", repos["mflux"], "mflux edit plate.png --prompt-file prompt.txt --steps 20",
           "plate-edit", [*ok, "{out}", "edit.png"])
    submit(queue_dir, "trellis-batch", repos["trellis"], "trellis2mlx warrior.png --resolution 768 --seed 7",
           "warrior-trellis", [*ok, "{out}", "output.glb"])
    submit(queue_dir, "example-agent", repos["moge"], "moge depth --image warrior.png",
           "warrior-depth", [*ok, "{out}", "depth.npz"])

    worker = subprocess.Popen(
        [sys.executable, "-m", "gpu_queue.cli", "--queue-dir", str(queue_dir), "worker", "--poll", "0.5"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if any((queue_dir / "running").iterdir()):
            break
        time.sleep(0.5)

    server = subprocess.Popen(
        [sys.executable, "-m", "gpu_queue.operator_server", "--queue-dir", str(queue_dir), "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    url = server.stdout.readline().strip()
    print(url, flush=True)
    print(f"worker_pid={worker.pid} server_pid={server.pid}", flush=True)

    def stop(*_):
        for proc in (server, worker):
            proc.send_signal(signal.SIGTERM)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
