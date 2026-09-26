#!/usr/bin/env python3
"""MLX Metal benchmark for MoGe-2 inference.

Measures: cold load, first inference (JIT), warm inference, peak Metal memory.
Output: JSON to stdout, diagnostics to stderr.

Designed to run standalone or through the GPU Greenroom queue.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
from PIL import Image


# Sibling checkouts next to this repo unless MOGE_WORKSPACE says otherwise.
WORKSPACE = Path(os.environ.get("MOGE_WORKSPACE", Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--json", action="store_true", default=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=None, help="Write results JSON to this directory")
    args = parser.parse_args()

    fixture = args.image or str(WORKSPACE / "moge-webgpu" / "public" / "test_fixtures" / "input.png")

    print(f"MLX MoGe-2 benchmark -- runs={args.runs}", file=sys.stderr)
    print(f"Image: {fixture}", file=sys.stderr)

    # --- Cold load (model instantiation + weight loading) ---
    from moge_mlx import MoGeModel
    from moge_mlx.weights import load_moge_weights

    print("Loading model...", file=sys.stderr)
    t_load_start = time.perf_counter()
    model = MoGeModel(normal_head=True)
    n_weights = load_moge_weights(
        model, model_name="Ruicheng/moge-2-vitl-normal", verbose=True
    )
    mx.eval(model.parameters())
    t_load = time.perf_counter() - t_load_start
    print(f"Model load: {t_load:.3f}s ({n_weights} weight arrays)", file=sys.stderr)

    # --- Prepare input ---
    img = Image.open(fixture).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    img_chw = np.transpose(img_np, (2, 0, 1))  # [3, H, W]
    img_tensor = mx.array(img_chw)

    print(f"Input shape: {img_tensor.shape}", file=sys.stderr)

    # Reset peak memory before inference
    mx.metal.reset_peak_memory()

    # --- First inference (includes JIT compilation) ---
    print("First inference (JIT compile)...", file=sys.stderr)
    t_first_start = time.perf_counter()
    output = model.infer(img_tensor, resolution_level=9)
    for v in output.values():
        if isinstance(v, mx.array):
            mx.eval(v)
    t_first = time.perf_counter() - t_first_start
    print(f"First inference: {t_first:.3f}s", file=sys.stderr)

    # --- Warm inference runs ---
    print(f"Running {args.runs} warm inferences...", file=sys.stderr)
    warm_times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        output = model.infer(img_tensor, resolution_level=9)
        for v in output.values():
            if isinstance(v, mx.array):
                mx.eval(v)
        elapsed = time.perf_counter() - t0
        warm_times.append(elapsed)
        print(f"  warm {i+1}: {elapsed:.3f}s", file=sys.stderr)

    # --- Memory ---
    peak_mem_bytes = mx.metal.get_peak_memory()
    active_mem_bytes = mx.metal.get_active_memory()
    cache_mem_bytes = mx.metal.get_cache_memory()

    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    rss_mb = rusage.ru_maxrss / 1024 / 1024  # macOS reports in bytes

    # --- Stats ---
    def stats(arr):
        s = sorted(arr)
        return {
            "min": s[0],
            "max": s[-1],
            "median": s[len(s) // 2],
            "mean": sum(s) / len(s),
            "samples": s,
        }

    warm_stats = stats(warm_times)

    results = {
        "runtime": "MLX Metal",
        "precision": "fp32",
        "model": "moge-2-vitl-normal",
        "mlxVersion": mx.__version__,
        "runs": args.runs,
        "modelLoadMs": t_load * 1000,
        "firstInferenceMs": t_first * 1000,
        "warmInferenceMs": [t * 1000 for t in warm_times],
        "warmStats": {
            k: v * 1000 if isinstance(v, float) else [x * 1000 for x in v] if isinstance(v, list) else v
            for k, v in warm_stats.items()
        },
        "memoryMB": {
            "metal_peak_mb": peak_mem_bytes / 1024 / 1024,
            "metal_active_mb": active_mem_bytes / 1024 / 1024,
            "metal_cache_mb": cache_mem_bytes / 1024 / 1024,
            "rss_mb": rss_mb,
        },
        "imageSize": f"{img.size[0]}x{img.size[1]}",
        "weightArrays": n_weights,
    }

    print(f"\n--- Results ---", file=sys.stderr)
    print(f"Model load:       {t_load:.3f}s", file=sys.stderr)
    print(f"First inference:  {t_first:.3f}s (includes JIT)", file=sys.stderr)
    print(f"Warm inference:   median={warm_stats['median']:.3f}s, min={warm_stats['min']:.3f}s, max={warm_stats['max']:.3f}s", file=sys.stderr)
    print(f"Peak Metal mem:   {peak_mem_bytes / 1024 / 1024:.1f} MB", file=sys.stderr)
    print(f"RSS:              {rss_mb:.1f} MB", file=sys.stderr)

    json_out = json.dumps(results, indent=2)
    print(json_out)

    # Write to output dir if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, "benchmark_mlx.json")
        with open(out_path, "w") as f:
            f.write(json_out)
        print(f"Results written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
