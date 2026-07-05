"""SAM3 element isolation job for GPU Greenroom.

This command is intentionally file-in/file-out. It writes an isolation receipt
even when it fails before model load so Greenroom jobs cannot masquerade as
silent missing outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def _split_prompts(raw_prompts: list[str] | None, raw_prompt: list[str] | None) -> list[str]:
    values: list[str] = []
    for raw in (raw_prompts or []) + (raw_prompt or []):
        for chunk in raw.replace(";", "|").split("|"):
            prompt = chunk.strip()
            if prompt:
                values.append(prompt)
    return values


def _parse_boxes(raw: str | None) -> list[list[float]]:
    if not raw:
        return []
    boxes: list[list[float]] = []
    for chunk in raw.replace(";", "|").split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [float(p.strip()) for p in chunk.split(",") if p.strip()]
        if len(parts) != 4:
            raise ValueError(f"box must be x1,y1,x2,y2; got {chunk!r}")
        boxes.append(parts)
    return boxes


def _write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _output_inventory(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.name != "isolation-receipt.json"
    )


def _ensure_rgba_cutout(image, mask, feather: float):
    from PIL import Image, ImageFilter

    rgba = image.convert("RGBA")
    alpha = Image.fromarray((mask.astype("uint8") * 255), mode="L")
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    rgba.putalpha(alpha)
    return rgba


def _draw_overlay(image, masks, boxes, scores, labels: list[str]):
    from PIL import Image, ImageDraw

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = [
        (255, 112, 24, 110),
        (80, 170, 255, 110),
        (255, 48, 96, 110),
        (64, 220, 128, 110),
        (224, 176, 64, 110),
        (160, 120, 255, 110),
    ]
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        mask_img = Image.fromarray((mask.astype("uint8") * 180), mode="L")
        color_img = Image.new("RGBA", base.size, color)
        overlay.alpha_composite(
            Image.composite(color_img, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask_img)
        )
        x1, y1, x2, y2 = [float(v) for v in boxes[idx]]
        draw.rectangle((x1, y1, x2, y2), outline=color[:3] + (255,), width=3)
        label = labels[idx] if idx < len(labels) else "mask"
        draw.text((x1 + 4, max(0, y1 - 18)), f"{idx} {float(scores[idx]):.3f} {label}", fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, overlay)


def _write_checker_contact(output_dir: Path, cutout_paths: list[Path], columns: int = 4) -> Path | None:
    if not cutout_paths:
        return None
    from PIL import Image, ImageDraw

    images = [Image.open(path).convert("RGBA") for path in cutout_paths]
    thumb = 256
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGBA", (columns * thumb, rows * thumb), (30, 30, 30, 255))
    checker = Image.new("RGBA", (thumb, thumb), (0, 0, 0, 0))
    draw = ImageDraw.Draw(checker)
    tile = 16
    for y in range(0, thumb, tile):
        for x in range(0, thumb, tile):
            c = 72 if ((x // tile) + (y // tile)) % 2 else 122
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(c, c, c, 255))
    for idx, image in enumerate(images):
        cell = checker.copy()
        image.thumbnail((thumb - 16, thumb - 16))
        ox = (thumb - image.width) // 2
        oy = (thumb - image.height) // 2
        cell.alpha_composite(image, (ox, oy))
        sheet.alpha_composite(cell, ((idx % columns) * thumb, (idx // columns) * thumb))
    path = output_dir / "cutouts-checker-contact.png"
    sheet.save(path)
    return path


def run(args: argparse.Namespace) -> int:
    started = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "isolation-receipt.json"
    prompts = _split_prompts(args.prompts, args.prompt)
    boxes_list = _parse_boxes(args.boxes)
    payload: dict[str, Any] = {
        "ok": False,
        "phase": "start",
        "image": str(args.image),
        "output_dir": str(output_dir),
        "model": args.model,
        "prompts": prompts,
        "boxes": boxes_list,
        "threshold": args.threshold,
        "max_results": args.max_results,
        "feather": args.feather,
        "outputs": {},
        "output_inventory": [],
        "started_at": started,
    }

    try:
        if not prompts:
            raise ValueError("at least one prompt is required")

        payload["phase"] = "load_image"
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(str(image_path))

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        payload["image_size"] = [image.width, image.height]

        payload["phase"] = "load_model"
        import numpy as np
        from mlx_vlm.utils import get_model_path, load_model

        model_path = get_model_path(args.model)
        payload["effective_model_path"] = str(model_path)
        model = load_model(model_path)
        boxes = np.asarray(boxes_list, dtype=np.float32) if boxes_list else None
        if "sam3.1" in args.model or "sam3_1" in args.model:
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.generate import predict_multi
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor

            processor = Sam31Processor.from_pretrained(str(model_path))
            payload["effective_processor"] = "sam3_1.Sam31Processor"
            payload["effective_predict_multi"] = "sam3_1.generate.predict_multi"
        else:
            from mlx_vlm.models.sam3.generate import Sam3Predictor, predict_multi
            from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

            processor = Sam3Processor.from_pretrained(str(model_path))
            payload["effective_processor"] = "sam3.Sam3Processor"
            payload["effective_predict_multi"] = "sam3.generate.predict_multi"
        predictor = Sam3Predictor(model, processor, score_threshold=args.threshold)

        payload["phase"] = "predict"
        if len(prompts) == 1:
            result = predictor.predict(image, text_prompt=prompts[0], boxes=boxes, score_threshold=args.threshold)
            labels = [prompts[0]] * len(result.scores)
        else:
            result = predict_multi(predictor, image, prompts, boxes=boxes, score_threshold=args.threshold)
            labels = result.labels or []

        order = np.argsort(-result.scores)[: args.max_results]
        masks = result.masks[order] if len(order) else np.zeros((0, image.height, image.width), dtype=np.uint8)
        boxes_out = result.boxes[order] if len(order) else np.zeros((0, 4), dtype=np.float32)
        scores = result.scores[order] if len(order) else np.zeros((0,), dtype=np.float32)
        labels_out = [labels[int(i)] if int(i) < len(labels) else "" for i in order]

        payload["phase"] = "write_outputs"
        detections = []
        cutout_paths = []
        for j, original_index in enumerate(order):
            mask = masks[j]
            mask_path = output_dir / f"mask-{j:02d}.png"
            cutout_path = output_dir / f"cutout-{j:02d}.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)
            _ensure_rgba_cutout(image, mask, args.feather).save(cutout_path)
            cutout_paths.append(cutout_path)
            payload["outputs"][f"mask_{j:02d}"] = str(mask_path)
            payload["outputs"][f"cutout_{j:02d}"] = str(cutout_path)
            detections.append({
                "index": int(original_index),
                "rank": j,
                "score": float(scores[j]),
                "box": [float(v) for v in boxes_out[j]],
                "mask_pixels": int(mask.sum()),
                "label": labels_out[j],
            })

        overlay_path = output_dir / "overlay.png"
        _draw_overlay(image, masks, boxes_out, scores, labels_out).save(overlay_path)
        payload["outputs"]["overlay"] = str(overlay_path)
        contact_path = _write_checker_contact(output_dir, cutout_paths)
        if contact_path:
            payload["outputs"]["cutouts_checker_contact"] = str(contact_path)
        payload["detections"] = detections
        payload["ok"] = True
        payload["phase"] = "complete"
        payload["finished_at"] = time.time()
        payload["duration_s"] = payload["finished_at"] - started
        payload["output_inventory"] = _output_inventory(output_dir)
        _write_receipt(receipt_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload["ok"] = False
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["finished_at"] = time.time()
        payload["duration_s"] = payload["finished_at"] - started
        payload["output_inventory"] = _output_inventory(output_dir)
        _write_receipt(receipt_path, payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local MLX SAM3 element isolation and save masks/cutouts.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts", action="append", help="Prompt string; split multiple prompts with '|'.")
    parser.add_argument("--prompt", action="append", help="Prompt string; may be repeated.")
    parser.add_argument("--boxes", default="", help="Optional xyxy boxes; split multiple boxes with '|'.")
    parser.add_argument("--model", default="mlx-community/sam3.1-bf16")
    parser.add_argument("--threshold", default=0.15, type=float)
    parser.add_argument("--max-results", default=4, type=int)
    parser.add_argument("--feather", default=1.25, type=float)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
