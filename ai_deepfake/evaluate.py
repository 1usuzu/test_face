"""
Evaluation harness for DeepTrust deepfake detector.

This script measures detector performance on a local dataset and writes
machine-readable JSON output for baseline comparison across versions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EvalCounts:
    tp: int = 0  # fake predicted fake
    tn: int = 0  # real predicted real
    fp: int = 0  # real predicted fake
    fn: int = 0  # fake predicted real

    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


def _safe_div(n: float, d: float) -> float:
    return n / d if d else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_div(2 * precision * recall, precision + recall)


def _collect_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _compute_metrics(c: EvalCounts) -> Dict[str, float]:
    total = c.total()
    accuracy = _safe_div(c.tp + c.tn, total)

    precision_fake = _safe_div(c.tp, c.tp + c.fp)
    recall_fake = _safe_div(c.tp, c.tp + c.fn)
    f1_fake = _f1(precision_fake, recall_fake)

    precision_real = _safe_div(c.tn, c.tn + c.fn)
    recall_real = _safe_div(c.tn, c.tn + c.fp)
    f1_real = _f1(precision_real, recall_real)

    f1_macro = (f1_fake + f1_real) / 2
    return {
        "accuracy": accuracy,
        "precision_fake": precision_fake,
        "recall_fake": recall_fake,
        "f1_fake": f1_fake,
        "precision_real": precision_real,
        "recall_real": recall_real,
        "f1_real": f1_real,
        "f1_macro": f1_macro,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deepfake detector on a labeled image dataset.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Dataset root with two folders: real/ and fake/.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write JSON evaluation report.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override threshold passed to detector.predict().",
    )
    parser.add_argument(
        "--detector-version",
        default="current",
        help="Optional free-text tag for version comparison (e.g. v1-frozen, v2-experiment).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.output).resolve()
    real_dir = data_dir / "real"
    fake_dir = data_dir / "fake"

    if not data_dir.exists():
        print(f"[ERROR] Dataset path does not exist: {data_dir}")
        print("[HINT] Expected structure: <data-dir>/real and <data-dir>/fake")
        return 2

    if not real_dir.exists() or not fake_dir.exists():
        print("[ERROR] Missing required dataset folders.")
        print(f"        real folder exists: {real_dir.exists()} ({real_dir})")
        print(f"        fake folder exists: {fake_dir.exists()} ({fake_dir})")
        print("[HINT] Expected structure: <data-dir>/real and <data-dir>/fake")
        return 2

    real_images = _collect_images(real_dir)
    fake_images = _collect_images(fake_dir)

    if not real_images and not fake_images:
        print("[ERROR] No images found in dataset folders.")
        print("[HINT] Supported extensions: .jpg .jpeg .png .bmp .webp")
        return 2

    try:
        from detect import DeepfakeDetector  # local module import
    except Exception as exc:
        print(f"[ERROR] Failed to import detector: {exc}")
        return 3

    # Explicit model weight check for clear error messages.
    model_dir = Path(__file__).parent / "models"
    has_v1 = (model_dir / "best_model.pth").exists()
    has_v2 = (model_dir / "best_model_v2.pth").exists()
    if not (has_v1 or has_v2):
        print("[ERROR] No model weights found.")
        print(f"        expected at: {model_dir}")
        print("        looked for: best_model.pth and best_model_v2.pth")
        return 4

    try:
        detector = DeepfakeDetector()
    except Exception as exc:
        print(f"[ERROR] Detector initialization failed: {exc}")
        return 5

    counts = EvalCounts()
    non_ok_count = 0
    non_ok_breakdown: Dict[str, int] = {}
    per_image = []

    def run_one(path: Path, truth_is_fake: bool) -> None:
        nonlocal non_ok_count
        result = detector.predict(str(path), threshold=args.threshold)
        status_obj = getattr(result, "status", None)
        status = getattr(status_obj, "value", str(status_obj or "ok")).lower()
        error_value = None
        if isinstance(result.details, dict):
            error_value = result.details.get("error")

        row = {
            "file": str(path),
            "truth": "FAKE" if truth_is_fake else "REAL",
            "status": status,
            "pred_label": "FAKE" if result.is_fake else "REAL",
            "fake_probability": float(getattr(result, "fake_probability", 0.0)),
            "confidence": float(getattr(result, "confidence", 0.0)),
            "risk_level": getattr(getattr(result, "risk_level", None), "value", str(getattr(result, "risk_level", ""))),
            "error": error_value,
        }

        if status != "ok":
            non_ok_count += 1
            non_ok_breakdown[status] = non_ok_breakdown.get(status, 0) + 1
            row["counted_in_metrics"] = False
            per_image.append(row)
            return

        row["counted_in_metrics"] = True
        per_image.append(row)

        pred_is_fake = bool(result.is_fake)
        if truth_is_fake and pred_is_fake:
            counts.tp += 1
        elif truth_is_fake and not pred_is_fake:
            counts.fn += 1
        elif (not truth_is_fake) and pred_is_fake:
            counts.fp += 1
        else:
            counts.tn += 1

    for p in real_images:
        run_one(p, truth_is_fake=False)
    for p in fake_images:
        run_one(p, truth_is_fake=True)

    metrics = _compute_metrics(counts)
    total_images = len(real_images) + len(fake_images)
    counted = counts.total()

    report = {
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "detector_version_tag": args.detector_version,
            "data_dir": str(data_dir),
            "threshold_override": args.threshold,
            "model_dir": str(model_dir),
            "models_present": {
                "best_model.pth": has_v1,
                "best_model_v2.pth": has_v2,
            },
            "total_images": total_images,
            "real_images": len(real_images),
            "fake_images": len(fake_images),
            "counted_images_for_metrics": counted,
            "excluded_non_ok_images": non_ok_count,
        },
        "status_summary": {
            "non_ok_breakdown": non_ok_breakdown,
        },
        "metrics": metrics,
        "confusion_matrix": {
            "tp_fake_as_fake": counts.tp,
            "tn_real_as_real": counts.tn,
            "fp_real_as_fake": counts.fp,
            "fn_fake_as_real": counts.fn,
        },
        "per_image": per_image,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Evaluation completed.")
    print(f"     total images: {total_images}")
    print(f"     counted for metrics (status=ok): {counted}")
    print(f"     excluded non-ok: {non_ok_count}")
    print(f"     report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
