# Baseline Protocol Frozen (C0.5)

## Purpose

Đóng băng protocol benchmark có thể tái lập để làm technical comparison anchor trước khi bắt đầu C1.

## Scope

- Không thay model architecture
- Không training mới
- Không tuning threshold trong C0.5
- Không rewrite historical baseline claim

## Frozen protocol definition

### Dataset

- Root: `D:\Codes\face\ai_deepfake\dataset_final\test`
- Split assumptions:
  - `fake/` = class fake
  - `real/` = class real
- Supported image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### Models

- `ai_deepfake/models/best_model.pth`
  - size: `16339622`
  - sha256: `1ae8c63ab7ea5d4629d771c0dba664f1bbfcafa2df713452c523a36d76736bec`
- `ai_deepfake/models/best_model_v2.pth`
  - size: `296785927`
  - sha256: `c2d114932ff3586c56c93b0bff3e73883ddafc9b702d247cdc72dc4f6906b01c`

### Threshold and scoring

- Default operating threshold (runtime / C1 policy): **`0.65`**
- Fake score semantics: `fake_probability` from ensemble pipeline
- Decision: `pred_is_fake = fake_probability >= 0.65`
- Historical note: C0 lineage comparison reports used threshold `0.5` where explicitly stated; current repo default is `0.65`.

### Commit lineage references

- Baseline lineage AI commit: `8905873`
- PR2 (inference contract hardening): `b48c017`
- PR3 (API hardening): `cb81991`
- PR4 (docs + eval harness): `198ce98`
- C0 artifacts commit: `3cbb20e`

### Scripts and artifacts

- Baseline lineage script reference: `ai_deepfake/test_model.py` (from commit `8905873`)
- Current measurement artifacts:
  - `docs/baseline_frozen_reproduced.json`
  - `docs/baseline_hardened_same_protocol.json`
  - `docs/per_image_diff.csv`
  - `docs/benchmark_lineage_report.md`

## Frozen baseline governance separation

- Historical claim: `83.33%` (historical only, currently non-auditable exact reproduction)
- Reproducible technical anchor: `94.00%` under frozen protocol above

## Integrity checks before any future comparison run

1. Verify dataset path exists and split folders are unchanged.
2. Verify both model file hashes exactly match frozen values.
3. Verify threshold is `0.65` (or document explicitly if reproducing a historical `0.5` run).
4. Verify outputs are exported as:
   - per-run JSON
   - per-image diff CSV (when comparing runs)
5. Record environment flags that may affect behavior (e.g., MTCNN/OpenCV availability).
