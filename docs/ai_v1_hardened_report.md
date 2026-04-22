# AI V1 Hardened Evaluation Report

## Benchmark setup

- Run date (local): 2026-04-08
- Repository: `deep-trust`
- Detector tag: `ai-v1.1-hardened`
- Runtime: CPU (`DEVICE=cpu` in this run)
- Face detection dependency: `facenet-pytorch` not installed (face extraction disabled during run)
- Signal analysis dependency: `opencv-python` not installed (signal boost path disabled during run)

## Dataset path used

- `D:\Codes\face\ai_deepfake\dataset_final\test`
- Structure validated by harness:
  - `real/`
  - `fake/`

## Model weights found

- `ai_deepfake/models/best_model.pth`: found
- `ai_deepfake/models/best_model_v2.pth`: found

## Threshold used

- `threshold_override`: `null` (default detector threshold in code path)

## Status counts (OK vs non-OK)

- Total images: `450`
- Counted for metrics (`status=ok`): `450`
- Excluded non-OK images: `0`
- Non-OK breakdown: `{}`

## Measured metrics

- Accuracy: `0.94`
- Precision (fake): `0.9125`
- Recall (fake): `0.9733333333333334`
- F1 (fake): `0.9419354838709678`
- Precision (real): `0.9714285714285714`
- Recall (real): `0.9066666666666666`
- F1 (real): `0.9379310344827586`
- Macro F1: `0.9399332591768632`

## Confusion matrix

- TP (fake predicted fake): `219`
- TN (real predicted real): `204`
- FP (real predicted fake): `21`
- FN (fake predicted real): `6`

## Comparison versus frozen baseline

- Frozen baseline source of truth: `83.33%` accuracy
- Hardened measured run: `94.00%` accuracy
- Drift vs frozen baseline: `+10.67` percentage points

## Interpretation: behavior drift vs safety-only hardening

This hardened run does **not** match the frozen baseline accuracy value. Measured behavior on the benchmark path above is materially different from the frozen source-of-truth metric.

Given this drift, we cannot claim with high confidence that changes were purely safety/contract handling at benchmark level. PR2 intentionally changed contract semantics (status propagation and confidence meaning), and while these should not directly improve raw classification in this environment, empirical benchmark output still differs from frozen baseline.

## Conclusion: safe to tag as `ai-v1.1-hardened`?

**Not yet recommended** as final tag for baseline lineage until reproducibility is reconciled:

1. Re-run the exact frozen-baseline procedure on the same benchmark protocol.
2. Run pre-hardened and hardened code side-by-side under the same environment and dataset.
3. Confirm whether the +10.67pp drift is expected (environment/protocol difference) or true classifier behavior change.

After reconciliation, tag decision can be finalized.
