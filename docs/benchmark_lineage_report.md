# Benchmark Lineage Reconciliation (C0)

## Reproducibility status

- Frozen baseline exact reproduction: **not reproducible exactly** from repository lineage alone.
- Reason: repo lineage does not contain an immutable artifact that yields 83.33% directly (no frozen eval JSON/script run output bound to that number).
- Closest reproducible lineage used:
  - commit: `8905873`
  - evaluation lineage script: `ai_deepfake/test_model.py`
  - detector lineage script: `ai_deepfake/detect.py`

## Protocol/data/model inputs

- Dataset used (both runs): `D:\Codes\face\ai_deepfake\dataset_final\test`
- Threshold used (both runs): `0.5`
- Weights (local-only, not committed):
  - best_model.pth: size `16339622` bytes, sha256 `1ae8c63ab7ea5d4629d771c0dba664f1bbfcafa2df713452c523a36d76736bec`
  - best_model_v2.pth: size `296785927` bytes, sha256 `c2d114932ff3586c56c93b0bff3e73883ddafc9b702d247cdc72dc4f6906b01c`

## Measured comparison

- Frozen source-of-truth accuracy: `0.8333` (83.33%)
- Legacy protocol reproduction accuracy: `0.9400` (94.00%)
- Hardened protocol accuracy: `0.9400` (94.00%)

### Per-image drift summary

- Images compared: `450`
- Prediction label changed count: `0`
- Face behavior changed count: `0`
- Status changed count: `0`

### Root-cause signals

1. **Prediction drift between legacy vs hardened on same dataset/protocol is zero-label-change** (`pred_changed_count=0`), so hardening itself did not alter classification labels on this run.
2. **Face/crop behavior did not change in this environment** (`face_changed_count=0`) because MTCNN dependency is unavailable in both runs.
3. **Status pipeline differs by design** (legacy implicit `ok`, hardened explicit `status`) but no non-ok cases occurred here.
4. Most likely cause for 83.33% vs 94.00% mismatch is **baseline lineage mismatch** (dataset/protocol/environment used to produce 83.33% is not fully reconstructable from repo history alone), not a hardening-side classifier change.

## Can C1 start safely?

- **Not yet.**
- C1 (calibration + threshold analysis) should start only after baseline lineage is reconciled with an exact, auditable frozen reproduction protocol.
