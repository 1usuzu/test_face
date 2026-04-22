# AI Module Stabilization Milestone (Stop at C1)

## Current detector architecture

- Inference engine uses a fixed ensemble of two image classifiers:
  - EfficientNet-B0 (`best_model.pth`)
  - EfficientNet-B4-head variant (`best_model_v2.pth`)
- Final fake score is `fake_probability`; decision is threshold-based.
- Current runtime default threshold is **0.65** (balanced operating point from C1).
- Status-aware behavior is active in API layer (`ok`, `no_face`, `face_detection_error`, `no_model`, `error`).

## Baseline governance status

- **83.33%** is retained as a **historical-only claim**.
- It is not currently an exactly auditable technical baseline from recoverable lineage artifacts.
- Governance records are frozen in:
  - `docs/baseline_decision_record.md`
  - `docs/baseline_protocol_frozen.md`
  - `docs/baseline_reproducible_anchor.json`

## Frozen reproducible anchor

- Reproducible technical anchor: **94.00%** under frozen C0.5 protocol.
- Dataset path used in reconciliation: `D:\Codes\face\ai_deepfake\dataset_final\test`
- Default operating threshold in repo (C1): **`0.65`** (C0 benchmark text may reference `0.5` where historical.)
- Local model file hashes and sizes are recorded in baseline governance docs.

## Threshold policy (C1 decision)

- Default mode: **balanced**
- Default threshold: **0.65**
- Alternative operating modes:
  - High-recall fake detection: `0.40`
  - High-precision fake detection: `0.75`
- Source of truth: `docs/c1_operating_points.json`

## Calibration limitation

- C1 measured calibration indicates weak confidence calibration quality:
  - ECE (10 bins): `0.149655`
  - Brier score: `0.055036`
- Therefore, `fake_probability` should currently be treated as a ranking/decision score, not as fully calibrated probability.

## Known risks

1. Local model weights are not committed artifacts; reproducibility depends on recorded file hashes.
2. Optional dependencies (`facenet-pytorch`, `opencv-python`) can affect behavior if environment changes.
3. Historical baseline (83.33%) remains governance-sensitive until exact lineage artifact recovery is possible.
4. Threshold policy is now explicit, but downstream consumers must avoid reintroducing hardcoded thresholds.

## Stabilization decision

- Stop at C1 is valid.
- Do **not** start model replacement from this milestone.
- Next phase (only when requested): C2 calibration improvements under the same frozen protocol.
