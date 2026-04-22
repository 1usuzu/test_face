# Repo Push Audit (`ai-v1.1-stabilized-c1`)

## Scope

This audit is for repository slimming and push safety only. No model changes, no calibration redesign, no training work.

## Working tree inspected

- Modified tracked: `.gitignore`, `README.md`, `ai_deepfake/ai_config.py`, `backend/.env.example`, `backend/Dockerfile`
- New untracked: `ai_deepfake/c1_analysis.py`, multiple docs under `docs/`
- Ignored local artifacts present: `ai_deepfake/models/`, `docs/c1_eval_base.json`, `docs/per_image_diff.csv`, `docs/baseline_frozen_reproduced.json`, `docs/baseline_hardened_same_protocol.json`

## File-by-file disposition

### Keep in repo (safe to push)

- `ai_deepfake/ai_config.py`
- `ai_deepfake/c1_analysis.py`
- `backend/.env.example`
- `backend/Dockerfile`
- `README.md`
- `docs/ai_module_stabilization_milestone.md`
- `docs/baseline_decision_record.md`
- `docs/baseline_protocol_frozen.md`
- `docs/baseline_reproducible_anchor.json`
- `docs/benchmark_lineage_report.md`
- `docs/c1_calibration_report.md`
- `docs/c1_operating_points.json`
- `docs/c1_threshold_sweep_report.md`
- `docs/deploy_readiness_checklist.md`
- `docs/release_readiness_checklist.md`

### Keep but move/archive (not required before this push)

- None required for this milestone.

### Ignore locally (do not push)

- `ai_deepfake/models/` (weights, machine-local runtime assets)
- `docs/c1_eval_base.json` (generated evaluation artifact)
- `docs/per_image_diff.csv` (generated per-image artifact)
- `docs/baseline_frozen_reproduced.json` (generated run artifact)
- `docs/baseline_hardened_same_protocol.json` (generated run artifact)

### Delete safely

- `frontend/consumer_output.txt` (generated Vite HTML dump, not source)

### Do not push

- Any real secret files: `backend/.env`, `frontend/.env`, root `.env*` with real keys
- Model binaries: `ai_deepfake/models/*.pth`, `ai_deepfake/models/*.pt`, `*.onnx`, `*.ckpt`

## .gitignore updates applied

- Added `ai_deepfake/models/` to ignore the full local model directory.
- Added `frontend/consumer_output.txt` to prevent re-introducing generated dump files.

## Push plan (exact)

1. Keep and stage only source/config/docs listed in **Keep in repo**.
2. Keep generated evaluation artifacts local-only (already ignored).
3. Keep model weights local-only (already ignored).
4. Keep `.env` secrets untracked.
5. Confirm staged set with:
   - `git status --short`
   - `git diff --cached --name-only`
