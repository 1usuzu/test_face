# Phase B: Stabilize + Harden — AI Deepfake Detection Module

**Date:** 2026-04-08
**Scope:** Repair and harden `ai_deepfake/` inference contract, backend API layers, tests, documentation.
**Status:** Approved — implementation not yet started.

---

## Hard constraints

1. The frozen baseline source of truth is the **measured baseline report at 83.33% accuracy**. The 93.33% figure in the legacy README is a historical claim only and must never be labeled as the frozen baseline.
2. Evaluation docs and sample JSON must not contain realistic-looking metric numbers unless actually measured in this repo. Use `<measured_value>` placeholders or mark values as illustrative.
3. `status != OK` is a **hard rule** across all API layers. Any non-OK detector result must not be interpreted as REAL or APPROVED by downstream code.
4. Phase B is focused on correctness, safety, testability, and benchmarkability. No capability expansion (no video support, no ONNX export, no TTA implementation) — only config cleanup and scaffolding.

---

## 1. Files and scope

### 1.1 Files to modify

| File | Why | Behavior change | Backward compat |
|---|---|---|---|
| `ai_deepfake/detect.py` | 8 bugs: confidence semantics, fail-open on NO_FACE/MTCNN error/catch-all/no-model, double MTCNN pass, ENABLE_SIGNAL_ANALYSIS not wired, singleton untestable | Add `status` field to DetectionResult. Fix `confidence` value. Error states get `status != OK`. Single MTCNN pass. | `fake_probability`, `is_fake`, `risk_level`, `processing_time`, `details`, `to_dict()`, `fake_prob` property, `real_prob` property all preserved. |
| `ai_deepfake/ai_config.py` | `ENABLE_TTA=True` but never implemented (misleading). `ENABLE_SIGNAL_ANALYSIS` not wired. Signal magic numbers hardcoded in detect.py. | Add signal constants. `ENABLE_TTA` default → False. | All existing field names preserved. Default values unchanged except TTA. |
| `backend/api.py` | Error path uses string matching on `details["error"]`. Lifespan only checks `best_model.pth` (misses V2). Response missing `status`, `face_detected`. Label "ERROR" is ambiguous. | Check `status` field. Add fields to response. Label "ERROR" → "INCONCLUSIVE". Lifespan checks V1 OR V2. | Existing response fields preserved. New fields are additive. |
| `backend/consumer_api.py` | **Does not check error state from detector** → fail-open: no-face image → `APPROVED`. Security vulnerability. | Add status check. NO_FACE/ERROR → `REJECTED`. | Response schema adds fields. `APPROVED` → `REJECTED` for non-face images (intentional security fix). |
| `backend/requirements.txt` | Missing `pydantic-settings` (imported by ai_config.py). Missing `pytest`. | Add dependencies. | N/A |
| `README.md` | 5 inaccuracies: (1) says B0 only, actually B0+B4 ensemble, (2) 93.33% is unverified legacy claim not frozen baseline, (3) Temperature Scaling listed but not in inference code, (4) lists `train.py`/`test_model.py` which don't exist, (5) response example missing fields. | Update to reflect actual code. | N/A |

### 1.2 New files

| File | Purpose |
|---|---|
| `ai_deepfake/__init__.py` | Package exports |
| `ai_deepfake/tests/__init__.py` | Test package marker |
| `ai_deepfake/tests/conftest.py` | Fixtures: mock models, mock MTCNN, sample PIL images |
| `ai_deepfake/tests/test_detect.py` | Unit tests for detector inference contract |
| `backend/tests/__init__.py` | Test package marker |
| `backend/tests/conftest.py` | Fixtures: FastAPI TestClient, mock detector |
| `backend/tests/test_api_verify.py` | Integration tests for `/api/verify`, `/api/verify-zkp` |
| `backend/tests/test_consumer_api.py` | Integration tests for `/v1/consumer/verify-image` |
| `ai_deepfake/evaluate.py` | Baseline evaluation harness |

---

## 2. Inference contract

### 2.1 DetectionResult after hardening

```python
class DetectionStatus(Enum):
    OK = "ok"
    NO_FACE = "no_face"
    FACE_DETECTION_ERROR = "face_detection_error"
    NO_MODEL = "no_model"
    ERROR = "error"

@dataclass
class DetectionResult:
    status: DetectionStatus           # NEW — caller MUST check before trusting is_fake
    is_fake: bool
    confidence: float                 # FIXED — now = P(predicted_class)
    fake_probability: float           # UNCHANGED — always = P(fake)
    risk_level: RiskLevel
    processing_time: float
    details: Dict[str, Any]
```

### 2.2 Behavior table

| Case | status | is_fake | confidence | fake_probability | risk_level |
|---|---|---|---|---|---|
| Predicted FAKE, fake_prob=0.85 | OK | True | 0.85 | 0.85 | CRITICAL |
| Predicted REAL, fake_prob=0.15 | OK | False | 0.85 | 0.15 | LOW |
| Predicted REAL, fake_prob=0.45 | OK | False | 0.55 | 0.45 | MEDIUM |
| MTCNN found no face | NO_FACE | False* | 0.0 | 0.0 | LOW |
| MTCNN raised exception | FACE_DETECTION_ERROR | False* | 0.0 | 0.0 | LOW |
| No model produced output | NO_MODEL | False* | 0.0 | 0.0 | LOW |
| Unhandled exception | ERROR | False* | 0.0 | 0.0 | LOW |

**(*) Hard rule:** When `status != OK`, `is_fake`/`confidence`/`fake_probability` are placeholders with no classification meaning. Callers MUST check `status` before trusting `is_fake`. Downstream code MUST NOT interpret non-OK results as REAL or APPROVED.

### 2.3 Confidence semantics fix

Before (bug): `confidence = fake_probability` always.
After (fix): `confidence = fake_probability if is_fake else (1.0 - fake_probability)`.

This aligns code with the README-documented response example where `label=REAL, confidence=0.92, real_prob=0.92`.

### 2.4 Double MTCNN fix

Before: two forward passes (`self.mtcnn()` + `self.mtcnn.detect()`).
After: single `self.mtcnn.detect()` call + manual crop.

---

## 3. API response schema

### 3.1 POST /api/verify — Success

```json
{
    "status": "ok",
    "label": "REAL",
    "confidence": "<measured_value>",
    "fake_probability": "<measured_value>",
    "real_prob": "<measured_value>",
    "fake_prob": "<measured_value>",
    "risk_level": "low|medium|high|critical",
    "face_detected": true,
    "image_hash": "...",
    "filename": "...",
    "signature": "0x...",
    "debug_msg": "...",
    "detector_version": "ensemble"
}
```

Changes: `status` (new), `face_detected` (new), `fake_probability` (new), `confidence` value fixed (bug fix). All existing fields preserved.

### 3.2 POST /api/verify — No Face

```json
{
    "status": "no_face",
    "label": "INCONCLUSIVE",
    "message": "...",
    "error_code": "NO_FACE_DETECTED",
    "face_detected": false,
    "confidence": 0.0,
    "risk_level": "unknown",
    "image_hash": "..."
}
```

No `signature` returned (no valid classification to sign).

### 3.3 POST /v1/consumer/verify-image — No Face (SECURITY FIX)

```json
{
    "found": false,
    "status": "no_face",
    "label": "INCONCLUSIVE",
    "decision": "REJECTED",
    "reason": "...",
    "image_hash": "...",
    "confidence": 0.0,
    "face_detected": false,
    "risk_level": "unknown"
}
```

Before: `decision: "APPROVED"` (fail-open). After: `decision: "REJECTED"` (fail-closed).

---

## 4. PR slicing

| PR | Branch | Scope | Depends on |
|---|---|---|---|
| PR1 | `fix/ai-foundation` | Package init, config cleanup, DetectionStatus enum, singleton fix, data contract tests | None |
| PR2 | `fix/inference-contract` | Confidence fix, status field wiring, double MTCNN fix, ENABLE_SIGNAL_ANALYSIS, inference tests | PR1 |
| PR3 | `fix/api-hardening` | api.py + consumer_api.py status checks, response schema, lifespan fix, integration tests | PR2 |
| PR4 | `docs/baseline-eval` | README alignment, evaluation harness | PR2 |

---

## 5. Risks and open questions

1. **Model weights missing** — `ai_deepfake/models/` is empty. Tests use mocks. Eval harness needs real weights.
2. **Class label mapping unverified** — Code assumes Class 0 = Fake. If inverted, all results flip. Eval harness detects this (accuracy < 50%).
3. **Frontend may depend on `label === "ERROR"`** — Mitigated by `error_code` field for gradual migration.
4. **Frontend may depend on old `confidence` values** — README already documents the corrected behavior.
5. **Temperature Scaling** — README mentions T=3.5 but code doesn't implement it. Phase B only fixes README. Implementation is Phase C.
6. **Signal analysis uncalibrated** — Magic numbers moved to config + documented as "uncalibrated defaults." Calibration is Phase C.
7. **93.33% is a legacy README claim** — The frozen baseline is 83.33% as measured. All docs must reflect this distinction.
