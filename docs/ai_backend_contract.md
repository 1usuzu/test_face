# AI-Backend Contract (Frozen v1)

This document freezes the HTTP contract between backend (`deep-trust/backend`) and AI Space (`ai`).

## Endpoint

- Method: `POST /predict`
- Content-Type: `multipart/form-data`
- Fields:
  - `file` (required, image file)
  - `threshold` (optional, float)

## Success response (`status=ok`)

```json
{
  "status": "ok",
  "label": "REAL",
  "confidence": 0.82,
  "fake_prob": 0.18,
  "real_prob": 0.82,
  "risk_level": "low",
  "details": {}
}
```

Rules:
- `label` MUST be `REAL` or `FAKE`.
- `confidence`, `fake_prob`, `real_prob` are numeric.
- `status=ok` means image is classifiable.

## Non-classifiable response (`status!=ok`)

```json
{
  "status": "no_face",
  "label": "ERROR",
  "message": "Khong the phan loai anh mot cach an toan. Vui long thu lai voi anh khuon mat ro hon.",
  "error_code": "NO_FACE_DETECTED",
  "face_detected": false,
  "confidence": 0.0,
  "risk_level": "unknown",
  "details": {}
}
```

Rules:
- `label` SHOULD be `ERROR`.
- `status` can be values like `no_face`, `face_detection_error`, `no_model`, `error`.
- Backend must not treat non-`ok` responses as REAL/approved.
