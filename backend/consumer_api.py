"""
consumer_api.py — Consumer-facing API endpoints
=================================================
Provides /v1/consumer/ endpoints that the Consumer App calls.
Wraps the core AI verification with policy-based decisions.
"""

import hashlib
import time
from typing import Optional

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

router = APIRouter(prefix="/v1")

# In-memory store for verified images (production: use a database)
_verified_images: dict[str, dict] = {}

# Policy thresholds (can be overridden via env)
APPROVAL_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.55

# Simple API key auth (loaded from main app)
API_KEYS: set[str] = set()


def _apply_policy(label: str, confidence: float, real_prob: float) -> tuple[str, str]:
    """Apply editorial policy to AI result. Returns (decision, reason)."""
    if label == "REAL" and real_prob >= APPROVAL_THRESHOLD:
        return "APPROVED", f"Image classified as REAL with {real_prob*100:.1f}% confidence (≥ {APPROVAL_THRESHOLD*100:.0f}% threshold)."
    elif label == "REAL" and real_prob >= REVIEW_THRESHOLD:
        return "REVIEW_REQUIRED", f"Image classified as REAL but confidence {real_prob*100:.1f}% is below auto-approval threshold."
    else:
        reason = f"Image classified as {label} with {confidence*100:.1f}% confidence."
        if label == "FAKE":
            reason += " Deepfake indicators detected."
        return "REJECTED", reason


def _check_api_key(x_api_key: Optional[str]):
    if not API_KEYS:
        return  # no keys configured = open access (dev mode)
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _detector_status_value(result) -> str:
    status = getattr(result, "status", None)
    if status is None:
        return "ok"
    return getattr(status, "value", str(status)).lower()


@router.post("/consumer/verify-image")
async def consumer_verify_image(
    file: UploadFile = File(...),
    user_address: str = Form(default="anonymous"),
    external_id: str = Form(default=""),
    x_api_key: Optional[str] = Header(None),
):
    """
    Consumer endpoint: AI verify + policy decision.
    Called by Consumer App's _call_verify().
    """
    _check_api_key(x_api_key)

    # Import detector from main app state
    import api
    if api.detector is None:
        raise HTTPException(status_code=503, detail="AI Detector not initialized")

    import tempfile, os
    temp_file = None
    try:
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large (max 10 MB)")

        image_hash = hashlib.sha256(contents).hexdigest()

        suffix = f".{file.filename.rsplit('.', 1)[-1]}" if file.filename and '.' in file.filename else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(contents)
        temp_file.flush()
        temp_path = temp_file.name
        temp_file.close()

        result = api.detector.predict(temp_path)

        status_value = _detector_status_value(result)
        if status_value != "ok":
            details = result.details if isinstance(result.details, dict) else {}
            error_msg = details.get("error", status_value)
            record = {
                "found": False,
                "image_hash": image_hash,
                "label": "ERROR",
                "confidence": 0.0,
                "real_prob": 0.0,
                "fake_prob": 0.0,
                "risk_level": "unknown",
                "status": status_value,
                "error": error_msg,
                "decision": "REJECTED",
                "reason": f"Image rejected: detector status={status_value} ({error_msg}).",
                "verification_link": f"/verify/{image_hash}",
                "external_id": external_id,
                "verified_at": int(time.time()),
                "filename": file.filename or "unknown",
            }
            _verified_images[image_hash] = record
            return record

        is_real = not result.is_fake
        label = "REAL" if is_real else "FAKE"
        confidence = result.confidence
        # Backward/forward compatibility: some detector versions expose
        # `fake_probability` (new) vs `fake_prob` (legacy).
        fake_prob = getattr(result, "fake_probability", None)
        if fake_prob is None:
            fake_prob = getattr(result, "fake_prob", None)
        if fake_prob is None:
            # Last resort: derive from confidence if detector didn't provide prob
            fake_prob = float(confidence) if not is_real else (1.0 - float(confidence))
        real_prob = 1.0 - fake_prob
        risk_level = result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level)

        decision, reason = _apply_policy(label, confidence, real_prob)

        record = {
            "found": True,
            "image_hash": image_hash,
            "label": label,
            "confidence": confidence,
            "real_prob": real_prob,
            "fake_prob": fake_prob,
            "risk_level": risk_level,
            "decision": decision,
            "reason": reason,
            "verification_link": f"/verify/{image_hash}",
            "external_id": external_id,
            "verified_at": int(time.time()),
            "filename": file.filename or "unknown",
        }
        _verified_images[image_hash] = record

        return record

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[consumer_api] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal verification error: {e}")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@router.get("/consumer/image-status-public/{image_hash}")
async def consumer_image_status_public(image_hash: str):
    """
    Public lookup: check if an image hash was previously verified.
    Called by Consumer App's _call_image_status_public().
    """
    record = _verified_images.get(image_hash)
    if not record:
        return {"found": False, "image_hash": image_hash}
    return record


@router.get("/health")
async def v1_health():
    """Health check endpoint at /v1/health for Consumer App."""
    import api
    return {
        "status": "ok" if api.detector is not None else "degraded",
        "detector": api.detector is not None,
        "zkp_oracle": api.zkp_oracle is not None,
        "did_service": api.did_service is not None,
    }
