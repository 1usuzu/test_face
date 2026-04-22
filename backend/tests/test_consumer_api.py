import api
import consumer_api
from ai_deepfake.detect import DetectionStatus, RiskLevel
from .conftest import DummyDetectionResult, DummyDetector


def _img_file():
    return {"file": ("face.jpg", b"fake-image-bytes", "image/jpeg")}


def test_consumer_ok_real_can_be_approved(client, monkeypatch):
    consumer_api._verified_images.clear()
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.OK,
                is_fake=False,
                confidence=0.92,
                fake_probability=0.08,
                risk_level=RiskLevel.LOW,
                details={"face_detected": True},
            )
        ),
    )

    response = client.post("/v1/consumer/verify-image", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "REAL"
    assert body["decision"] in {"APPROVED", "REVIEW_REQUIRED"}


def test_consumer_no_face_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.NO_FACE,
                is_fake=False,
                confidence=0.0,
                fake_probability=0.0,
                risk_level=RiskLevel.LOW,
                details={"error": "NO_FACE_DETECTED", "face_detected": False},
            )
        ),
    )

    response = client.post("/v1/consumer/verify-image", data={"user_address": "0xabc"}, files=_img_file())
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "no_face"
    assert body["decision"] == "REJECTED"
    assert body["label"] == "ERROR"


def test_consumer_face_detection_error_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.FACE_DETECTION_ERROR,
                is_fake=False,
                confidence=0.0,
                fake_probability=0.0,
                risk_level=RiskLevel.LOW,
                details={"error": "FACE_DETECTION_ERROR: mock", "face_detected": False},
            )
        ),
    )
    response = client.post("/v1/consumer/verify-image", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "face_detection_error"
    assert body["decision"] == "REJECTED"


def test_consumer_generic_error_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.ERROR,
                is_fake=False,
                confidence=0.0,
                fake_probability=0.0,
                risk_level=RiskLevel.LOW,
                details={"error": "runtime failure"},
            )
        ),
    )
    response = client.post("/v1/consumer/verify-image", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"
    assert body["decision"] == "REJECTED"


def test_consumer_no_model_is_rejected(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.NO_MODEL,
                is_fake=False,
                confidence=0.0,
                fake_probability=0.0,
                risk_level=RiskLevel.LOW,
                details={"error": "No model prediction"},
            )
        ),
    )
    response = client.post("/v1/consumer/verify-image", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_model"
    assert body["decision"] == "REJECTED"
