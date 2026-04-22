import pytest

import api
from ai_deepfake.detect import DetectionStatus, RiskLevel
from .conftest import DummyDetectionResult, DummyDetector


class _MockZkpOracle:
    oracle_address = "0x0000000000000000000000000000000000000001"

    def create_zkp_input(self, image_hash, is_real, confidence, timestamp):
        from types import SimpleNamespace

        return SimpleNamespace(
            oracle_secret="abc123oracle",
            timestamp=timestamp,
        )


def _img_file():
    return {"file": ("face.jpg", b"fake-image-bytes", "image/jpeg")}


def test_api_verify_ok_result_returns_classification_and_signature(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.OK,
                is_fake=False,
                confidence=0.88,
                fake_probability=0.12,
                risk_level=RiskLevel.LOW,
                details={"face_detected": True},
            )
        ),
    )

    response = client.post("/api/verify", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "REAL"
    assert body["confidence"] == 0.88
    assert body["fake_prob"] == 0.12
    assert "signature" in body and body["signature"]


def test_api_verify_non_ok_status_returns_error_payload(client, monkeypatch):
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

    response = client.post("/api/verify", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_face"
    assert body["label"] == "ERROR"
    assert body["error_code"] == "NO_FACE_DETECTED"
    assert body["face_detected"] is False
    assert "signature" not in body


def test_api_verify_zkp_non_ok_status_never_generates_proof(client, monkeypatch):
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

    response = client.post("/api/verify-zkp", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "face_detection_error"
    assert body["can_generate_proof"] is False
    assert body["label"] == "ERROR"


def test_api_verify_zkp_real_includes_probabilities_parity_with_verify(client, monkeypatch):
    monkeypatch.setattr(api, "zkp_oracle", _MockZkpOracle())
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.OK,
                is_fake=False,
                confidence=0.752,
                fake_probability=0.248,
                risk_level=RiskLevel.LOW,
                details={"face_detected": True},
            )
        ),
    )

    response = client.post("/api/verify-zkp", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["label"] == "REAL"
    assert body["confidence"] == pytest.approx(0.752)
    assert body["fake_prob"] == pytest.approx(0.248)
    assert body["real_prob"] == pytest.approx(0.752)
    assert body["risk_level"] == "low"
    assert body["detector_version"] == api.DETECTOR_VERSION
    assert body["can_generate_proof"] is True
    assert "zkp_input" in body
    assert body["zkp_input"]["oracle_secret"] == "abc123oracle"


def test_api_verify_zkp_fake_includes_probabilities(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "detector",
        DummyDetector(
            DummyDetectionResult(
                status=DetectionStatus.OK,
                is_fake=True,
                confidence=0.85,
                fake_probability=0.85,
                risk_level=RiskLevel.HIGH,
                details={"face_detected": True},
            )
        ),
    )

    response = client.post("/api/verify-zkp", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["label"] == "FAKE"
    assert body["can_generate_proof"] is False
    assert body["fake_prob"] == pytest.approx(0.85)
    assert body["real_prob"] == pytest.approx(0.15)
    assert body["risk_level"] == "high"
    assert "image_hash" in body


def test_api_credential_issue_non_ok_status_rejected(client, monkeypatch):
    class _DummyDidService:
        def __init__(self):
            self.issue_called = False

        def issue_verification_credential(self, **_kwargs):
            self.issue_called = True
            raise AssertionError("issue_verification_credential must not be called when status != ok")

    did_service = _DummyDidService()
    monkeypatch.setattr(
        api,
        "did_service",
        did_service,
    )
    monkeypatch.setattr(api, "DID_AVAILABLE", True)
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

    response = client.post("/api/credential/issue", data={"user_address": "0xabc"}, files=_img_file())
    assert response.status_code == 422
    assert "non-classifiable: no_model" in response.json()["detail"]
    assert did_service.issue_called is False
