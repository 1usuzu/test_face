import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from ai_deepfake.ai_config import settings
from ai_deepfake.detect import (
    DeepfakeDetector,
    DetectionResult,
    DetectionStatus,
    RiskLevel,
)


class TestDetectionStatusEnum:
    def test_detection_status_enum_values(self):
        assert DetectionStatus.OK.value == "ok"
        assert DetectionStatus.NO_FACE.value == "no_face"
        assert DetectionStatus.FACE_DETECTION_ERROR.value == "face_detection_error"
        assert DetectionStatus.NO_MODEL.value == "no_model"
        assert DetectionStatus.ERROR.value == "error"
        assert len(DetectionStatus) == 5


class TestRiskLevelEnum:
    def test_risk_level_enum_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"
        assert len(RiskLevel) == 4


class TestDetectionResult:
    def test_detection_result_to_dict(self):
        result = DetectionResult(
            is_fake=True,
            confidence=0.85,
            fake_probability=0.85,
            risk_level=RiskLevel.CRITICAL,
            processing_time=0.123,
            details={"face_detected": True, "model_score": 0.83},
        )
        d = result.to_dict()
        assert d["is_fake"] is True
        assert d["confidence"] == 0.85
        assert d["fake_probability"] == 0.85
        assert d["risk_level"] == "critical"
        assert d["status"] == "ok"
        assert d["processing_time"] == 0.123
        assert d["details"]["face_detected"] is True

    def test_detection_result_fake_prob_property(self):
        result = DetectionResult(
            is_fake=True,
            confidence=0.9,
            fake_probability=0.9,
            risk_level=RiskLevel.CRITICAL,
            processing_time=0.1,
            details={},
        )
        assert result.fake_prob == result.fake_probability

    def test_detection_result_real_prob_property(self):
        result = DetectionResult(
            is_fake=False,
            confidence=0.8,
            fake_probability=0.2,
            risk_level=RiskLevel.LOW,
            processing_time=0.1,
            details={},
        )
        assert result.real_prob == pytest.approx(1.0 - result.fake_probability)


class TestConfig:
    def test_config_defaults(self):
        assert settings.DEFAULT_THRESHOLD == 0.65
        assert settings.V1_WEIGHT == 0.4
        assert settings.V2_WEIGHT == 0.6
        assert settings.ENABLE_TTA is False
        assert settings.ENABLE_SIGNAL_ANALYSIS is True

    def test_config_signal_constants(self):
        assert settings.SIGNAL_LAPLACIAN_THRESHOLD == 100.0
        assert settings.SIGNAL_HIGH_FREQ_THRESHOLD == 13.0
        assert settings.SIGNAL_BOOST_STEP == 0.03


class _FakeModel:
    def __init__(self, fake_prob: float):
        # Build logits whose softmax gives exactly fake_prob at index 0.
        self._logits = torch.log(torch.tensor([[fake_prob, 1.0 - fake_prob]], dtype=torch.float32))

    def __call__(self, _x):
        return self._logits


class _FakeMTCNNNoFace:
    def detect(self, _img):
        return None, None


class _FakeMTCNNRaises:
    def detect(self, _img):
        raise RuntimeError("mock mtcnn failure")


def _make_test_detector(
    fake_prob_v1: float | None = 0.9,
    fake_prob_v2: float | None = None,
    threshold: float = 0.65,
    mtcnn=None,
    signal_boost: float = 0.0,
) -> DeepfakeDetector:
    if fake_prob_v1 is not None and fake_prob_v2 is None:
        # Mirror V1 into V2 so ensemble output matches desired fake probability.
        fake_prob_v2 = fake_prob_v1

    detector = object.__new__(DeepfakeDetector)
    detector.device = "cpu"
    detector.threshold = threshold
    detector.model_v1 = _FakeModel(fake_prob_v1) if fake_prob_v1 is not None else None
    detector.model_v2 = _FakeModel(fake_prob_v2) if fake_prob_v2 is not None else None
    detector.tx_v1 = lambda _img: torch.zeros((3, 224, 224), dtype=torch.float32)
    detector.tx_v2 = lambda _img: torch.zeros((3, 380, 380), dtype=torch.float32)
    detector.mtcnn = mtcnn
    detector._analyze_signal = lambda _np: signal_boost
    return detector


def _make_temp_image() -> str:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.new("RGB", (32, 32), color=(20, 30, 40)).save(tmp.name)
        return tmp.name


class TestInferenceContract:
    def test_fake_high_probability_case(self):
        detector = _make_test_detector(fake_prob_v1=0.9)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.status == DetectionStatus.OK
        assert result.is_fake is True
        assert result.fake_probability == pytest.approx(0.9, abs=1e-6)
        assert result.confidence == pytest.approx(0.9, abs=1e-6)

    def test_real_low_fake_probability_case(self):
        detector = _make_test_detector(fake_prob_v1=0.2)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.status == DetectionStatus.OK
        assert result.is_fake is False
        assert result.fake_probability == pytest.approx(0.2, abs=1e-6)
        assert result.confidence == pytest.approx(0.8, abs=1e-6)

    @pytest.mark.parametrize(
        "fake_prob,expected_is_fake,expected_confidence",
        [(0.8, True, 0.8), (0.2, False, 0.8)],
    )
    def test_confidence_semantics_for_real_vs_fake(self, fake_prob, expected_is_fake, expected_confidence):
        detector = _make_test_detector(fake_prob_v1=fake_prob)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.is_fake is expected_is_fake
        assert result.confidence == pytest.approx(expected_confidence, abs=1e-6)

    def test_no_face_case_returns_status_no_face(self):
        detector = _make_test_detector(fake_prob_v1=0.9, mtcnn=_FakeMTCNNNoFace())
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.status == DetectionStatus.NO_FACE
        assert result.details["error"] == "NO_FACE_DETECTED"
        assert result.details["face_detected"] is False

    def test_face_detector_exception_returns_status_face_detection_error(self):
        detector = _make_test_detector(fake_prob_v1=0.9, mtcnn=_FakeMTCNNRaises())
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.status == DetectionStatus.FACE_DETECTION_ERROR
        assert result.details["error"].startswith("FACE_DETECTION_ERROR:")
        assert result.details["face_detected"] is False

    def test_signal_boost_disabled_respects_config(self, monkeypatch):
        detector = _make_test_detector(fake_prob_v1=0.5, signal_boost=0.4)
        monkeypatch.setattr(settings, "ENABLE_SIGNAL_ANALYSIS", False)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.fake_probability == pytest.approx(0.5, abs=1e-6)
        assert result.details["signal_boost"] == pytest.approx(0.0, abs=1e-6)

    def test_threshold_exact_boundary(self):
        detector = _make_test_detector(fake_prob_v1=0.5, threshold=0.5)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.is_fake is True

    def test_threshold_below_boundary(self):
        detector = _make_test_detector(fake_prob_v1=0.499, threshold=0.5)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.is_fake is False

    @pytest.mark.parametrize(
        "fake_prob,expected_risk",
        [
            (0.86, RiskLevel.CRITICAL),
            (0.66, RiskLevel.HIGH),
            (0.41, RiskLevel.MEDIUM),
            (0.39, RiskLevel.LOW),
        ],
    )
    def test_risk_level_boundaries(self, fake_prob, expected_risk):
        detector = _make_test_detector(fake_prob_v1=fake_prob)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.risk_level == expected_risk

    def test_no_model_path_sets_status_and_legacy_error(self):
        detector = _make_test_detector(fake_prob_v1=None, fake_prob_v2=None)
        image_path = _make_temp_image()
        try:
            result = detector.predict(image_path)
        finally:
            Path(image_path).unlink(missing_ok=True)

        assert result.status == DetectionStatus.NO_MODEL
        assert result.details["error"] == "No model prediction"
