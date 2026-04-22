import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure `import api` resolves backend/api.py
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import api  # noqa: E402
from ai_deepfake.detect import DetectionStatus, RiskLevel  # noqa: E402


@dataclass
class DummyDetectionResult:
    status: DetectionStatus
    is_fake: bool
    confidence: float
    fake_probability: float
    risk_level: RiskLevel
    details: dict


class DummyDetector:
    def __init__(self, result: DummyDetectionResult):
        self._result = result

    def predict(self, _image_path: str):
        return self._result


@pytest.fixture
def client(monkeypatch):
    @asynccontextmanager
    async def _noop_lifespan(_app):
        yield

    monkeypatch.setattr(api.app.router, "lifespan_context", _noop_lifespan)
    monkeypatch.setattr(api, "SERVER_PRIVATE_KEY", "0xa5c7c4ced626ea8b1c899955aae0332823c6846966d5cb2a815154c65068b866")
    monkeypatch.setattr(api, "detector", None)
    monkeypatch.setattr(api, "zkp_oracle", None)
    monkeypatch.setattr(api, "did_service", None)
    monkeypatch.setattr(api, "blockchain_client", None)

    with TestClient(api.app) as test_client:
        yield test_client
