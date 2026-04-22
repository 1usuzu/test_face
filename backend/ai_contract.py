from pydantic import BaseModel, Field


class AIPredictOkResponse(BaseModel):
    status: str = Field(pattern="^ok$")
    label: str = Field(pattern="^(REAL|FAKE)$")
    confidence: float
    fake_prob: float
    real_prob: float
    risk_level: str
    details: dict = Field(default_factory=dict)


class AIPredictErrorResponse(BaseModel):
    status: str
    label: str = Field(default="ERROR")
    message: str
    error_code: str
    face_detected: bool = False
    confidence: float = 0.0
    risk_level: str = "unknown"
    details: dict = Field(default_factory=dict)


def validate_ai_predict_payload(payload: dict) -> dict:
    status = str(payload.get("status", "")).lower()
    if status == "ok":
        return AIPredictOkResponse.model_validate(payload).model_dump()
    return AIPredictErrorResponse.model_validate(payload).model_dump()
