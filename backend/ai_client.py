import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from ai_contract import validate_ai_predict_payload


class AIServiceError(Exception):
    pass


@dataclass
class AIClientConfig:
    base_url: str
    timeout_seconds: float = 20.0
    retry_count: int = 2
    retry_delay_seconds: float = 0.4


class AISpaceClient:
    def __init__(self, config: AIClientConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "AISpaceClient | None":
        base_url = (os.environ.get("AI_SPACE_URL") or "").strip().rstrip("/")
        if not base_url:
            return None
        timeout_seconds = float(os.environ.get("AI_TIMEOUT_SECONDS", "20"))
        retry_count = int(os.environ.get("AI_RETRY_COUNT", "2"))
        retry_delay_seconds = float(os.environ.get("AI_RETRY_DELAY_SECONDS", "0.4"))
        return cls(
            AIClientConfig(
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                retry_count=retry_count,
                retry_delay_seconds=retry_delay_seconds,
            )
        )

    async def health(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            resp = await client.get(f"{self.config.base_url}/health")
        if resp.status_code >= 400:
            raise AIServiceError(f"AI health check failed: {resp.status_code}")
        return resp.json()

    async def predict(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        data = {}
        if threshold is not None:
            data["threshold"] = str(threshold)

        files = {
            "file": (filename or "upload.jpg", file_bytes, content_type or "application/octet-stream"),
        }

        last_error: Exception | None = None
        total_attempts = self.config.retry_count + 1
        for attempt in range(total_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                    resp = await client.post(
                        f"{self.config.base_url}/predict",
                        data=data,
                        files=files,
                    )

                if resp.status_code >= 500:
                    raise AIServiceError(f"AI service 5xx: {resp.status_code}")

                if resp.status_code >= 400:
                    detail = self._extract_error_detail(resp)
                    raise AIServiceError(f"AI service request rejected: {detail}")

                payload = resp.json()
                return self._validate_payload(payload)
            except (httpx.TimeoutException, httpx.TransportError, AIServiceError) as exc:
                last_error = exc
                if attempt >= self.config.retry_count:
                    break
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))

        raise AIServiceError(str(last_error) if last_error else "Unknown AI service error")

    @staticmethod
    def _extract_error_detail(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            if isinstance(body, dict):
                if "detail" in body:
                    return str(body["detail"])
                return str(body)
        except Exception:
            pass
        return f"HTTP {resp.status_code}"

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return validate_ai_predict_payload(payload)
        except ValidationError as exc:
            raise AIServiceError(f"AI response schema invalid: {exc}") from exc
