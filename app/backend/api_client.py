"""Thin HTTP client for the Databricks-hosted AyurGenix RAG API.

The API surface is defined in
``databricks_notebooks/rag_pipeline/06_api_serving.py``:

* ``GET  /health``      -> {"status": "ok"}
* ``GET  /languages``   -> {"<code>": "<human-readable name>", ...}
* ``POST /ask``         -> AskResponse (see below)

This module deliberately has no Streamlit imports so it can be reused from a
CLI, notebook, or test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


DEFAULT_TIMEOUT_S = 60


class AyurGenixAPIError(RuntimeError):
    """Raised for any failure when calling the AyurGenix RAG API."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 detail: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class Source:
    chunk_text: str
    source_file: Optional[str] = None
    page_number: Optional[int] = None
    score: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        return cls(
            chunk_text=str(data.get("chunk_text") or ""),
            source_file=data.get("source_file"),
            page_number=data.get("page_number"),
            score=data.get("score"),
        )


@dataclass(frozen=True)
class AskResult:
    answer: str
    language: str
    detected_language: str
    disclaimer: str
    sources: List[Source]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AskResult":
        return cls(
            answer=str(data.get("answer") or ""),
            language=str(data.get("language") or "en"),
            detected_language=str(data.get("detected_language") or "en"),
            disclaimer=str(data.get("disclaimer") or ""),
            sources=[Source.from_dict(s) for s in (data.get("sources") or [])],
        )


class AyurGenixClient:
    """HTTP client for the Databricks RAG API.

    Parameters
    ----------
    base_url:
        Root URL of the deployed FastAPI service, e.g.
        ``https://<workspace>.cloud.databricks.com/serving-endpoints/<app>/invocations``
        or any plain host where ``uvicorn`` is running. The trailing slash is
        normalized automatically.
    token:
        Optional bearer token. If your API is behind a Databricks personal
        access token / service principal token, supply it here and it will be
        sent as ``Authorization: Bearer <token>``.
    timeout:
        Per-request timeout in seconds. The server itself has a 30s ceiling
        per RAG stage, so 60s end-to-end is a safe default.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        self._session.headers.setdefault("Content-Type", "application/json")
        self._session.headers.setdefault("Accept", "application/json")

    # -- internal -----------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._session.request(
                method, self._url(path), timeout=self._timeout, **kwargs
            )
        except requests.Timeout as exc:
            raise AyurGenixAPIError(
                f"Request to {path} timed out after {self._timeout}s."
            ) from exc
        except requests.ConnectionError as exc:
            raise AyurGenixAPIError(
                f"Could not connect to AyurGenix API at {self._base_url}."
            ) from exc
        except requests.RequestException as exc:
            raise AyurGenixAPIError(f"HTTP error calling {path}: {exc}") from exc

        if resp.status_code >= 400:
            detail = None
            try:
                payload = resp.json()
                detail = payload.get("detail") if isinstance(payload, dict) else None
            except ValueError:
                detail = (resp.text or "").strip()[:500] or None
            raise AyurGenixAPIError(
                f"API returned HTTP {resp.status_code} for {path}.",
                status_code=resp.status_code,
                detail=detail,
            )

        try:
            return resp.json()
        except ValueError as exc:
            raise AyurGenixAPIError(
                f"API returned non-JSON body for {path}: {resp.text[:200]!r}"
            ) from exc

    # -- public surface -----------------------------------------------------

    def health(self) -> Dict[str, str]:
        return self._request("GET", "/health")

    def languages(self) -> Dict[str, str]:
        return self._request("GET", "/languages")

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AskResult:
        if not question or not question.strip():
            raise ValueError("question is required")
        payload: Dict[str, Any] = {"question": question.strip(), "top_k": int(top_k)}
        if source_filter in {"csv", "pdf"}:
            payload["source_filter"] = source_filter
        if language:
            payload["language"] = language
        data = self._request("POST", "/ask", json=payload)
        if not isinstance(data, dict):
            raise AyurGenixAPIError("Unexpected /ask response shape.")
        return AskResult.from_dict(data)
