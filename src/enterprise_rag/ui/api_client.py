"""Thin HTTP client for the Phase 13 API."""
import os
from typing import Any
import requests

class APIClientError(RuntimeError): pass

class APIClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0, session: Any | None = None):
        self.base_url = (base_url or os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout; self.session = session or requests
    def health(self) -> dict[str, Any]: return self._request("GET", "/health")
    def query(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip(): raise APIClientError("Question must not be empty.")
        return self._request("POST", "/query", json={"query": query.strip()})
    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try: response = self.session.request(method, self.base_url + path, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc: raise APIClientError("Unable to connect to the document intelligence service.") from exc
        if response.status_code >= 400:
            detail = response.json().get("detail", "The service could not process the request.") if hasattr(response, "json") else "The service could not process the request."
            raise APIClientError(str(detail))
        try: payload = response.json()
        except Exception as exc: raise APIClientError("The service returned an invalid response.") from exc
        if not isinstance(payload, dict): raise APIClientError("The service returned an invalid response.")
        return payload
