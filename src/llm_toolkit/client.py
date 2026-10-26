"""Thin, typed client for the Ollama HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


@dataclass
class ChatMessage:
    """A single chat turn (``role`` is ``system``, ``user`` or ``assistant``)."""

    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class GenerationResult:
    """Full (non-streamed) response with the timing stats Ollama reports."""

    text: str
    model: str
    eval_count: int = 0
    eval_duration_ns: int = 0
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def tokens_per_second(self) -> float:
        """Generation speed, or 0.0 if the server reported no timing."""
        if not self.eval_duration_ns:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)


class OllamaClient:
    """Minimal client covering model listing, generate, chat and embeddings."""

    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 120.0,
                 session: Optional[requests.Session] = None) -> None:
        self.host = host.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    # -- low level ---------------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any], stream: bool = False) -> requests.Response:
        try:
            resp = self._session.post(f"{self.host}{path}", json=payload,
                                      timeout=self.timeout, stream=stream)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        self._check(resp)
        return resp

    @staticmethod
    def _check(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise OllamaError(f"HTTP {resp.status_code}: {detail}")

    @staticmethod
    def _iter_json_lines(resp: requests.Response) -> Iterator[Dict[str, Any]]:
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise OllamaError(str(data["error"]))
            yield data

    # -- API ---------------------------------------------------------------
    def list_models(self) -> List[str]:
        """Names of the models installed locally."""
        try:
            resp = self._session.get(f"{self.host}/api/tags", timeout=self.timeout)
        except requests.RequestException as exc:
            raise OllamaError(f"cannot reach Ollama at {self.host}: {exc}") from exc
        self._check(resp)
        return [m["name"] for m in resp.json().get("models", [])]

    def generate(self, model: str, prompt: str, system: Optional[str] = None,
                 options: Optional[Dict[str, Any]] = None) -> GenerationResult:
        """Run a single prompt and return the complete answer."""
        payload = self._generate_payload(model, prompt, system, options, stream=False)
        data = self._post("/api/generate", payload).json()
        return GenerationResult(text=data.get("response", ""), model=data.get("model", model),
                                eval_count=data.get("eval_count", 0),
                                eval_duration_ns=data.get("eval_duration", 0), raw=data)

    def stream_generate(self, model: str, prompt: str, system: Optional[str] = None,
                        options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Yield the answer piece by piece as the model produces it."""
        payload = self._generate_payload(model, prompt, system, options, stream=True)
        for chunk in self._iter_json_lines(self._post("/api/generate", payload, stream=True)):
            if chunk.get("response"):
                yield chunk["response"]

    def chat(self, model: str, messages: List[ChatMessage],
             options: Optional[Dict[str, Any]] = None) -> str:
        """Send a conversation and return the assistant's reply."""
        payload = self._chat_payload(model, messages, options, stream=False)
        return self._post("/api/chat", payload).json().get("message", {}).get("content", "")

    def stream_chat(self, model: str, messages: List[ChatMessage],
                    options: Optional[Dict[str, Any]] = None) -> Iterator[str]:
        """Yield the assistant's reply piece by piece."""
        payload = self._chat_payload(model, messages, options, stream=True)
        for chunk in self._iter_json_lines(self._post("/api/chat", payload, stream=True)):
            piece = chunk.get("message", {}).get("content")
            if piece:
                yield piece

    def embed(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed one or more texts; returns one vector per input."""
        data = self._post("/api/embed", {"model": model, "input": texts}).json()
        return data.get("embeddings", [])

    # -- payload builders --------------------------------------------------
    @staticmethod
    def _generate_payload(model: str, prompt: str, system: Optional[str],
                          options: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": stream}
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        return payload

    @staticmethod
    def _chat_payload(model: str, messages: List[ChatMessage],
                      options: Optional[Dict[str, Any]], stream: bool) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "stream": stream,
                                   "messages": [m.to_dict() for m in messages]}
        if options:
            payload["options"] = options
        return payload
