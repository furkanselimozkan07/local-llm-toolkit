"""Tests run against a tiny in-process fake Ollama server (no real model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from llm_toolkit import ChatMessage, OllamaClient, OllamaError

REQUESTS = []


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send(200, {"models": [{"name": "llama3:8b"}, {"name": "mistral:7b"}]})
        else:
            self._send(404, {"error": "nope"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append((self.path, body))
        if body["model"] == "missing":
            return self._send(404, {"error": "model 'missing' not found"})
        if self.path == "/api/generate" and body["stream"]:
            lines = [{"response": "Hel"}, {"response": "lo"}, {"response": "", "done": True}]
            self._send(200, "\n".join(map(json.dumps, lines)).encode(), "application/x-ndjson")
        elif self.path == "/api/generate":
            self._send(200, {"model": body["model"], "response": "Hello", "eval_count": 20,
                             "eval_duration": 2_000_000_000})
        elif self.path == "/api/chat" and body["stream"]:
            lines = [{"message": {"role": "assistant", "content": "Hi"}},
                     {"message": {"role": "assistant", "content": "!"}, "done": True}]
            self._send(200, "\n".join(map(json.dumps, lines)).encode(), "application/x-ndjson")
        elif self.path == "/api/chat":
            self._send(200, {"message": {"role": "assistant", "content": "Hi there"}})
        elif self.path == "/api/embed":
            self._send(200, {"embeddings": [[0.1, 0.2] for _ in body["input"]]})
        else:
            self._send(404, {"error": "unknown"})


@pytest.fixture(scope="module")
def client():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield OllamaClient(f"http://127.0.0.1:{server.server_port}", timeout=5)
    server.shutdown()


def test_list_models(client):
    assert client.list_models() == ["llama3:8b", "mistral:7b"]


def test_generate_reports_speed_and_sends_options(client):
    res = client.generate("llama3", "hi", system="be brief", options={"temperature": 0})
    assert res.text == "Hello"
    assert res.tokens_per_second == pytest.approx(10.0)
    path, body = REQUESTS[-1]
    assert path == "/api/generate"
    assert body["system"] == "be brief" and body["options"] == {"temperature": 0}
    assert body["stream"] is False


def test_stream_generate_joins_chunks(client):
    assert "".join(client.stream_generate("llama3", "hi")) == "Hello"


def test_chat_and_stream_chat(client):
    msgs = [ChatMessage("user", "hello")]
    assert client.chat("llama3", msgs) == "Hi there"
    assert REQUESTS[-1][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert "".join(client.stream_chat("llama3", msgs)) == "Hi!"


def test_embed_returns_one_vector_per_text(client):
    assert client.embed("nomic-embed-text", ["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]


def test_http_error_becomes_ollama_error(client):
    with pytest.raises(OllamaError, match="not found"):
        client.generate("missing", "hi")


def test_unreachable_server_raises():
    with pytest.raises(OllamaError, match="cannot reach"):
        OllamaClient("http://127.0.0.1:1", timeout=1).list_models()


def test_tokens_per_second_zero_without_timing():
    from llm_toolkit.client import GenerationResult
    assert GenerationResult(text="", model="m").tokens_per_second == 0.0
