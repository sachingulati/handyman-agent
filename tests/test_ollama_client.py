import pytest
import requests

from handyman import ollama_client
class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json_data


def test_chat_returns_message(monkeypatch):
    fake = FakeResponse(
        json_data={"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    )
    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **k: fake)
    message = ollama_client.chat("http://localhost:11434", "some-model", [], [])
    assert message == {"role": "assistant", "content": "hi"}


def test_chat_raises_ollama_error_on_connection_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "post", raise_conn_error)
    with pytest.raises(ollama_client.OllamaError, match="is it running"):
        ollama_client.chat("http://localhost:11434", "some-model", [], [])


def test_chat_raises_ollama_error_on_missing_model(monkeypatch):
    fake = FakeResponse(status_code=404)
    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **k: fake)
    with pytest.raises(ollama_client.OllamaError, match="pull it"):
        ollama_client.chat("http://localhost:11434", "missing-model", [], [])


def test_model_is_pulled_true(monkeypatch):
    fake = FakeResponse(json_data={"models": [{"name": "some-model"}]})
    monkeypatch.setattr(ollama_client.requests, "get", lambda *a, **k: fake)
    assert ollama_client.model_is_pulled("http://localhost:11434", "some-model") is True


def test_model_is_pulled_false(monkeypatch):
    fake = FakeResponse(json_data={"models": [{"name": "other-model"}]})
    monkeypatch.setattr(ollama_client.requests, "get", lambda *a, **k: fake)
    assert ollama_client.model_is_pulled("http://localhost:11434", "some-model") is False


def test_model_is_pulled_raises_ollama_error_on_connection_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "get", raise_conn_error)
    with pytest.raises(ollama_client.OllamaError, match="is it running"):
        ollama_client.model_is_pulled("http://localhost:11434", "some-model")


def test_pull_model_raises_ollama_error_on_connection_error(monkeypatch):
    def raise_conn_error(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(ollama_client.requests, "post", raise_conn_error)
    with pytest.raises(ollama_client.OllamaError, match="is it running"):
        ollama_client.pull_model("http://localhost:11434", "some-model")


def test_pull_model_posts_to_api_pull(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    ollama_client.pull_model("http://localhost:11434", "some-model")
    assert captured["url"] == "http://localhost:11434/api/pull"
    assert captured["json"] == {"model": "some-model", "stream": False}
