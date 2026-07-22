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


def test_chat_retries_on_rate_limit_then_succeeds(monkeypatch):
    """A hosted provider rate-limits per minute and a burst of jobs hits it
    routinely. These are transient, so a retry turns a failed job into a
    slow one rather than losing the work."""
    calls = []

    class _R:
        def __init__(self, code, payload=None):
            self.status_code = code
            self._p = payload or {"choices": [{"message": {"content": "ok"}}]}

        def json(self):
            return self._p

        def raise_for_status(self):
            pass

    def fake_post(*a, **k):
        calls.append(1)
        return _R(429) if len(calls) < 3 else _R(200)

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda s: None)
    msg = ollama_client.chat("http://h", "m", [], [])
    assert msg == {"content": "ok"}
    assert len(calls) == 3


def test_chat_gives_up_after_repeated_rate_limits(monkeypatch):
    class _R:
        status_code = 429

        def json(self):
            return {}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(ollama_client.requests, "post", lambda *a, **k: _R())
    monkeypatch.setattr(ollama_client.time, "sleep", lambda s: None)
    with pytest.raises(ollama_client.OllamaError, match="rate limited"):
        ollama_client.chat("http://h", "m", [], [])


def test_chat_does_not_retry_when_the_input_is_too_large(monkeypatch):
    """A provider answers 429 both for throttling and for an input over the
    per-request allowance. Only the second is hopeless: retrying the same
    oversized request fails identically, just slower and mislabelled."""
    calls = []

    class _R:
        status_code = 429

        def json(self):
            return [{"error": {"message":
                     "Quota exceeded for metric: ...input_token_count, limit: 16000"}}]

        def raise_for_status(self):
            pass

    def fake_post(*a, **k):
        calls.append(1)
        return _R()

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda s: None)
    # The request must actually be near the stated limit. The metric name
    # alone is ambiguous: the same name covers a per-minute quota, and
    # treating that as fatal breaks retry for the common case.
    huge = [{"role": "user", "content": "x" * 80000}]
    with pytest.raises(ollama_client.OllamaError, match="split it into smaller steps"):
        ollama_client.chat("http://h", "m", huge, [])
    assert len(calls) == 1, "an oversized request must not be retried"


def test_chat_honours_a_provider_supplied_retry_delay(monkeypatch):
    slept = []
    calls = []

    class _R:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            return {"error": {"message": "rate limited. Please retry in 7.5s"}}

        def raise_for_status(self):
            pass

    class _OK(_R):
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(*a, **k):
        calls.append(1)
        return _R(429) if len(calls) == 1 else _OK(200)

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda s: slept.append(s))
    ollama_client.chat("http://h", "m", [], [])
    assert slept == [7.5], f"should wait the suggested 7.5s, waited {slept}"


def test_chat_retries_a_quota_429_even_though_it_names_the_token_metric(monkeypatch):
    """A per-minute token quota reports the same metric as a per-request
    size cap. A small request hitting it is throttling, not an oversized
    request, and must still be retried."""
    calls = []

    class _R:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            if self.status_code == 429:
                return [{"error": {"message":
                         "Quota exceeded for metric: ...input_token_count, limit: 16000"}}]
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            pass

    def fake_post(*a, **k):
        calls.append(1)
        return _R(429) if len(calls) == 1 else _R(200)

    monkeypatch.setattr(ollama_client.requests, "post", fake_post)
    monkeypatch.setattr(ollama_client.time, "sleep", lambda s: None)
    ollama_client.chat("http://h", "m", [{"role": "user", "content": "tiny"}], [])
    assert len(calls) == 2, "a small request hitting a quota window must retry"
