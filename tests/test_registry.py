import pytest
import requests

from conftest import make_config
from handyman import registry


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


LOCAL_TAGS = {"models": [{"name": "qwen3:8b"}, {"name": "gemma4:12b"}]}
HOSTED_LIST = {"data": [{"id": "models/gemma-4-31b-it"}, {"id": "models/gemini-x"}]}


def _cfg(tmp_path, **kw):
    return make_config(tmp_path, **kw)


def _two_providers(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    return _cfg(tmp_path, providers={
        "local": {"host": "http://localhost:11434"},
        "google": {"host": "https://example/openai", "chat_path": "/chat/completions",
                   "api_key_env": "HM_KEY"},
    })


def _route(monkeypatch):
    """Answer each provider's listing endpoint with its own catalogue."""
    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/api/tags"):
            return _Resp(LOCAL_TAGS)
        if url.endswith("/models"):
            return _Resp(HOSTED_LIST)
        return _Resp({}, 404)

    monkeypatch.setattr(requests, "get", fake_get)


# --- discovery ------------------------------------------------------------

def test_discovers_installed_local_models(tmp_path, monkeypatch):
    _route(monkeypatch)
    provider = registry.Provider("local", "http://localhost:11434")
    assert registry.discover(provider) == ["qwen3:8b", "gemma4:12b"]


def test_discovers_hosted_models_and_strips_the_namespace(tmp_path, monkeypatch):
    """A hosted listing returns ids like "models/gemma-4-31b-it"; the
    caller should be able to ask for the plain name."""
    _route(monkeypatch)
    provider = registry.Provider("google", "https://example/openai",
                                 api_key_env="HM_KEY")
    monkeypatch.setenv("HM_KEY", "k")
    assert registry.discover(provider) == ["gemma-4-31b-it", "gemini-x"]


def test_discovery_failure_is_not_fatal(tmp_path, monkeypatch):
    """An unreachable provider must not stop a caller naming a model."""
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("down")

    monkeypatch.setattr(requests, "get", boom)
    assert registry.discover(registry.Provider("local", "http://x")) == []


# --- resolution -----------------------------------------------------------

def test_resolves_a_local_model_to_the_local_provider(tmp_path, monkeypatch):
    cfg = _two_providers(tmp_path, monkeypatch)
    _route(monkeypatch)
    model = registry.resolve(cfg, "qwen3:8b")
    assert model.provider.name == "local"
    assert model.provider.hosted is False


def test_resolves_a_hosted_model_to_the_hosted_provider(tmp_path, monkeypatch):
    """The point of the registry: asking for a hosted model selects the
    hosted endpoint, instead of sending it to whichever host was configured.

    Registered here because a discovered hosted model is deliberately not
    reachable until someone has chosen it - hosted pricing ranges from
    free to expensive."""
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, providers={
        "local": {"host": "http://localhost:11434"},
        "google": {"host": "https://example/openai", "chat_path": "/chat/completions",
                   "api_key_env": "HM_KEY"},
    }, models=[{"name": "gemma-4-31b-it", "provider": "google",
                "model": "gemma-4-31b-it", "cost": "free"}])
    _route(monkeypatch)
    model = registry.resolve(cfg, "gemma-4-31b-it")
    assert model.provider.name == "google"
    assert model.provider.hosted is True
    assert model.provider.chat_path == "/chat/completions"
    assert model.cost == "free"


def test_unknown_model_lists_what_is_available(tmp_path, monkeypatch):
    cfg = _two_providers(tmp_path, monkeypatch)
    _route(monkeypatch)
    with pytest.raises(registry.ModelUnavailable) as exc:
        registry.resolve(cfg, "no-such-model")
    message = str(exc.value)
    assert "no-such-model" in message
    assert "qwen3:8b" in message


def test_provider_filter_narrows_the_search(tmp_path, monkeypatch):
    cfg = _two_providers(tmp_path, monkeypatch)
    _route(monkeypatch)
    with pytest.raises(registry.ModelUnavailable):
        registry.resolve(cfg, "qwen3:8b", provider_name="hosted")


def test_explicit_registration_gives_an_alias(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path,
               providers={"google": {"host": "https://example/openai",
                                     "chat_path": "/chat/completions",
                                     "api_key_env": "HM_KEY"}},
               models=[{"name": "big", "provider": "google",
                        "model": "gemma-4-31b-it"}])
    _route(monkeypatch)
    model = registry.resolve(cfg, "big")
    assert model.model_id == "gemma-4-31b-it"
    assert model.provider.name == "google"


def test_registration_naming_an_unknown_provider_is_rejected(tmp_path):
    cfg = _cfg(tmp_path, models=[{"name": "x", "provider": "nope", "model": "m"}])
    with pytest.raises(registry.ModelUnavailable, match="not configured"):
        registry.registered(cfg)


def test_falls_back_to_the_configured_tier_when_no_name_is_given(tmp_path, monkeypatch):
    """A configured model is trusted even when discovery cannot see it.

    Discovery is a convenience and can fail - the server may be briefly
    down, or a provider may not support listing - and that must not make a
    deliberately configured model unusable."""
    cfg = _cfg(tmp_path, providers={"local": {"host": "http://localhost:11434"}})
    _route(monkeypatch)
    model = registry.resolve(cfg, None)
    assert model.model_id == cfg.tiers[0].model
    assert model.provider.name == "local"


def test_a_config_without_providers_still_works(tmp_path, monkeypatch):
    """Existing configs predate the providers section and must keep working."""
    cfg = _cfg(tmp_path)
    _route(monkeypatch)
    providers = registry.providers_from_config(cfg)
    assert len(providers) == 1
    assert providers[0].host == cfg.ollama_host


def test_available_prefers_a_registered_alias_over_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path,
               providers={"local": {"host": "http://localhost:11434"}},
               models=[{"name": "qwen3:8b", "provider": "local",
                        "model": "pinned-variant"}])
    _route(monkeypatch)
    match = [m for m in registry.available(cfg) if m.name == "qwen3:8b"]
    assert len(match) == 1
    assert match[0].model_id == "pinned-variant"
