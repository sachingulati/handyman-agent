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


def _route(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        if url.endswith("/api/tags"):
            return _Resp({"models": [{"name": "qwen3:8b"}]})
        if url.endswith("/models"):
            return _Resp({"data": [{"id": "models/gemma-4-31b-it"},
                                   {"id": "models/gemini-2.5-pro"}]})
        return _Resp({}, 404)

    monkeypatch.setattr(requests, "get", fake_get)


def _cfg(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("HM_KEY", "k")
    base = dict(providers={
        "local": {"host": "http://localhost:11434"},
        "cloud": {"host": "https://example/openai", "api_key_env": "HM_KEY"},
    })
    base.update(kw)
    return make_config(tmp_path, **base)


def test_a_discovered_local_model_can_be_used(tmp_path, monkeypatch):
    """Local models are already downloaded and cost nothing to run, so
    discovering one is enough to use it."""
    cfg = _cfg(tmp_path, monkeypatch)
    _route(monkeypatch)
    assert registry.resolve(cfg, "qwen3:8b").provider.name == "local"


def test_a_discovered_hosted_model_is_refused_until_enabled(tmp_path, monkeypatch):
    """Hosted models may bill per token and vary enormously in price. A
    delegating model must not be able to reach one that nobody chose."""
    cfg = _cfg(tmp_path, monkeypatch)
    _route(monkeypatch)
    with pytest.raises(registry.ModelUnavailable, match="not enabled"):
        registry.resolve(cfg, "gemini-2.5-pro")


def test_the_refusal_explains_how_to_enable_it(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    _route(monkeypatch)
    with pytest.raises(registry.ModelUnavailable) as exc:
        registry.resolve(cfg, "gemini-2.5-pro")
    message = str(exc.value)
    assert "models:" in message and "cloud" in message


def test_a_registered_hosted_model_is_usable(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch,
               models=[{"name": "big", "provider": "cloud",
                        "model": "gemma-4-31b-it", "cost": "free"}])
    _route(monkeypatch)
    model = registry.resolve(cfg, "big")
    assert model.provider.name == "cloud"
    assert model.cost == "free"


def test_cost_defaults_to_unknown_rather_than_free(tmp_path, monkeypatch):
    """Silence about price must not read as "free"."""
    cfg = _cfg(tmp_path, monkeypatch,
               models=[{"name": "big", "provider": "cloud", "model": "x"}])
    _route(monkeypatch)
    assert registry.resolve(cfg, "big").cost == "unknown"


def test_local_models_are_reported_as_free(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    _route(monkeypatch)
    assert registry.resolve(cfg, "qwen3:8b").cost == "free"


def test_a_provider_can_opt_into_using_anything_it_offers(tmp_path, monkeypatch):
    """An escape hatch for someone on a flat-rate or self-hosted endpoint,
    who does not want to register every model by hand."""
    monkeypatch.setenv("HM_KEY", "k")
    cfg = make_config(tmp_path, providers={
        "cloud": {"host": "https://example/openai", "api_key_env": "HM_KEY",
                  "allow_discovered": True},
    })
    _route(monkeypatch)
    assert registry.resolve(cfg, "gemini-2.5-pro").provider.name == "cloud"


def test_listing_still_shows_models_that_are_not_enabled(tmp_path, monkeypatch):
    """Visible so they can be found and chosen - hiding them would mean
    nobody could discover what is on offer."""
    cfg = _cfg(tmp_path, monkeypatch)
    _route(monkeypatch)
    names = {m.name: m for m in registry.available(cfg)}
    assert "gemini-2.5-pro" in names
    assert names["gemini-2.5-pro"].enabled is False
    assert names["qwen3:8b"].enabled is True
