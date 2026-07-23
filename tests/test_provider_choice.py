import pytest

from conftest import make_config
from handyman import provider


def _cfg(tmp_path, **kw):
    return make_config(tmp_path, **kw)


def _hosted(tmp_path, monkeypatch, **kw):
    monkeypatch.setenv("HM_KEY", "k")
    return _cfg(tmp_path, api_key_env="HM_KEY", **kw)


# --- the caller asked for something specific ------------------------------

def test_explicit_local_is_honoured(tmp_path, monkeypatch):
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested="local", local_available=True,
                           at_capacity=False) == "local"


def test_explicit_hosted_is_honoured(tmp_path, monkeypatch):
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested="hosted", local_available=True,
                           at_capacity=False) == "hosted"


def test_explicit_hosted_is_honoured_even_at_capacity(tmp_path, monkeypatch):
    """Asking for hosted is the whole point of overflow; the local cap is
    irrelevant to work that runs on someone else's hardware."""
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested="hosted", local_available=True,
                           at_capacity=True) == "hosted"


def test_explicit_local_still_queues_at_capacity(tmp_path, monkeypatch):
    """Asking for local means local. A busy queue waits rather than being
    silently redirected off the machine."""
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested="local", local_available=True,
                           at_capacity=True) == "local"


def test_explicit_hosted_without_a_key_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("HM_KEY", raising=False)
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    with pytest.raises(provider.ProviderUnavailable, match="no API key"):
        provider.choose(cfg, requested="hosted", local_available=True,
                        at_capacity=False)


def test_explicit_local_with_the_server_down_is_refused(tmp_path, monkeypatch):
    cfg = _hosted(tmp_path, monkeypatch)
    with pytest.raises(provider.ProviderUnavailable, match="not reachable"):
        provider.choose(cfg, requested="local", local_available=False,
                        at_capacity=False)


def test_an_unknown_provider_name_is_refused(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(provider.ProviderUnavailable, match="unknown provider"):
        provider.choose(cfg, requested="magic", local_available=True,
                        at_capacity=False)


# --- the caller did not say ----------------------------------------------

def test_defaults_to_local_when_nothing_is_requested(tmp_path, monkeypatch):
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested=None, local_available=True,
                           at_capacity=False) == "local"


def test_default_queues_rather_than_going_hosted_by_itself(tmp_path, monkeypatch):
    """Hosted work leaves the machine. That is never chosen on the
    caller's behalf, however convenient it would be."""
    cfg = _hosted(tmp_path, monkeypatch)
    assert provider.choose(cfg, requested=None, local_available=True,
                           at_capacity=True) == "local"


def test_default_refuses_and_names_the_option_when_local_is_down(tmp_path, monkeypatch):
    cfg = _hosted(tmp_path, monkeypatch)
    with pytest.raises(provider.ProviderUnavailable) as exc:
        provider.choose(cfg, requested=None, local_available=False, at_capacity=False)
    assert 'provider="hosted"' in str(exc.value)


def test_hosted_only_install_defaults_to_hosted(tmp_path, monkeypatch):
    """With no local tiers there is nothing to opt out of."""
    cfg = _hosted(tmp_path, monkeypatch, tiers=[])
    assert provider.choose(cfg, requested=None, local_available=False,
                           at_capacity=False) == "hosted"


def test_no_provider_at_all_is_reported_clearly(tmp_path):
    cfg = _cfg(tmp_path, tiers=[], api_key_env="")
    with pytest.raises(provider.ProviderUnavailable, match="handyman setup"):
        provider.choose(cfg, requested=None, local_available=False, at_capacity=False)


# --- model selection ------------------------------------------------------

def test_model_defaults_to_the_first_configured_tier(tmp_path):
    cfg = _cfg(tmp_path)
    assert provider.resolve_model(cfg, requested=None) == cfg.tiers[0].model


def test_an_explicitly_requested_model_wins(tmp_path):
    cfg = _cfg(tmp_path)
    assert provider.resolve_model(cfg, requested="qwen3:14b") == "qwen3:14b"


def test_a_requested_model_need_not_be_in_the_config(tmp_path):
    """The caller may know about a model this install has never used;
    refusing it would make the argument useless."""
    cfg = _cfg(tmp_path)
    assert provider.resolve_model(cfg, requested="something:new") == "something:new"


def test_resolving_a_model_with_no_tiers_and_no_request_fails(tmp_path):
    cfg = _cfg(tmp_path, tiers=[])
    with pytest.raises(provider.ProviderUnavailable, match="no model"):
        provider.resolve_model(cfg, requested=None)
