import pytest

from conftest import make_config
from handyman import provider


def _cfg(tmp_path, **kw):
    return make_config(tmp_path, **kw)


def test_local_when_no_hosted_is_configured(tmp_path):
    cfg = _cfg(tmp_path, api_key_env="")
    assert provider.choose(cfg, local_available=True, at_capacity=False,
                           allow_hosted=False) == "local"


def test_local_when_a_slot_is_free(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    assert provider.choose(cfg, local_available=True, at_capacity=False,
                           allow_hosted=True) == "local"


def test_queues_at_capacity_without_opt_in(tmp_path, monkeypatch):
    """Today's behaviour is preserved: a busy queue waits rather than
    silently sending work to a third party."""
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    assert provider.choose(cfg, local_available=True, at_capacity=True,
                           allow_hosted=False) == "local"


def test_overflows_to_hosted_at_capacity_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    assert provider.choose(cfg, local_available=True, at_capacity=True,
                           allow_hosted=True) == "hosted"


def test_refuses_rather_than_silently_going_hosted_when_local_is_down(tmp_path, monkeypatch):
    """Sending a task hosted means the task text and whatever files the
    model reads leave the machine. A user who configured a key months ago
    should not have an unrelated outage redirect their code."""
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    with pytest.raises(provider.ProviderUnavailable, match="allow_hosted"):
        provider.choose(cfg, local_available=False, at_capacity=False,
                        allow_hosted=False)


def test_uses_hosted_when_local_is_down_and_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    assert provider.choose(cfg, local_available=False, at_capacity=False,
                           allow_hosted=True) == "hosted"


def test_hosted_only_install_needs_no_opt_in(tmp_path, monkeypatch):
    """With no local tiers there is nothing to opt out of."""
    monkeypatch.setenv("HM_KEY", "k")
    cfg = _cfg(tmp_path, api_key_env="HM_KEY", tiers=[])
    assert provider.choose(cfg, local_available=False, at_capacity=False,
                           allow_hosted=False) == "hosted"


def test_error_names_both_remedies_when_local_is_down(tmp_path):
    cfg = _cfg(tmp_path, api_key_env="")
    with pytest.raises(provider.ProviderUnavailable) as exc:
        provider.choose(cfg, local_available=False, at_capacity=False,
                        allow_hosted=False)
    message = str(exc.value)
    assert "start" in message.lower()


def test_hosted_requires_the_key_to_actually_be_set(tmp_path, monkeypatch):
    """A configured variable name with nothing in it is not a provider."""
    monkeypatch.delenv("HM_KEY", raising=False)
    cfg = _cfg(tmp_path, api_key_env="HM_KEY")
    with pytest.raises(provider.ProviderUnavailable):
        provider.choose(cfg, local_available=False, at_capacity=False,
                        allow_hosted=True)
