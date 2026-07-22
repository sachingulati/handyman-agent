import os
from pathlib import Path

import pytest

from handyman import config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_single_tier(tmp_path):
    path = _write(tmp_path, """
tiers:
  - name: small
    model: qwen3:8b
    threshold_tokens: 0
""")
    cfg = config.load(path)
    assert len(cfg.tiers) == 1
    assert cfg.tiers[0].name == "small"
    assert cfg.tiers[0].model == "qwen3:8b"
    assert cfg.tiers[0].threshold_tokens == 0


def test_loads_three_tiers_in_order(tmp_path):
    path = _write(tmp_path, """
tiers:
  - name: small
    model: m-small
    threshold_tokens: 0
  - name: mid
    model: m-mid
    threshold_tokens: 24000
  - name: big
    model: m-big
    threshold_tokens: 48000
""")
    cfg = config.load(path)
    assert [t.name for t in cfg.tiers] == ["small", "mid", "big"]
    assert [t.threshold_tokens for t in cfg.tiers] == [0, 24000, 48000]


def test_defaults_apply_when_file_omits_them(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: m, threshold_tokens: 0}
""")
    cfg = config.load(path)
    assert cfg.ollama_host == "http://localhost:11434"
    assert cfg.max_concurrent_jobs == 3
    assert cfg.max_iterations == 40
    assert cfg.max_wall_clock_seconds == 1200
    assert cfg.watchdog_max_retries == 3


def test_file_values_override_defaults(tmp_path):
    path = _write(tmp_path, """
ollama_host: http://elsewhere:1234
max_concurrent_jobs: 7
tiers:
  - {name: small, model: m, threshold_tokens: 0}
""")
    cfg = config.load(path)
    assert cfg.ollama_host == "http://elsewhere:1234"
    assert cfg.max_concurrent_jobs == 7


def test_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDYMAN_MAX_CONCURRENT_JOBS", "9")
    path = _write(tmp_path, """
max_concurrent_jobs: 7
tiers:
  - {name: small, model: m, threshold_tokens: 0}
""")
    assert config.load(path).max_concurrent_jobs == 9


def test_missing_file_yields_defaults_and_no_tiers(tmp_path):
    cfg = config.load(tmp_path / "nope.yaml")
    assert cfg.tiers == []
    assert cfg.max_concurrent_jobs == 3


def test_rejects_more_than_three_tiers(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 0}
  - {name: b, model: b, threshold_tokens: 1}
  - {name: c, model: c, threshold_tokens: 2}
  - {name: d, model: d, threshold_tokens: 3}
""")
    with pytest.raises(config.ConfigError, match="1 to 3 tiers"):
        config.load(path)


def test_rejects_first_tier_not_named_small(tmp_path):
    """run_job initializes current_tier to db.BASE_TIER and
    try_claim_with_cap blocks claims when running jobs disagree on tier, so
    a differently-named first tier silently breaks cross-job tier blocking."""
    path = _write(tmp_path, """
tiers:
  - {name: tiny, model: a, threshold_tokens: 0}
""")
    with pytest.raises(config.ConfigError, match="must be named 'small'"):
        config.load(path)


def test_rejects_nonzero_first_threshold(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 500}
""")
    with pytest.raises(config.ConfigError, match="threshold_tokens: 0"):
        config.load(path)


def test_rejects_non_ascending_thresholds(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, model: a, threshold_tokens: 0}
  - {name: mid, model: b, threshold_tokens: 40000}
  - {name: big, model: c, threshold_tokens: 20000}
""")
    with pytest.raises(config.ConfigError, match="ascending"):
        config.load(path)


def test_rejects_tier_missing_model(tmp_path):
    path = _write(tmp_path, """
tiers:
  - {name: small, threshold_tokens: 0}
""")
    with pytest.raises(config.ConfigError, match="model"):
        config.load(path)


def test_tavily_key_prefers_handyman_specific_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "shared")
    monkeypatch.setenv("HANDYMAN_TAVILY_API_KEY", "specific")
    assert config.load(tmp_path / "nope.yaml").tavily_api_key == "specific"


def test_tavily_key_falls_back_to_shared_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HANDYMAN_TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "shared")
    assert config.load(tmp_path / "nope.yaml").tavily_api_key == "shared"


def test_tavily_key_is_none_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("HANDYMAN_TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert config.load(tmp_path / "nope.yaml").tavily_api_key is None


@pytest.mark.skipif(
    os.name == "nt",
    reason="patching os.name to posix makes pathlib build PosixPath, which "
           "Windows cannot instantiate; the CI matrix covers this on Linux/macOS",
)
def test_default_config_path_is_platform_appropriate(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert config.default_config_path() == Path("/tmp/xdg/handyman/config.yaml")


@pytest.mark.skipif(
    os.name == "nt",
    reason="patching os.name to posix makes pathlib build PosixPath, which "
           "Windows cannot instantiate; the CI matrix covers this on Linux/macOS",
)
def test_default_data_dir_is_platform_appropriate(monkeypatch):
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setenv("XDG_DATA_HOME", "/tmp/share")
    assert config.default_data_dir() == Path("/tmp/share/handyman")


@pytest.mark.skipif(os.name != "nt", reason="Windows path layout")
def test_default_paths_use_appdata_on_windows(monkeypatch):
    monkeypatch.setenv("APPDATA", r"C:\Roaming")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Local")
    assert config.default_config_path() == Path(r"C:\Roaming") / "handyman" / "config.yaml"
    assert config.default_data_dir() == Path(r"C:\Local") / "handyman"
