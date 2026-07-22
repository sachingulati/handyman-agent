import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from handyman import config as _config


def make_config(tmp_path, **overrides):
    """Build a Config for tests, defaulting paths under tmp_path.

    Replaces the old pattern of monkeypatching module-level constants,
    which no longer exist now that configuration is loaded rather than
    imported.
    """
    values = dict(
        tiers=[_config.Tier(name="small", model="test-model", threshold_tokens=0)],
        ollama_host="http://localhost:11434",
        max_concurrent_jobs=3,
        max_iterations=40,
        max_wall_clock_seconds=1200,
        watchdog_max_retries=3,
        tavily_api_key=None,
        db_path=tmp_path / "jobs.db",
        jobs_log_dir=tmp_path / "jobs",
    )
    values.update(overrides)
    return _config.Config(**values)


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """A loaded Config pointing at tmp_path, with config.load() patched to
    return it, so code under test picks it up without touching the real
    user config file."""
    c = make_config(tmp_path)
    (tmp_path / "jobs").mkdir(exist_ok=True)
    monkeypatch.setattr(_config, "load", lambda *a, **k: c)
    return c
