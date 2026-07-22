import importlib


def test_defaults(monkeypatch):
    for var in (
        "GEMMA_MAX_CONCURRENT_JOBS",
        "GEMMA_OLLAMA_HOST",
        "GEMMA_MODEL_NAME",
        "GEMMA_MODEL_NAME_MID",
        "GEMMA_MODEL_NAME_BIG",
        "GEMMA_CONTEXT_GROWTH_THRESHOLD_MID_TOKENS",
        "GEMMA_CONTEXT_GROWTH_THRESHOLD_BIG_TOKENS",
        "GEMMA_MAX_ITERATIONS",
        "GEMMA_MAX_WALL_CLOCK_SECONDS",
        "GEMMA_MAX_TOTAL_TOKENS",
        "GEMMA_WATCHDOG_MAX_RETRIES",
        "GEMMA_TAVILY_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    import config
    importlib.reload(config)

    assert config.MAX_CONCURRENT_JOBS == 3
    assert config.OLLAMA_HOST == "http://localhost:11434"
    assert config.MODEL_NAME == "gemma-12b-gpu:latest"
    assert config.MODEL_NAME_MID == "gemma-12b-gpu-mid:latest"
    assert config.MODEL_NAME_BIG == "gemma-12b-gpu-big:latest"
    assert config.CONTEXT_GROWTH_THRESHOLD_MID_TOKENS == 24000
    assert config.CONTEXT_GROWTH_THRESHOLD_BIG_TOKENS == 48000
    assert config.MAX_ITERATIONS == 40
    assert config.MAX_WALL_CLOCK_SECONDS == 1200
    assert config.WATCHDOG_MAX_RETRIES == 3
    assert config.TAVILY_API_KEY is None


def test_env_override(monkeypatch):
    monkeypatch.setenv("GEMMA_MAX_CONCURRENT_JOBS", "5")
    import config
    importlib.reload(config)
    assert config.MAX_CONCURRENT_JOBS == 5


def test_tavily_api_key_from_gemma_specific_env(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("GEMMA_TAVILY_API_KEY", "tvly-gemma-specific")
    import config
    importlib.reload(config)
    assert config.TAVILY_API_KEY == "tvly-gemma-specific"


def test_tavily_api_key_falls_back_to_shared_env_var(monkeypatch):
    monkeypatch.delenv("GEMMA_TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-shared")
    import config
    importlib.reload(config)
    assert config.TAVILY_API_KEY == "tvly-shared"


def test_tavily_api_key_gemma_specific_overrides_shared(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-shared")
    monkeypatch.setenv("GEMMA_TAVILY_API_KEY", "tvly-gemma-specific")
    import config
    importlib.reload(config)
    assert config.TAVILY_API_KEY == "tvly-gemma-specific"
