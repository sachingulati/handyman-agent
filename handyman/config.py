"""Configuration: defaults, overlaid by a YAML file, overlaid by env vars.

The tier list is data rather than code so a machine can run one, two or
three model tiers without a code change. worker.main turns it into the
(threshold, name, chat_fn) triples run_job already consumes.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

APP_NAME = "handyman"

DEFAULTS = {
    "ollama_host": "http://localhost:11434",
    "max_concurrent_jobs": 3,
    "max_iterations": 40,
    "max_wall_clock_seconds": 20 * 60,
    "watchdog_max_retries": 3,
    # A single chat request must cover a cold model load, which for a
    # large model on a machine that cannot hold it in VRAM is well over a
    # minute before any tokens are produced. The old 120s default killed
    # jobs mid-request that were working correctly.
    "request_timeout_seconds": 900,
    # Reasoning models spend their budget deliberating before acting. On a
    # large or ambiguous task that can consume the whole request, and the
    # deliberation has been observed leaking into generated source. The
    # control is effectively binary: "none" disables it, intermediate
    # settings behave like the default.
    "reasoning_effort": "none",
    # Hosted providers speak the same OpenAI-compatible shape but mount it
    # at a different path and need a bearer token. Empty api_key means
    # local: no auth header, and the model must be pulled before use.
    "chat_path": "/v1/chat/completions",
    "api_key_env": "",
}

# Env var -> (config key, type). Env always wins over the file.
ENV_OVERRIDES = {
    "HANDYMAN_OLLAMA_HOST": ("ollama_host", str),
    "HANDYMAN_MAX_CONCURRENT_JOBS": ("max_concurrent_jobs", int),
    "HANDYMAN_MAX_ITERATIONS": ("max_iterations", int),
    "HANDYMAN_MAX_WALL_CLOCK_SECONDS": ("max_wall_clock_seconds", int),
    "HANDYMAN_WATCHDOG_MAX_RETRIES": ("watchdog_max_retries", int),
    "HANDYMAN_REQUEST_TIMEOUT_SECONDS": ("request_timeout_seconds", int),
    "HANDYMAN_REASONING_EFFORT": ("reasoning_effort", str),
    "HANDYMAN_CHAT_PATH": ("chat_path", str),
    "HANDYMAN_API_KEY_ENV": ("api_key_env", str),
}

BASE_TIER_NAME = "small"


class ConfigError(Exception):
    """Raised when a config file is present but unusable."""


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    threshold_tokens: int


@dataclass(frozen=True)
class Config:
    tiers: list
    ollama_host: str
    max_concurrent_jobs: int
    max_iterations: int
    max_wall_clock_seconds: int
    watchdog_max_retries: int
    request_timeout_seconds: int
    reasoning_effort: str
    chat_path: str
    api_key_env: str
    tavily_api_key: str | None
    db_path: Path
    jobs_log_dir: Path


def default_config_path() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME / "config.yaml"


def default_data_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / APP_NAME


def parse_tiers(raw) -> list:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigError("'tiers' must be a list")
    if not 1 <= len(raw) <= 3:
        raise ConfigError(f"config must define 1 to 3 tiers, got {len(raw)}")

    tiers = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"tier {index} must be a mapping")
        for key in ("name", "model"):
            if not entry.get(key):
                raise ConfigError(f"tier {index} is missing required key '{key}'")
        tiers.append(
            Tier(
                name=str(entry["name"]),
                model=str(entry["model"]),
                threshold_tokens=int(entry.get("threshold_tokens", 0)),
            )
        )

    # run_job initializes current_tier to db.BASE_TIER, and
    # try_claim_with_cap refuses to admit a job whose tier differs from a
    # running one. A differently-named first tier would leave every job on
    # a tier no config entry matches, silently disabling cross-job tier
    # blocking - the model-thrashing bug that took 45 reloads to find.
    if tiers[0].name != BASE_TIER_NAME:
        raise ConfigError(
            f"the first tier must be named '{BASE_TIER_NAME}', got '{tiers[0].name}'"
        )
    if tiers[0].threshold_tokens != 0:
        raise ConfigError("the first tier must have threshold_tokens: 0")

    thresholds = [t.threshold_tokens for t in tiers]
    if thresholds != sorted(thresholds) or len(set(thresholds)) != len(thresholds):
        raise ConfigError(
            f"tier threshold_tokens must be strictly ascending, got {thresholds}"
        )
    return tiers


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from a .env beside the project, without
    overwriting anything already set in the real environment."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def api_key_for(cfg) -> str | None:
    """The bearer token for a hosted provider, or None when running local."""
    if not cfg.api_key_env:
        return None
    return os.environ.get(cfg.api_key_env) or None


def load(path=None) -> Config:
    _load_dotenv()
    if path is None:
        path = Path(os.environ.get("HANDYMAN_CONFIG") or default_config_path())
    path = Path(path)

    raw = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ConfigError(f"{path} must contain a YAML mapping at the top level")
            raw = loaded

    values = dict(DEFAULTS)
    for key in DEFAULTS:
        if key in raw:
            values[key] = raw[key]
    for env_var, (key, caster) in ENV_OVERRIDES.items():
        if env_var in os.environ:
            values[key] = caster(os.environ[env_var])

    data_dir = Path(os.environ.get("HANDYMAN_DATA_DIR") or default_data_dir())

    # A handyman-specific key wins so this tool can be pointed at its own
    # key or quota; a plain TAVILY_API_KEY already in the environment is
    # accepted as a convenience so a second key isn't required.
    tavily = (
        os.environ.get("HANDYMAN_TAVILY_API_KEY")
        or os.environ.get("TAVILY_API_KEY")
        or None
    )

    return Config(
        tiers=parse_tiers(raw.get("tiers")),
        tavily_api_key=tavily,
        db_path=Path(os.environ.get("HANDYMAN_DB_PATH") or data_dir / "jobs.db"),
        jobs_log_dir=Path(os.environ.get("HANDYMAN_JOBS_LOG_DIR") or data_dir / "jobs"),
        **values,
    )
