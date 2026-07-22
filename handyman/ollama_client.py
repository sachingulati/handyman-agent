import random
import time

import requests


RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 4  # seconds; doubles each attempt


class OllamaError(Exception):
    pass


def chat(host: str, model: str, messages: list[dict], tools: list[dict],
         timeout: int = 900, reasoning_effort: str | None = None,
         api_key: str | None = None, chat_path: str = "/v1/chat/completions") -> dict:
    try:
        resp = requests.post(
            f"{host}{chat_path}",
            json={"model": model, "messages": messages, "tools": tools,
                  **({"reasoning_effort": reasoning_effort} if reasoning_effort else {})},
            timeout=timeout,
            headers=({"Authorization": f"Bearer {api_key}"} if api_key else None),
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "could not connect to Ollama — is it running? (`ollama serve`)"
        ) from exc

    # A hosted provider rate-limits per minute, and a burst of jobs hits
    # it routinely. These are transient by definition, so retrying with
    # backoff turns a failed job into a slow one. Jitter keeps several
    # workers from retrying in lockstep.
    for attempt in range(RATE_LIMIT_RETRIES):
        if resp.status_code != 429:
            break
        delay = RATE_LIMIT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
        time.sleep(delay)
        try:
            resp = requests.post(
                f"{host}{chat_path}",
                json={"model": model, "messages": messages, "tools": tools,
                      **({"reasoning_effort": reasoning_effort} if reasoning_effort else {})},
                timeout=timeout,
                headers=({"Authorization": f"Bearer {api_key}"} if api_key else None),
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaError(
                "could not connect to Ollama — is it running? (`ollama serve`)"
            ) from exc

    if resp.status_code == 429:
        raise OllamaError(
            f"rate limited by the model provider after {RATE_LIMIT_RETRIES} retries — "
            "the request budget is exhausted, try again later"
        )

    if resp.status_code == 404:
        raise OllamaError(f"model '{model}' not found — pull it with `ollama pull {model}`")
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def model_is_pulled(host: str, model: str) -> bool:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "could not connect to Ollama — is it running? (`ollama serve`)"
        ) from exc
    resp.raise_for_status()
    names = {m["name"] for m in resp.json().get("models", [])}
    return model in names


def pull_model(host: str, model: str) -> None:
    try:
        resp = requests.post(f"{host}/api/pull", json={"model": model, "stream": False}, timeout=None)
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "could not connect to Ollama — is it running? (`ollama serve`)"
        ) from exc
    resp.raise_for_status()
