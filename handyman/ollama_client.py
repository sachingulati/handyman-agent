import random
import re
import time

import requests


RATE_LIMIT_RETRIES = 4
RATE_LIMIT_BASE_DELAY = 4  # seconds; doubles each attempt
MAX_SUGGESTED_DELAY = 60  # cap on a provider-supplied retry hint


def _rate_limit_detail(resp) -> tuple[bool, float | None]:
    """Inspect a 429 body: is it retryable, and how long should we wait?

    A provider can answer 429 for two very different reasons. A genuine
    per-minute throttle clears on its own. An input larger than the
    account's per-request token allowance never will - retrying the same
    oversized request just fails identically, slower. Only the body
    distinguishes them.
    """
    try:
        payload = resp.json()
    except Exception:
        return True, None
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    message = str((payload or {}).get("error", {}).get("message", ""))

    if "input_token_count" in message or "input token count" in message.lower():
        return False, None

    match = re.search(r"retry in ([0-9.]+)s", message)
    return True, (min(float(match.group(1)), MAX_SUGGESTED_DELAY) if match else None)


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
        retryable, suggested = _rate_limit_detail(resp)
        if not retryable:
            raise OllamaError(
                "the request is larger than this account's per-request token "
                "allowance, so retrying cannot help - shorten the task, or "
                "split it into smaller steps"
            )
        delay = suggested if suggested is not None else (
            RATE_LIMIT_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1))
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
