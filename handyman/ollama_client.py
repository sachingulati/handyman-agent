import requests


class OllamaError(Exception):
    pass


def chat(host: str, model: str, messages: list[dict], tools: list[dict], timeout: int = 120) -> dict:
    try:
        resp = requests.post(
            f"{host}/v1/chat/completions",
            json={"model": model, "messages": messages, "tools": tools},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            "could not connect to Ollama — is it running? (`ollama serve`)"
        ) from exc

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
