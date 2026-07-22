"""Choosing a model for this machine, and proving it actually works.

Two things make this more than a config generator.

First, hardware detection is unreliable enough that it must degrade into
asking rather than guessing wrong.

Second, and more important: a model's advertised capabilities cannot be
trusted. One model tested here listed "tools" in its metadata and still
returned tool calls as raw JSON text with an empty tool_calls field, and
another wrote prose describing how a person would use the tool, complete
with invented command-line syntax. Both would have been written into a
config as working. So every candidate is put through a real request with
a real tool schema, and is only accepted when tool_calls comes back
actually populated.
"""

import json
import re
import subprocess

import requests

# Sizes are measured from the registry rather than inferred from a tag.
# "Smaller variant of the same family" does not reliably mean smaller: a
# fixed vision-projector overhead means one 5B-parameter build is larger
# on disk than a 12B one from the same family.
CANDIDATES = [
    # (vram_gb_needed, model, approx_gb, note)
    (0.0, "qwen3:1.7b", 1.4, "CPU-only fallback; expect it to be slow"),
    (3.5, "qwen3:4b", 2.5, "smallest GPU-resident option"),
    (6.5, "qwen3:8b", 5.2, "verified tool-calling"),
    (9.0, "gemma4:12b", 7.6, "verified tool-calling, well proven"),
    (11.0, "qwen3:14b", 9.3, "verified tool-calling"),
    (20.0, "gemma4:26b", 18.0, "large; mixture-of-experts, so faster than its size suggests"),
]

PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a text file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}]
PROBE_PROMPT = "Write the word hello into a file called probe.txt. Use the tool."


def detect_vram_gb() -> float | None:
    """Total VRAM in GB, or None when it cannot be determined.

    Returning None matters: a wrong number silently picks a model that
    will not fit, so an unknown answer has to become a question instead.
    """
    probes = [
        (["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
         lambda out: float(out.strip().splitlines()[0]) / 1024),
        (["rocm-smi", "--showmeminfo", "vram", "--csv"],
         lambda out: float(re.search(r"(\d{4,})", out).group(1)) / (1024 ** 3)),
        (["system_profiler", "SPHardwareDataType"],
         lambda out: float(re.search(r"Memory:\s*(\d+)\s*GB", out).group(1))),
    ]
    for argv, parse in probes:
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            if result.returncode == 0 and result.stdout.strip():
                return round(parse(result.stdout), 1)
        except Exception:
            continue
    return None


def recommend(vram_gb: float | None) -> list[tuple[str, float, str]]:
    """Candidates that fit, largest first. Unknown VRAM offers everything."""
    if vram_gb is None:
        usable = CANDIDATES
    else:
        # Leave headroom for the KV cache; weights alone are not the whole cost.
        usable = [c for c in CANDIDATES if c[0] <= vram_gb]
    return [(model, size, note) for _, model, size, note in reversed(usable)]


def verify_tool_calling(host: str, model: str, timeout: int = 300,
                        api_key: str | None = None,
                        chat_path: str = "/v1/chat/completions") -> tuple[bool, str]:
    """Send one real request and require a populated tool_calls field.

    Checking that the field merely exists is not enough - the failure this
    catches returns a 200 with tool_calls set to null and the call written
    into the message text instead.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    body = {"model": model, "tools": PROBE_TOOL, "stream": False,
            "messages": [{"role": "user", "content": PROBE_PROMPT}]}
    try:
        resp = requests.post(f"{host}{chat_path}", json=body,
                             headers=headers, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return False, "could not reach the model server"
    if resp.status_code == 404:
        return False, "model not found on the server"
    if resp.status_code >= 400:
        detail = resp.text[:120].replace("\n", " ")
        return False, f"server rejected the request ({resp.status_code}): {detail}"

    try:
        message = resp.json()["choices"][0]["message"]
    except Exception:
        return False, "unexpected response shape"

    calls = message.get("tool_calls")
    if not calls:
        content = (message.get("content") or "").strip()
        if "write_file" in content:
            return False, ("returned the tool call as plain text instead of a "
                           "tool call - unusable for this")
        return False, "did not call the tool"
    try:
        name = calls[0]["function"]["name"]
        json.loads(calls[0]["function"]["arguments"])
    except Exception:
        return False, "tool call was malformed"
    if name != "write_file":
        return False, f"called '{name}' instead of the requested tool"
    return True, "ok"


def build_config(model: str, host: str = "http://localhost:11434") -> dict:
    """A single-tier config. Extra tiers are an optimisation for machines
    that cannot hold a large context, and are not worth guessing at."""
    return {
        "ollama_host": host,
        "max_concurrent_jobs": 1,
        "tiers": [{"name": "small", "model": model, "threshold_tokens": 0}],
    }
