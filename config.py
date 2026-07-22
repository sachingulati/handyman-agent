import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = PROJECT_ROOT / "jobs.db"
JOBS_LOG_DIR = PROJECT_ROOT / "jobs"
JOBS_LOG_DIR.mkdir(exist_ok=True)

MAX_CONCURRENT_JOBS = int(os.environ.get("GEMMA_MAX_CONCURRENT_JOBS", "3"))
OLLAMA_HOST = os.environ.get("GEMMA_OLLAMA_HOST", "http://localhost:11434")
# gemma-12b-gpu:latest is a local-only tag built from a Modelfile
# (huihui_ai/gemma-4-abliterated:12b + num_gpu 999 + num_ctx 32768 - see
# D:\ai-tools\ollama-models\gemma-12b-gpu-full.Modelfile). It is NOT
# pullable from any registry, so worker.py's auto-pull-on-first-use
# safety net does not apply to it: if this tag is missing (fresh
# machine, deleted tag), `ollama create gemma-12b-gpu -f Modelfile` must
# be run manually before jobs will work. Chosen over the registry
# huihui_ai/gemma-4-abliterated:26b default because 26b's 17GB weights
# always exceed this machine's 8GB VRAM and partially CPU-offload,
# which was too slow for practical use.
#
# 32768 (not the model's native 131072 max) is deliberate: measured
# live for the same ~434-token prompt, the larger the configured
# num_ctx, the slower prompt eval and generation get - a real compute
# cost tied to the allocated ctx size itself, not a VRAM one (even
# 131072 barely used more VRAM than 32768, thanks to Gemma 4's sliding-
# window attention keeping the real KV cache small regardless of
# nominal context length):
#   32768:  937.94 tok/s prompt eval, 28.81 tok/s generation
#   65536:  130.09 tok/s prompt eval, 24.73 tok/s generation
#   131072: 117.22 tok/s prompt eval, 11.13 tok/s generation
# There's a sharp cliff on prompt-eval speed between 32768 and 65536,
# but generation degrades more gradually - see MODEL_NAME_MID/_BIG
# below, which escalate a job to progressively larger contexts only if
# its own conversation actually grows that large. 32768 is already
# ~75x gemma-agent's typical per-task prompt size, so most jobs never
# pay any of this cost.
MODEL_NAME = os.environ.get(
    "GEMMA_MODEL_NAME",
    "gemma-12b-gpu:latest",
)

# Same build as MODEL_NAME but with num_ctx 65536 - see
# D:\ai-tools\ollama-models\gemma-12b-gpu-mid.Modelfile. worker.py
# switches a job to this model, for its remaining iterations only, if
# the conversation's estimated token count crosses
# CONTEXT_GROWTH_THRESHOLD_MID_TOKENS. Same local-only/no-auto-pull
# caveat as MODEL_NAME applies.
MODEL_NAME_MID = os.environ.get(
    "GEMMA_MODEL_NAME_MID",
    "gemma-12b-gpu-mid:latest",
)
# ~73% of MODEL_NAME's 32768 context, leaving headroom to still switch
# and complete the response before actually hitting that ceiling.
CONTEXT_GROWTH_THRESHOLD_MID_TOKENS = int(
    os.environ.get("GEMMA_CONTEXT_GROWTH_THRESHOLD_MID_TOKENS", "24000")
)

# Same build as MODEL_NAME but with num_ctx 131072 (the model's native
# max) - see D:\ai-tools\ollama-models\gemma-12b-gpu-big.Modelfile.
# worker.py escalates a job to this model if its conversation grows
# past MODEL_NAME_MID's own budget. Same local-only/no-auto-pull
# caveat as MODEL_NAME applies.
MODEL_NAME_BIG = os.environ.get(
    "GEMMA_MODEL_NAME_BIG",
    "gemma-12b-gpu-big:latest",
)
# ~73% of MODEL_NAME_MID's 65536 context.
CONTEXT_GROWTH_THRESHOLD_BIG_TOKENS = int(
    os.environ.get("GEMMA_CONTEXT_GROWTH_THRESHOLD_BIG_TOKENS", "48000")
)

MAX_ITERATIONS = int(os.environ.get("GEMMA_MAX_ITERATIONS", "40"))
MAX_WALL_CLOCK_SECONDS = int(os.environ.get("GEMMA_MAX_WALL_CLOCK_SECONDS", str(20 * 60)))
MAX_TOTAL_TOKENS = int(os.environ.get("GEMMA_MAX_TOTAL_TOKENS", "200000"))
WATCHDOG_MAX_RETRIES = int(os.environ.get("GEMMA_WATCHDOG_MAX_RETRIES", "3"))

# Optional: when set, tools.web_search uses Tavily (more reliable,
# free tier 1000 credits/month) instead of the free DuckDuckGo scraper.
# GEMMA_TAVILY_API_KEY (gemma-agent-specific) takes priority so a
# distribution/user can point gemma-agent at its own key or quota; if
# unset, falls back to a plain TAVILY_API_KEY already in the environment
# (e.g. one configured for another tool) as a convenience; if neither is
# set, TAVILY_API_KEY stays None and web_search falls back to DuckDuckGo.
TAVILY_API_KEY = os.environ.get("GEMMA_TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY") or None
