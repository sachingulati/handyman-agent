# gemma-agent OSS Generalization — Design

> **Status: brainstorming complete; ready for implementation planning.**
> All blocking design questions are resolved. The two items left in "Open
> questions" are a deferred naming choice and a pre-release verification
> task — neither blocks writing an implementation plan.
>
> **One caveat:** a small amount of code was implemented ahead of this
> design being finalized (the Tavily `web_search` integration, commits
> `eb98d60`/`767c194`), built against the current single-machine structure
> rather than the generalized one described here. See "Next steps".
>
> This document exists so the design survives even if the conversation that
> produced it doesn't. Update it as the discussion continues; do not treat
> silence on an "open question" as a decision.

## Goal

Take the working, single-machine `gemma-agent` (see
`docs/superpowers/specs/2026-07-21-gemma-subagent-design.md` and
`.superpowers/sdd/progress.md` for its build history) and generalize it for
open-source distribution to a wide audience — not tied to this specific
machine's OS, GPU, or model choice.

## Decisions made so far

| Question | Decision |
|---|---|
| Scope of this pass | Full roadmap first, then implement in order |
| Model default | **Official Gemma** as default; the current uncensored/abliterated build becomes opt-in, not default |
| Platforms | Windows, Mac, and Linux |
| Hardware/VRAM config | Auto-detect (nvidia-smi / rocm-smi / Mac unified-memory query) with a guided question fallback when detection fails or is ambiguous |
| Distribution | PyPI package (`pip install` / `uvx`), with a CLI entry point for `claude mcp add` |
| Tier count | Configurable (1-3 tiers), not hardcoded — machines with ample VRAM (24GB+) can run a single tier with no escalation at all |
| Backend | **Ollama for local** — no pluggable LM Studio/raw-llama.cpp abstraction. **Revised this round:** one *hosted* provider is now in scope alongside it, restricted to OpenAI-compatible endpoints (see "Hosted provider" below). This does not reopen the local-backend question. |
| License | MIT |
| Safety gate | Docs-only (clear README disclosure of what the path jail does/doesn't cover) — no code-level consent gate |
| Model family scope | **Model-family-agnostic**, not Gemma-only — architecture already supports this (see Architecture note below) |
| Project name | **STILL OPEN — not confirmed.** `handyman` is the current *working placeholder* (replacing the earlier `oddjob` placeholder). A repo exists at `github.com/sachingulati/handyman-agent` and the surface mapping is decided *if the name is kept*, but the name itself is not final. See open question 1. |

## Architecture (draft)

Four new pieces, layered onto the existing working code:

1. **Setup wizard** (`handyman/setup.py`, new) — detects OS/GPU/VRAM,
   decides model + tier count, builds the necessary Ollama tags, writes a
   config file (`~/.config/handyman/config.yaml`). Falls back to asking
   the user when detection is ambiguous.
2. **Tier config file** — replaces today's hardcoded `MODEL_NAME` /
   `MODEL_NAME_MID` / `MODEL_NAME_BIG` env vars with a declarative list:
   model name, context size, escalation threshold, per tier (1-3 entries).
   `worker.py`'s escalation logic (see `.superpowers/sdd/progress.md`,
   commit `6e30989`) already takes an ordered list of
   `(threshold, tier_name, chat_fn)` — it just needs to build that list from
   config instead of three fixed constants. No change needed to the
   escalation/tier-aware-concurrency mechanism itself.
3. **Cross-platform process-utils module** — most process code already
   degrades gracefully (`db.is_pid_alive` already has both Windows/POSIX
   paths; `CREATE_NO_WINDOW` is already `getattr`-guarded to `0` on
   non-Windows). The one real gap: `tools.run_bash`'s Windows-only
   `taskkill /T /F` timeout-kill needs a POSIX branch (process group +
   `SIGKILL`), same pattern as `is_pid_alive`.

4. **Hosted provider (decided this round).**
   An optional second execution target, alongside local Ollama. See the
   dedicated section below for its scope and boundaries.

**New requirement, added this round:** the setup wizard must not trust a
candidate model's declared capabilities (`ollama show`'s "tools" label) or
its family's reputation. It must run a **live tool-call self-test** —
send one real request with a trivial tool schema, and check that
`tool_calls` is actually populated in the response (not just present in
metadata) — before finalizing any model into the config. See "Model
verification findings" below for exactly why this is load-bearing, not
theoretical caution.

## Hosted provider (decided this round — resolves the "revisit Ollama-only" question)

**Decision: yes, add a hosted fallback, restricted to OpenAI-compatible
endpoints.** This is a deliberate narrow slice of the broader proposal, not
the "pluggable backend" effort that was scoped out earlier.

**Why it earned its place.** The three use cases already listed below
("concrete use cases for a hosted fallback") are all real, but the
strongest argument is one they don't cover: **onboarding**. For a wide OSS
audience, a hosted path means someone with no capable GPU can install,
paste a free Google AI Studio key, and have a working tool immediately,
instead of pulling a 7.6GB model first and discovering their hardware
can't run it. That converts the project's hardest adoption barrier into a
60-second path.

**In scope**

- One code path: OpenAI-compatible `POST /chat/completions` — the same
  request shape `ollama_client.chat()` already sends. A provider is
  nothing but `(base_url, api_key, model_name)`; no provider-specific
  branching.
- Verified-working targets: **Google AI Studio direct**
  (`generativelanguage.googleapis.com/v1beta/openai/`, 3/3 live tool-call
  successes on both `gemma-4-26b-a4b-it` and `gemma-4-31b-it`) and
  **OpenRouter with the user's own linked key** (BYOK).
- Key resolution mirrors the Tavily pattern already implemented: a
  project-specific env var first, then a conventional shared one
  (e.g. `GOOGLE_API_KEY`/`GEMINI_API_KEY`), then unset → hosted disabled.
- The setup wizard's **live tool-call self-test applies here too.** A
  hosted model is not written into config until a real request with a real
  tool schema comes back with a populated `tool_calls`. Same standard as
  local models, same reason.

**Explicitly out of scope for v1**

- **Google's native `generateContent` search grounding.** It works and it
  is genuinely better than scraping (verified live: real citations, fresh
  results), but it is a *second, non-OpenAI request/response shape* —
  meaningfully more work than the doc previously assumed, Google-only, and
  now largely redundant since Tavily landed on the local path. Revisit
  after v1. This is the one piece of the original proposal being cut.
- **OpenRouter's unauthenticated/pooled free tier.** Proven unreliable
  live this session (rotation to paid mid-session; 6/6 rate-limited).
  BYOK only.
- **Tier escalation on the hosted path.** Measured: no meaningful
  context-size speed penalty hosted (2.61s at ~3 tokens vs 2.25s at
  ~13.2K). The whole grow-with-need tier mechanism exists to work around
  consumer-GPU constraints that don't apply here. Hosted is a single
  entry, always full context.

**Non-obvious implication — hosted jobs must bypass the local concurrency
machinery.** The `current_tier`/`escalating`/`is_sole_runner` logic added
in commit `6e30989` exists solely because one GPU holds one model at a
time. A hosted job occupies no VRAM, so it must **not** count against the
local concurrency cap and must **not** constrain which tier local jobs may
claim. If hosted jobs are allowed to participate in that bookkeeping, they
will block local jobs for no reason.

**Positioning: opportunistic, never depended on.** Local stays the default
for routine, high-volume delegation. The free hosted tier has a
documented-unreliable reputation (forum reports of up to 70% error rates,
500s counting against quota, undocumented rate limits) and a hard
**16,000-token input cap** discovered empirically, not from docs. Two
consequences for error handling: a hosted failure must never strand a job
at a non-terminal status (the failure mode Tasks 7-9 spent 7 review rounds
hardening against), and a conversation that outgrows the 16K input cap
must fail with a clear, actionable message or hand back to local — never
silently retry.

**Practical note carried forward:** don't regex-validate the API key's
shape. A key sourced from a Gemini CLI auth session had an `AQ.Ab8...`
prefix rather than the classic `AIzaSy...` and worked identically. Test by
calling the endpoint.

## Data flow (agreed this round)

```
handyman setup  (new, one-time)
   detect OS/GPU/VRAM  ->  pick candidates from the curated list
   ->  ensure pulled  ->  LIVE TOOL-CALL SELF-TEST  ->  next candidate on failure
   ->  write ~/.config/handyman/config.yaml   { tiers: [1-3], hosted: {...}? }

gemma_delegate(task, working_dir, allow_hosted=False)      [server.py]
   reap dead running jobs
   ->  DECIDE PROVIDER (local | hosted)                    <- new
   ->  db.create_job(..., provider)                        <- new column
   ->  try_claim_with_cap   (hosted bypasses the cap)
   ->  spawn_worker

worker.main(job_id)
   read job.provider
   ->  build base chat_fn + escalation_tiers FROM CONFIG   <- replaces the 3 constants
        local  -> N tiers from config (N = 1-3)
        hosted -> single entry, escalation_tiers=[]
   ->  run_job(...)                                        <- UNCHANGED

run_job(...)                                               <- UNCHANGED
```

**Key structural finding: `run_job` needs no changes at all.** Its
`escalation_tiers` parameter — a list of `(threshold, tier_name, chat_fn)`
triples — is already the right seam. A provider is just a different
`chat_fn` closure; `run_job` never learns that providers exist. All the
new wiring lands in `worker.main()` and `server.py`.

**Architecture consequence — the provider decision must be persisted, not
computed in `main()`.** The concurrency cap is enforced by
`db.try_claim_with_cap` / `db.claim_next_queued_job`, which run from
`server.py` and from `worker.py`'s `finally` hand-off — both *before*
`main()` exists to decide anything. So "hosted jobs bypass the local cap"
cannot be implemented inside `main()`. The provider is chosen at delegate
time and stored on the job row (a `provider` column, alongside the
`current_tier`/`escalating` columns added in `6e30989`), then read by both
the claim logic and `main()`. Same idempotent `ALTER TABLE` migration
pattern already used in `db.connect()`.

**Provider decision rule.** `allow_hosted` governs *both* contention and
availability — hosted is never entered silently:

| Situation | Result |
|---|---|
| No hosted configured | Local. Queue if at cap. |
| Hosted configured, local slot free | Local. |
| Hosted configured, at concurrency cap, `allow_hosted=False` | **Queue** (today's behavior, unchanged). |
| Hosted configured, at concurrency cap, `allow_hosted=True` | Hosted. |
| Hosted configured, local Ollama unreachable, `allow_hosted=False` | **Terminal error** naming both fixes: start Ollama, or re-delegate with `allow_hosted=True`. |
| Hosted configured, local Ollama unreachable, `allow_hosted=True` | Hosted. |
| Hosted-only install (no local tiers in config) | Hosted. |

The `allow_hosted=False` + Ollama-down row is the debatable one, and it is
deliberate. Sending a task to a hosted provider ships the task text and
whatever file contents the model reads to a third party. A user who
configured a hosted key during onboarding and later installed Ollama
should not have an Ollama restart silently redirect their code to Google.
Extending the caller's explicit opt-in to cover this case keeps one rule
instead of two, at the cost of some of use case 3's convenience. Reversible
if it proves annoying in practice.

**Two smaller consequences of moving to a config file:**

- `config.py` currently reads env vars at import time and calls
  `JOBS_LOG_DIR.mkdir()` as an import side effect. A config file turns
  that into a `load()` function; the import-time `mkdir` should move with
  it. `tools.py` stays config-free (already true — keys and paths are
  passed as explicit arguments), so it is unaffected.
- **Delete `MAX_TOTAL_TOKENS` rather than porting it.** It is dead config:
  the token cap was dropped from `run_job`'s signature back in Task 7, so
  the setting has never done anything (already flagged as a plan defect in
  `.superpowers/sdd/progress.md`). Carrying it into the new schema would
  make it look load-bearing.

## Model selection policy (decided this round — resolves the DeepSeek and family-preference questions)

**Decision: the wizard carries a curated candidate list per VRAM tier,
ordered by *actually measured* download size, with the live tool-call
self-test as the gate.** There is no hardcoded family preference. Where
Qwen3 wins a tier below, that is an outcome of the measurements, not a
family judgment — consistent with the "never trust family reputation"
rule this document already established.

**DeepSeek: deprioritized, but not blocked.** It stays out of the
recommended list and gets no further proactive verification effort (2/2
failures, in two different ways, plus its free hosted tier disappeared in
July 2026). It is deliberately *not* hardcoded as a denylist: the live
self-test is already the gate, so if a user points at a DeepSeek build
that passes, it passes. Both observed failure modes get documented in the
README's model-compatibility notes so users understand the omission.

### Measured registry sizes (queried live this round, no download required)

Sizes come from the Ollama registry manifest endpoint
(`registry.ollama.ai/v2/library/<name>/manifests/<tag>`, summing layer
sizes) rather than from pulling. **Method validated:** it returns exactly
5.2GB for `qwen3:8b` and 7.6GB for the 12b, matching the sizes measured
by actually pulling them earlier in this document.

| Official Gemma 4 | Size | | Qwen3 | Size |
|---|---|---|---|---|
| `gemma4:e2b` | 7.2GB | | `qwen3:0.6b` | 0.5GB |
| `gemma4:12b` | 7.6GB | | `qwen3:1.7b` | 1.4GB |
| `gemma4:e4b` | 9.6GB | | `qwen3:4b` | 2.5GB |
| `gemma4:26b` | 18.0GB | | `qwen3:8b` | 5.2GB |
| `gemma4:31b` | 19.9GB | | `qwen3:14b` | 9.3GB |
| | | | `qwen3:30b` | 18.6GB |
| | | | `qwen3:32b` | 20.2GB |

**Three findings that drive the tier table:**

1. **The official-vs-abliterated sizing gap is closed.** Official Gemma 4
   sizes are identical to the `huihui_ai` abliterated builds at every
   shared tag (e2b 7.2GB, e4b 9.6GB, 12b 7.6GB). Abliteration changes
   weights, not architecture or quantization. Every Gemma size figure
   elsewhere in this document therefore transfers to the official
   default-path model unchanged. (Tool-calling on official builds is a
   separate matter — still unverified, see caveat below.)
2. **Gemma cannot serve the low-VRAM tiers at all.** Its smallest build is
   7.2GB. The fixed vision/multimodal-projector overhead means there is no
   small Gemma, so the CPU-only and <6GB rows have no Gemma option — not a
   preference, an absence.
3. **The vision tax is confirmed on official builds too:** `gemma4:12b`
   (7.6GB) is *smaller* than `gemma4:e4b` (9.6GB) despite the larger
   parameter count.

### Curated candidate list (draft)

Sizing rule: the model must fit in VRAM with roughly 1-1.5GB of headroom
for the KV cache. Ordered by preference within each tier.

| VRAM | Primary | Alternate | Tiers | Verified? |
|---|---|---|---|---|
| No GPU / CPU-only | `qwen3:1.7b` (1.4GB) | `qwen3:0.6b` (0.5GB) | 1, small ctx | **No — must self-test** |
| <6GB | `qwen3:4b` (2.5GB) | — (no Gemma exists this small) | 1 | **No — must self-test** |
| 6-8GB | `qwen3:8b` (5.2GB) | — (Gemma's 7.2GB leaves no headroom) | 1-2 | **Yes**, 3/3 |
| 8-12GB (dev machine) | `gemma4:12b` (7.6GB) | `qwen3:8b` (5.2GB) | up to 3 | **Yes** — production-proven build history |
| 12-16GB | `qwen3:14b` (9.3GB) | `gemma4:12b` (7.6GB) | 1-2 | **Yes**, 3/3 |
| 16-24GB | `gemma4:12b` or `qwen3:14b` at full native context | — | 1 | **Yes** |
| 24GB+ | `gemma4:26b` (18.0GB) | `qwen3:30b` (18.6GB) | 1, no escalation | **No — must self-test** |

**Caveat, and the reason the self-test gate is not optional here:** only
`qwen3:8b`, `qwen3:14b`, and the 12b/e2b/e4b Gemma builds have actually
been tool-call verified. The small Qwen3 entries (0.6b/1.7b/4b) and the
large entries (26b/30b) are **size-measured but tool-call unverified**.
Small models are exactly where tool-calling reliability is most likely to
degrade, and this document already contains one model that advertised
`tools` and silently failed (`qwen2.5-coder:7b`). Treat every "No" row as
a hypothesis the wizard must confirm on the user's machine before writing
it into config, and fall back to the next candidate on failure.

**Superseded caveat (kept for context):** an earlier draft of this table
warned that "smaller variant of the same model family" does NOT reliably
mean "smaller download/VRAM footprint." That is now quantified rather than
merely suspected — see finding 3 above.

## Model verification findings (empirical, this session)

Every row is a **live-tested** result (pulled via Ollama, real
`/v1/chat/completions` request with a real tool schema, response inspected
for a populated `tool_calls` field) — not inferred from `ollama show`
metadata or family reputation alone, because that metadata proved
unreliable (see qwen2.5-coder row).

| Model | Size on disk | `ollama show` claims `tools`? | Actually works (verified live)? |
|---|---|---|---|
| `huihui_ai/gemma-4-abliterated:e2b` (5.1B params) | 7.2GB | yes | **yes** — correct `tool_calls` |
| `huihui_ai/gemma-4-abliterated:e4b` (8.0B params) | 9.6GB | yes | **yes** — correct `tool_calls` |
| `huihui_ai/gemma-4-abliterated:12b` (11.9B params, in production use) | 7.6GB | yes | **yes** — proven across the whole gemma-agent build/smoke-test history |
| `deepseek-coder:6.7b` | 3.8GB | **no** (Capabilities lists only `completion`) | **no** — Ollama hard-rejects the request: `"does not support tools"` |
| `qwen2.5-coder:7b` | 4.7GB | **yes** (Capabilities lists `tools`) | **no** — request succeeds, but response dumps the tool call as raw JSON *text* inside `content`, with `tool_calls: null` and `finish_reason: "stop"` (not `"tool_calls"`). Confirmed **consistent across 3 repeated attempts**, not a fluke. This is the concrete case motivating the "live self-test, don't trust metadata" requirement above. |
| `qwen3:8b` (8.2B params) | 5.2GB | yes | **yes** — correct `tool_calls`, confirmed consistent across 3 repeated attempts (`finish_reason: "tool_calls"` every time). Qwen3 fixed whatever broke tool-calling in qwen2.5-coder:7b. |
| `qwen3:14b` (14.8B params) | 9.3GB | yes | **yes** — correct `tool_calls`, 3/3 consistent. Bigger/more capable Qwen3 tier, same reliability as the 8b. |
| `deepseek-r1:14b` (14.8B params, actually a Qwen2-architecture distill under the DeepSeek-R1 name) | 9.0GB | yes (claimed) | **no** — 3/3 consistent failure, and a *worse* failure mode than qwen2.5-coder: instead of dumping structured JSON, it writes human-oriented prose explaining how a person would use the tool, including a hallucinated fake CLI syntax (`write_file --name=test.txt --content=test`). Matches DeepSeek-R1's reasoning-centric chat template not integrating with structured function calling. Removed after verification (`ollama rm`). |

**Key surprise, corrected an earlier assumption:** Gemma's "e" (effective
parameter) naming does NOT mean proportionally smaller download size — e4b
(9.6GB) is larger on disk than the 12b model (7.6GB) in current production
use, almost certainly because of a fixed vision/multimodal-projector
overhead that doesn't shrink with the base LLM size. Any low-VRAM tier
recommendation must be based on actual measured size, not the tag name.
By contrast, `qwen3:8b` (8.2B params, 5.2GB) scales proportionally — no
vision-tax bloat — making the Qwen family a stronger low/mid-VRAM
candidate than Gemma's own smaller variants, pending further testing.

**Cleanup note:** `deepseek-coder:6.7b`, `qwen2.5-coder:7b`, and
`deepseek-r1:14b` were all removed from Ollama after failing verification
(`ollama rm`) — don't re-recommend any without re-verifying, in case a
newer build fixes the tool-calling gap. `qwen3:8b` and `qwen3:14b` were
kept (both passed verification).

**Two verified-working model families so far, at multiple sizes:** Gemma
(via `huihui_ai`'s builds: e2b/e4b/12b) and **Qwen3 specifically** (8b and
14b both pass; qwen2.5-coder does not — the "3" generation is the
dividing line, not the Qwen family in general). DeepSeek has been tried
twice (deepseek-coder:6.7b, deepseek-r1:14b) and failed both times, in two
different ways — deprioritize DeepSeek as a family until/unless a build is
found that passes.

## Related: free hosted model access (research; the decision it fed is in "Hosted provider" above)

Verified via web search (July 2026 data — check currency before relying on
this section, these tiers rotate):

- **Qwen3** (including Coder variants) — free via **OpenRouter** (`:free`
  tier, OpenAI-compatible endpoint, no credit card, ~28+ free models,
  rotates over time).
- **Official Gemma** — free via **Google AI Studio** directly (matches the
  "official Gemma as default" decision — Google hosts its own model for
  free with rate limits).
- **DeepSeek** — was free on OpenRouter through 2025; **as of July 2026,
  every DeepSeek model on OpenRouter converted to paid-only.** Another
  reason DeepSeek is a weaker bet for this project right now, independent
  of the tool-calling failures above.
- **The abliterated/uncensored builds we actually use (`huihui_ai`'s
  Gemma) are not hosted anywhere for free** — no legitimate provider hosts
  community uncensored fine-tunes. This cleanly reinforces the existing
  design split: the **official-model path could be either local Ollama or
  a free hosted API**; the **uncensored opt-in path is local-only, always.**

**This question is now RESOLVED — see the "Hosted provider" section
above.** It originally read: OpenRouter/Google AI Studio are also
OpenAI-compatible endpoints, the same request shape
`ollama_client.chat()` already sends, so an optional hosted fallback might
be a small addition rather than the "pluggable backend" effort scoped out
earlier. That framing held up, and the decision went that way — with one
correction found while deciding: it is only a small addition for the
OpenAI-compatible path. Google's native search grounding is a separate
API shape and was cut from v1 on that basis.

**Live-tested this round, and it reinforced the caution above rather than
resolving it:** using a real OpenRouter key,
`qwen/qwen3-coder:free` returned `404 - model unavailable for free, use
qwen/qwen3-coder [paid] instead` — meaning **it rotated to paid-only in
the time between researching it and testing it minutes later**, a live
demonstration of the exact rotation risk the research warned about.
Querying OpenRouter's `/models` endpoint live, the *only* remaining free
models in the Qwen/DeepSeek/Gemma set were `google/gemma-4-26b-a4b-it:free`
and `google/gemma-4-31b-it:free` — no Qwen or DeepSeek free tier survives
at all right now. Testing the free Gemma option then hit a `429` rate
limit on the very first attempt (3/3 attempts), with the error metadata
revealing OpenRouter's free Gemma routing proxies through **Google AI
Studio**'s own free pool under the hood — apparently oversubscribed enough
to throttle a cold first request. Retried both free Gemma sizes
(`26b-a4b-it` and `31b-it`), 3 attempts each — **6/6 hit the same 429**,
so this isn't one congested model, it's OpenRouter's whole shared/pooled
Google AI Studio free quota. The error message itself hints at the fix:
`"add your own key to accumulate your rate limits"` — i.e. OpenRouter's
*unauthenticated* pooled free quota for Google-backed models is
persistently oversubscribed, but linking your own Google AI Studio key
(BYOK) would use your own quota instead. Net effect: **a hosted fallback
should be designed as an optional convenience, never something gemma-agent
depends on working** — tool-calling reliability on any specific free
hosted model/tier remains unverified as of this session, and the tier
itself may not even still exist by the time it's checked again.

**Done, and it resolved cleanly:** tested `gemma-4-26b-a4b-it` **and**
`gemma-4-31b-it` directly against
`generativelanguage.googleapis.com/v1beta/openai/chat/completions` using a
real Google AI Studio key (own quota, not OpenRouter's pool). **3/3
attempts succeeded for both models** — genuine `tool_calls` populated,
correct arguments, `finish_reason: "tool_calls"` every time, no billing
setup required for either. (One harmless Google-specific extra field
appears on each tool call, `extra_content.google.thought_signature` —
irrelevant to gemma-agent's own parsing, which only reads
`function.name`/`function.arguments`/`id`.)

**Conclusion: a genuine, working, free hosted fallback for the official
Gemma path exists** — direct Google AI Studio access, not routed through
OpenRouter. This resolves the "is any hosted option actually reliable"
question for Gemma specifically: yes, when accessed directly with the
user's own key. OpenRouter's *pooled/unauthenticated* free routing remains
unreliable (rotation + rate limits, both observed live this session);
direct-provider access with your own key is the pattern that actually
works. This should inform the "revisit Ollama-only" open question below —
if a hosted fallback is added, it should call providers directly
(Google AI Studio, and presumably OpenRouter *with the user's own linked
key* for BYOK routing) rather than relying on any provider's shared free
pool.

**Practical note from getting this working:** Google AI Studio API keys
don't only come in the classic `AIzaSy...` format — a key sourced from an
existing Gemini CLI auth session had a different `AQ.Ab8...`-style prefix
and worked identically. Don't validate key format too strictly if this is
automated later; test by calling the endpoint, not by regex-checking the
key's shape.

**Max-context speed test (this round):** queried `models/gemma-4-26b-a4b-it`
directly - the model's *technical* max is `inputTokenLimit: 262144`
(double the local abliterated build's 131072), `outputTokenLimit: 32768`.
But the **free tier hard-caps input at 16,000 tokens per request**,
independent of the model's real capability - confirmed by a genuine
`RESOURCE_EXHAUSTED` error (`generate_content_free_tier_input_token_count,
limit: 16000`) when a ~205K-token request was sent; it was rejected in
0.85s, never reaching the model at all. Rebuilt the test within the free
cap instead:

| Prompt size | Response time |
|---|---|
| ~3 tokens (baseline) | 2.61s |
| ~13,163 tokens (near free-tier max) | **2.25s** |

**No meaningful context-size speed penalty** on the hosted path — the
larger request was, if anything, marginally faster (noise-level
difference), a sharp contrast to the local finding earlier in this doc
(32768→131072 context cost ~8x on prompt eval for the *same trivial
prompt size* locally). Almost certainly explained by Google's serving
infrastructure not sharing the consumer-GPU VRAM/compute constraints this
whole local-tiering design exists to work around. Practical implication:
if a hosted fallback is added, it doesn't need the local build's
grow-with-need tier escalation at all - it can just always request
whatever it needs, up to the free tier's 16K-token ceiling (or higher on
a paid plan).

## Related: web search/research tooling

No model (Gemma, Qwen, DeepSeek, or otherwise) has inherent search/browsing
capability — that's always tool-calling + external tooling, never a model
capability. gemma-agent already has this via `tools.py`'s `web_search`/
`web_fetch` (free, DuckDuckGo HTML scraping, some known fragility already
fixed once during the original build — see Task 5 in
`.superpowers/sdd/progress.md`). Since tool-calling is model-agnostic, any
verified-working model (Gemma, Qwen3) can use these same tools without
change.

**Implemented this round (commit `eb98d60`), not just designed:**
`tools.web_search()` now uses Tavily's API when a key is configured,
falling back to the free DuckDuckGo scraper otherwise. Key resolution is
layered (`GEMMA_TAVILY_API_KEY` first, then a plain `TAVILY_API_KEY`
already in the environment, then `None`/DuckDuckGo) - revised from an
initial stricter "always isolated" design after feedback: a
gemma-agent-specific key still overrides (lets a distribution/user point
gemma-agent at its own key or quota), but it's convenient to fall back to
a shared key already configured for something else (e.g. this repo's own
dev session already has a Tavily key configured for Claude Code's
research use) rather than requiring a second, separate one. `tools.py`
stays config-free as a module (the key is passed as an explicit function
argument, same as every other tools.py function); `worker.py`'s
`execute_tool_call` is the only place that reads `config.TAVILY_API_KEY`
and passes it through. Verified against Tavily's own API docs before
implementing (`POST api.tavily.com/search`, Bearer auth, `{"query": ...}`
body). Tavily's free tier is 1,000 credits/month (confirmed via their own
pricing page). 11 new/updated tests, 127/127 suite passing.
**Not yet live-tested against the real Tavily API** (only mocked unit
tests so far) - do that before considering this fully verified, same
standard applied to every other integration in this doc.

**New this round: hosted Gemma (via Google AI Studio) has a genuinely
better built-in option than either of the above.** Live-tested Google's
native search-grounding tool (`{"google_search": {}}`, via the native
`generateContent` endpoint) against `gemma-4-26b-a4b-it` with a real
current-events question. It worked correctly — fresh, accurate,
properly-cited answer (mentioned events happening the same day, well
beyond any training cutoff), with a full `groundingMetadata` block: the
actual search query used, real cited source URLs, and inline
segment-to-citation mapping. This is Google's own integrated search
infrastructure, not a scraper, confirmed working for **Gemma specifically**
(not just Gemini) and free. If the hosted-fallback path is added, it
should use this native tool instead of routing through the local
DuckDuckGo-scraper `web_search`/`web_fetch` tools.

**Reliability caveat on the hosted-Gemma path overall (found via forum
search, not just docs):** a Google AI Developers Forum thread specifically
about `gemma-4-26b-a4b-it` reports serious free-tier reliability problems
under real usage — up to a 70% error rate, with 500 errors themselves
counting against the usage quota ("I simply cannot use gemma"). A separate
thread confirms Gemma's rate limits aren't officially documented anywhere
Google publishes (unlike Gemini's clearly-tiered RPM/RPD/TPM tables) - the
16,000-input-token free-tier cap found earlier in this doc was discovered
empirically, not from documentation. Our own handful of test requests all
succeeded, but that's a small sample against a service with documented,
recent instability. **Conclusion holds and is reinforced: hosted Gemma is
a fine occasional/opportunistic option, not something to depend on for
routine token-saving delegation.**

## Related: concrete use cases for a hosted fallback (now decided — see reconciliation at the end of this section)

These were the three triggers proposed for a hosted fallback, kept here as
the reasoning that led to the decision. **All three were revised when the
decision was actually made — see the reconciliation below before treating
any of them as the spec.**

1. **Search-dependent tasks.** Local's only research option is the fragile
   DuckDuckGo scraper (or Tavily now, see above). Hosted Gemma's native
   search grounding is qualitatively better (verified above) - tasks that
   genuinely need current information should route here.
2. **Overflow capacity when the local GPU is busy.** Local gemma-agent can
   only run one model tier at a time (VRAM-bound) and competes with any
   concurrent image/video generation (Wan2GP/ComfyUI) for the same GPU.
   Hosted Gemma runs on Google's infrastructure - zero contention.
3. **Fallback when Ollama isn't running.** Per the standing preference
   Claude no longer auto-starts `ollama serve` invisibly (see the
   `feedback-dont-background-ollama-serve` memory - it now uses a visible,
   titled Windows Terminal tab instead, started automatically), there are
   real gaps where local gemma-agent can't function until the user's
   terminal tab is up. A hosted fallback could cover delegation during
   those gaps.

Explicitly **not** the intended use: routine, high-frequency task
delegation (the core "save Claude tokens" use case) - that's exactly where
the documented 70% error-rate risk and undocumented rate limits bite
hardest. Local stays the default for volume.

**Reconciliation — what actually got decided.** Each of the three changed,
and a fourth trigger turned out to matter more than any of them:

| Proposed trigger | As decided |
|---|---|
| 1. Search-dependent tasks | **Cut from v1.** Needs Google's native `generateContent` API, a second non-OpenAI request shape — more work than assumed, and largely redundant now that Tavily covers search on the local path. |
| 2. Overflow when the GPU is busy | **Opt-in only.** Jobs queue as they do today unless the caller passes `allow_hosted=True`. Not automatic. |
| 3. Fallback when Ollama isn't running | **Opt-in only**, same flag. Without it, an Ollama-down job fails with a message naming both fixes, rather than silently shipping the task to a third party. |
| *(not originally listed)* Onboarding | **The strongest reason to build it.** Users with no capable GPU get a working tool without a multi-GB download. |

The "local stays the default for volume" position above is unchanged and
is now enforced structurally: hosted is never entered without an explicit
opt-in or a hosted-only config.

## Robustness findings from live use (2026-07-22, during Plan A execution)

Found by actually delegating work to the local model rather than by
reading the code. All three are pre-existing defects in the
single-machine build, and all three get worse with a wide audience.

**1. `pid=0` makes a job permanently unreapable — wedges the queue.**
`server.py`'s `gemma_delegate` claims a job with a placeholder
`pid=0` (`try_claim_with_cap(conn, job_id, pid=0, ...)`); the real pid is
only written later by `worker.main()` via `db.set_pid`. But
`db.reap_dead_running_jobs` guards each row with
`if pid and not is_pid_alive(pid)` — and `0` is falsy, so such a row is
**skipped by the reaper forever**, while `count_running` still counts it
against the concurrency cap.

Any worker that dies in the window between claim and `set_pid` therefore
strands its job at `status='running'` with no way back. Observed live:
one job sat at `running`/`pid=0` with no worker process in existence,
and had to be cleared with a manual SQL `UPDATE`. Accumulate
`MAX_CONCURRENT_JOBS` of these and every future delegation returns
`queued` forever.

**Sharpened by a direct comparison:** the same job id, run in the
foreground as `python worker.py <job_id>`, completed correctly and wrote
its output file. The detached spawn path produced no worker process, no
log bytes, and no error anywhere — only a row stuck at `running`/`pid=0`.
So the model, the tool dispatch, and the job logic are all fine; the
defect is isolated to `spawn_worker`'s detached launch plus the
unreapable placeholder pid. Two of three failures today came from this
path, which makes it the single least reliable part of the system and a
priority for the OSS pass.

This is the exact failure mode the reaper was added to prevent (see the
final-review notes in `.superpowers/sdd/progress.md`), with a hole left
for the reaper's own placeholder value. **Fix direction:** make the
claim write the real spawning pid, or use `NULL` rather than `0` and
reap rows that are `running` with a NULL pid older than some threshold.
Do not simply change the guard to `if pid is not None` — `0` is a
legitimate "not yet known" marker and a job claimed microseconds ago must
not be reaped before its worker sets its pid.

**2. `ollama_client.chat`'s `timeout=120` does not account for cold model
load.** Ollama evicts an idle model after 5 minutes by default. The next
request pays a full reload — measured at 16.5s here for a 7.6GB model on
a secondary drive, and it was slow enough on the first observed failure
to exceed 120s outright, surfacing as a raw `TimeoutError` socket
traceback in `result_summary`. A first-run user on slower storage, or
pulling the model for the first time, hits this reliably. **Fix
direction:** separate the load budget from the inference budget, or raise
the timeout substantially and document why.

**3. The job log stays empty through the entire failure.** Confirmed
again here: `jobs/<id>.log` was 0 bytes while a job ran, failed, and
stranded. Diagnosis required querying the SQLite row directly. This is
the observability gap already recorded as Task 11 finding #1 (no logging
around the pull/load step) — previously judged cosmetic. It is not: it
was the difference between a legible failure and an opaque one.

**Local-path speed, measured this session** (`gemma-12b-gpu`, 100% GPU,
ctx 32768, 8GB VRAM): cold load 16.5s; a trivial warm request
("say ok") still took 10.0s. A multi-iteration agentic job therefore
spends minutes, and any single generation-heavy call is uncomfortably
close to the 120s ceiling. This is context for how much the local path
actually saves versus a hosted call, and reinforces that tiering exists
for good reason.

## Job observability: continuous status (decided 2026-07-22)

**Problem, found by using the tool rather than reading it.** While a job
runs, `gemma_check` returns only `{job_id, status}`. A running job is a
black box. Diagnosing a single slow job this session required reading a
68KB prose log, querying SQLite by hand, and diffing files — the exact
"audit everything" cost the delegating model should never have to pay.

**Decision: build both a current-state heartbeat and an append-only event
log.** They answer different questions and are written from one code path
so they cannot drift.

| | A — heartbeat columns | B — `job_events` table |
|---|---|---|
| Answers | where the job is *now* | what actually *happened* |
| Shape | columns on the `jobs` row | append-only rows |
| Writes | 1 UPDATE per iteration | ~3-5 INSERTs per iteration |
| Read by | `gemma_check`, every poll | audits and post-mortems |

**Critical property: zero cost to the delegated model.** Status is written
by `worker.py` in the job loop — ordinary Python, a SQLite write of tens
of microseconds against model calls of 5-45 seconds. The model never sees
it, never generates it, and spends no tokens on it. A design where the
*model* reports its own progress was considered and rejected for exactly
this reason: it would add token cost and depend on model compliance.

**Implementation shape.** One `record(conn, job_id, event_type, detail)`
helper called from each interesting site in `run_job`. It appends the
event row and updates the heartbeat columns in the same transaction, so
"latest event" and "current state" can never disagree. `gemma_check`
returns the heartbeat fields while running, plus the last N events.

**Explicitly deferred: automatic termination.** Termination is already
bounded four ways — the `TASK_COMPLETE` sentinel, the watchdog's 3
nudges, `MAX_ITERATIONS`, and the wall-clock cap. The job that motivated
this (881.6s for work finished at ~190s) completed *correctly*; it was
slow, not runaway. Building a kill heuristic now would mean tuning it
blind. Once B's event data exists, where the turns actually go becomes
measurable, and termination can be designed against real numbers.

## Reasoning-model overhead (measured 2026-07-22)

The local model (`gemma-4`, 12B) advertises a `thinking` capability and
has it on by default, emitting a `reasoning` block before each turn.

**The control is binary in practice, not graduated.** Measured on
`/v1/chat/completions` with a real tool schema:

| `reasoning_effort` | Time | Reasoning emitted | Tool call correct |
|---|---|---|---|
| unset (default) | 8.5s | 200 chars | yes |
| `none` | **5.8s** | **0 chars** | yes |
| `low` | 8.7s | 229 chars | yes |
| `medium` | 9.2s | 299 chars | yes |
| `high` | 8.1s | 196 chars | yes |

`low`/`medium`/`high` are indistinguishable from the default; only `none`
disables it. (`think: false` and `chat_template_kwargs` are ignored on the
OpenAI-compatible endpoint; `think: false` *does* work on the native
`/api/chat`.) **Decision: do not disable reasoning** — it is a real
capability and the knob has no middle setting.

**The cost is driven by ambiguity, not by reasoning itself.** On a clean,
unambiguous task reasoning cost 2.7s. On one ambiguous instruction it cost
~14 minutes: a delegation prompt said "fix the imports in four files" and
then listed edits across three, and the job log shows the model
re-deriving that contradiction across turns ("*Wait, I counted 3...*").

**Practical rule, and the cheaper lever:** write delegation prompts that
are internally consistent and state exact counts. Prompt precision buys
more than disabling reasoning, and costs no capability. Worth surfacing in
the README's delegation guidance, since users will hit the same trap.

## `run_bash` is not shell-portable (found 2026-07-22, three live reproductions)

**Severity: high. This is the biggest cross-platform defect found so far,
and it is silent.**

`tools.run_bash` calls `subprocess.Popen(command, shell=True)`. On POSIX
that resolves to `/bin/sh`; on Windows it resolves to **cmd.exe**. These
are not interchangeable, and the differences hit ordinary commands:

| Construct | POSIX `/bin/sh` | Windows cmd.exe |
|---|---|---|
| `'single quotes'` | quotes the content | **does not quote at all** |
| `\|` inside quotes | literal character | **parsed as a pipe** |
| `a/b/c.exe` | runs | **not recognized** (needs backslashes) |
| `dir/*.py` | shell expands the glob | passed through literally |

**Three reproductions, all in one session, all from commands that are
correct POSIX shell:**

1. `sed -E 's/(config\|db\|server)/x/' tests/*.py`
   -> `'db' is not recognized as an internal or external command`, exit 255.
   cmd.exe read the regex alternation as a pipeline.
2. `.venv/Scripts/python.exe script.py`
   -> `'.venv' is not recognized`. Forward slashes are not accepted for
   the executable path.
3. The same command with backslashes succeeded in 27.8s.

**Why this is worse than a portability footnote.** The *model* writes
these commands. When one fails, the failure surfaces as a confusing
stderr string attributed to a job, so it reads as model incompetence
rather than a platform defect. In this session the controller was two
failures deep before checking `return_code`/`stderr` closely enough to
tell them apart - while actively auditing. An ordinary user would
conclude the local agent is broken and stop using it.

It also silently biases every delegation: the model naturally writes
POSIX-shell idioms (quoting, pipes, globs, `&&`), which is precisely the
set that breaks on the platform this project was developed on.

**Fix directions (decide during implementation, not ad hoc):**

- Drop `shell=True` and accept an argv list, losing shell features but
  gaining exact, portable behaviour; or
- Invoke a known shell explicitly (`bash -c` where available, documenting
  the dependency); or
- Keep `shell=True` but document hard which constructs are portable, and
  have the system prompt steer the model toward them.

**Portable pattern that worked, worth recommending regardless:** stage a
script file and invoke it with a bare interpreter call containing no
shell metacharacters. That is how the failing transformation was finally
completed.

**Related, same function:** `run_bash`'s timeout kill is also
Windows-only (`taskkill /T /F` with no POSIX branch) - see the
cross-platform process-utils item in the Architecture section. Both
defects live in `run_bash`; this one is the more damaging because it is
silent and affects every command, not only timed-out ones.

## Local-model delegation limits (measured 2026-07-22)

Measured by delegating real build work to the local model and auditing
every result. These shape what the tool should advertise itself as good
for, and they inform `docs/DELEGATING.md`.

**Reliable**

| Mode | Typical | Notes |
|---|---|---|
| Shell commands (`run_bash`) | 20-30s | Fastest and most reliable mode |
| Copying a file | 88s | Use a shell `cp`; see the anti-pattern below |
| Single-line edits | ~30-60s each | Correct, including across 8 in one job |
| Authoring a new file under ~60 lines | 80-95s | Against tests written first |

**Unreliable or unsafe**

- **Authoring more than ~60 lines in one call.** A ~130-line module
  produced *nothing* in 911s - no partial file, no error, just a timeout.
  The ceiling sits between 60 and 130 lines and the failure is silent.
- **Multi-line edits inside an indented block.** Keeps the first line's
  indentation and strips it from the rest, producing invalid Python
  without any error.
- **`ctypes` / native API calls.** Produced a memory-corrupting defect:
  `GetExitCodeProcess` called with one argument instead of two, so an
  arbitrary address was passed as the out-pointer. Do not delegate FFI.
- **Transcribing content it could copy.** Timed out twice reproducing a
  60-line file that a shell `cp` handled in 88s.

**Reasoning is binary, and can block all output.** The model ships with
thinking enabled. `reasoning_effort` `low`/`medium`/`high` are
indistinguishable from the default; only `none` disables it. On a large
or ambiguous task, reasoning consumed the entire budget and produced
**zero tool calls in 28 minutes**; the identical task with reasoning off
finished in 94s with working code.

**The cost driver is ambiguity, not reasoning.** One internally
inconsistent instruction - a prompt saying "four files" that then listed
three - cost ~14 minutes of visible re-derivation in the job log. A clean
task spends ~3s deliberating. Prompt precision is a far cheaper lever
than disabling a capability.

**Attribution caveat, recorded deliberately.** Over this session more
defects originated with the *controller* than with the model: shell
quoting that reached the model as literal backslashes, inconsistent
counts, fix instructions naming one of two identical bugs, and the same
heredoc escaping bug twice. A workflow that assumes the small model is
the unreliable component will misattribute most of its failures. The
job-status and event trail added this round exists partly to make that
distinction cheap.

## Live testing round (2026-07-22, after Plan A)

Ran the real CLI against the real config and a real local model, rather
than trusting the suite. Three findings, two of them things 161 green
tests could never have caught.

**1. Fresh-install blocker (FIXED).** `config` resolves `db_path` into a
platform data directory that nothing created, so the very first command a
new user runs failed with a bare `unable to open database file`. Invisible
to the test suite because every test points `db_path` at `tmp_path`, which
always exists. `db.connect` now creates the directory.

**2. Job stranding via the pid placeholder (FIXED).** Previously recorded
in this document as a defect; both halves are now closed.
`gemma_delegate` claims with `pid=0`, and the reaper guarded on
`if pid and not is_pid_alive(pid)` - zero is falsy, so the row was skipped
forever while still counting against the concurrency cap.

The naive fix is wrong: the pid is *legitimately* unknown for a moment
after the claim, so reaping any falsy pid would kill every job the instant
it started. The reaper now distinguishes:

| Row state | Action |
|---|---|
| pid set, process dead | reap |
| pid unset, claimed recently | leave alone |
| pid unset, older than `UNSET_PID_GRACE_SECONDS` | reap |

`spawn_worker` also returns the child pid now, and `gemma_delegate`
records it immediately, so the grace period is a backstop rather than the
primary defence. Verified live: a job records a real pid at spawn where
the old code left `0`.

**3. Job observability works.** A running job now answers with iteration,
last action and its event trail from one query:

```
status=done  iteration=2  last_action=done
recent=[1: chat, 1: tool_call write_file, 2: chat, 2: done]
```

Two live jobs completed end to end in ~15s each with correct output.

**Caveat on the spawn path.** The detached spawn succeeded here, having
failed repeatedly earlier in the same session from the old repo's
registered MCP server process. That suggests those failures were specific
to that long-lived server process rather than to the spawn code. Treat the
spawn as "working when launched fresh", not as proven under MCP.

## Testing plan (agreed this round)

Current state: 127 tests across 11 files, all mocked/unit, **plus** a
history of live smoke tests against real Ollama. **No CI exists.**

**1. CI matrix is the highest-value addition, and it is not optional
here.** The entire point of this pass is Mac and Linux support, but every
test to date has run on Windows only. Three specific things are *claims*
rather than verified facts until they run on the target OS: `run_bash`'s
new POSIX process-group `SIGKILL` branch, `db.is_pid_alive`'s POSIX
branch, and config-path resolution (`~/.config` vs `%APPDATA%`).
GitHub Actions matrix across `windows-latest`, `macos-latest`,
`ubuntu-latest`. Without this, "cross-platform" is untested marketing.

**2. Turn this document's verification findings into the self-test's test
corpus.** Every model failure observed this session is a recorded, real
response shape — they should become fixtures rather than staying prose:

| Fixture from | Shape the self-test must reject |
|---|---|
| `deepseek-coder:6.7b` | Ollama hard-rejects: `"does not support tools"` |
| `qwen2.5-coder:7b` | 200 OK, tool call dumped as raw JSON *text* in `content`, `tool_calls: null`, `finish_reason: "stop"` |
| `deepseek-r1:14b` | 200 OK, human-oriented prose describing the tool, including hallucinated CLI syntax |
| `qwen3:8b` / `gemma4:12b` | 200 OK, populated `tool_calls`, `finish_reason: "tool_calls"` — the only accepted shape |

The `qwen2.5-coder` case matters most: it is the one that advertised
`tools` in metadata and still failed, so a self-test that merely checks
for a 200 or for `tool_calls` being *present* would pass it. The assertion
must be that `tool_calls` is **populated**.

**3. Network-dependent tests must be skippable.** Hosted-provider and
Tavily live tests require API keys. Gate them behind
`pytest.mark.skipif(not os.environ.get(...))` so a contributor with no
keys can still run the full suite green. Mocked equivalents carry the
logic coverage; the live tests confirm the integration.

**4. Setup wizard detection.** Record real `nvidia-smi` / `rocm-smi` /
`system_profiler` outputs as fixtures and parse them in tests, including
the ambiguous/failed-detection cases that trigger the guided-question
fallback. Detection is pure parsing once the subprocess is mocked.

**5. Regression protection for the concurrency invariants.** Adding a
`provider` column touches `try_claim_with_cap` / `claim_next_queued_job` —
the exact code hardened over 7 review rounds against stranded jobs, and
the tier-thrashing fix from `6e30989`. `test_db.py`'s 30 tests cover this;
they must stay green unmodified. If a change requires editing them, that
is a signal the invariant moved, not that the test is stale.

**6. Live verification gaps outstanding** (this project's established
standard is that mocked tests alone don't count as verified):

- **Tavily `web_search`** — implemented and unit-tested, still never
  called against the real API.
- **Official (non-abliterated) Gemma tool-calling** — see open question 2.

## Open questions (not yet decided)

1. **What should the project be renamed to? — STILL OPEN.**
   `handyman` is the current **working placeholder**, replacing the
   earlier `oddjob` placeholder. Not confirmed. Treat every occurrence in
   code, config, and docs as rename-pending, and keep it confined so the
   eventual rename stays mechanical.

   **Surface mapping, decided conditionally** (applies if `handyman` is
   kept; the same shape applies to whatever name wins):

   | Surface | Value |
   |---|---|
   | PyPI distribution | `handyman-agent` |
   | Import package / package dir | `handyman` |
   | CLI command | `handyman` |
   | Env prefix | `HANDYMAN_` |
   | Config dir | `~/.config/handyman/` (`%APPDATA%` on Windows) |
   | Repo | `github.com/sachingulati/handyman-agent` |

   **Why the distribution name is hyphenated but the import name is not:**
   bare `handyman` is already taken on PyPI (also npm, and ~3,289 GitHub
   repos). Accepted deliberately for short imports and a short CLI verb,
   with the tradeoff understood: if a user ever installs the unrelated
   PyPI `handyman` alongside this one, the import names clash. Do not
   "fix" the mismatch — it is the point.

   **Availability, checked live across PyPI, npm, and GitHub** (don't
   re-run this): `handyman-agent` and `handyman-mcp` are free on all
   three. Other candidates still free if a rename is wanted: `prepcook`
   (clean on all three), `secondchair`, `understudy-mcp`, `standin` (npm
   taken). Confirmed taken as bare PyPI names: `handyman`, `oddjob`,
   `understudy`, `apprentice`, `deputy`, `errand`, `sidekick`,
   `journeyman`, `subagent`, `minion`, `gopher`, `legwork`, `factotum`,
   `adjutant`, `souschef`, `drudge`, `roustabout`, `handoff`.

   **Rejected outright:** `oddjob` — collides with the established Oddjob
   Java job scheduler (`uk.co.rgordon:oddjob`, still actively released)
   *and* its own `oddjob-mcp` wrapper, in this project's own problem
   domain.

2. **Tool-calling on *official* (non-abliterated) Gemma is still
   unverified locally.** Sizing is now confirmed identical to the
   abliterated builds, and the abliterated 12b passes, so the prior is
   strong — but the default path's model has never itself passed the live
   self-test. Not a design blocker (the wizard self-tests on the user's
   machine regardless), but it should be confirmed once before release so
   the documented default isn't shipping on inference alone.

**Resolved, no longer open:**

- **Tavily opt-in for `web_search`** — implemented, see "Related: web
  search/research tooling" above, commits `eb98d60`, `767c194`.
- **Data flow and testing plan** — presented and agreed; see the "Data
  flow" and "Testing plan" sections.
- **Hosted overflow behavior when the local cap is hit** — queue as today
  by default, with an explicit `allow_hosted` opt-in on `gemma_delegate`.
  See the provider decision rule in "Data flow".
- **Should Ollama-only be revisited?** — **Yes, narrowly.** Resolved this
  round: add a hosted provider restricted to OpenAI-compatible endpoints,
  with native Google search grounding explicitly cut from v1. Full scope,
  boundaries, and error-handling rules in the "Hosted provider" section
  above.
- **DeepSeek: deprioritize or keep trying?** — **Deprioritize, no
  denylist.** See "Model selection policy" above.
- **Prefer Qwen3 for low/mid tiers?** — Replaced by a better-framed
  answer: **no hardcoded family preference at all**, just a curated list
  ordered by measured size behind the self-test gate. Qwen3 does win the
  low tiers on the current measurements — because Gemma has no build
  smaller than 7.2GB. See "Model selection policy" above.

## Next steps

1. ~~Resolve the project rename.~~ Deferred by the user; `handyman` is the
   working placeholder. Candidate research is captured in open question 1
   so the decision can be made later without re-running it.
2. ~~Decide whether to spend more effort verifying DeepSeek, or
   deprioritize it for now.~~ **Done** — deprioritized without a denylist;
   see "Model selection policy".
3. ~~Decide on the hosted-fallback proposal.~~ **Done** — yes, restricted
   to OpenAI-compatible endpoints; see the "Hosted provider" section.
4. ~~Resume and finish the brainstorming skill's remaining steps (data
   flow, testing, spec self-review).~~ **Done** — see the "Data flow" and
   "Testing plan" sections; self-review pass applied inline. Next action
   is `superpowers:writing-plans`.
5. **Code implemented so far, ahead of the design being finalized:** the
   Tavily `web_search` integration (commits `eb98d60`, `767c194`) was
   built directly against the *current single-machine* `tools.py`/`config.py`,
   not yet against any generalized/config-file-driven structure from this
   doc's Architecture section - that structure doesn't exist as code yet.
   No conflict today, but keep this in mind once the setup-wizard/tier-config
   work actually starts, so the Tavily key resolution logic gets carried
   forward consistently rather than redone.
