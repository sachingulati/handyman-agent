# handyman

> **Note:** `handyman` is a working placeholder name and may change before release.

Delegate grunt work to a local LLM, so your expensive agent doesn't spend
tokens on it. Runs as an MCP server: your agent calls `gemma_delegate`, a
local model does the work in a background process, and your agent collects
the result later.

## Status

Pre-release. Works on one machine; being generalized for wider use.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally, with a tool-calling-capable model

## Install

```bash
uv pip install -e .
```

## Writing good delegated tasks

The local model is far more sensitive to ambiguity than a frontier model,
and the cost is not subtle. In testing, one internally inconsistent
instruction — a prompt that said "fix the imports in four files" and then
listed edits across three — cost roughly fourteen minutes, as the model
re-derived the contradiction turn after turn. The same model on a clean,
unambiguous task spent under three seconds deliberating.

Practical rules:

- State exact counts, and make sure they match what you then list.
- Give exact paths and exact strings. Do not make the model search.
- Prefer "write this whole file" over "edit part of that file".
- Say explicitly when the task is finished, so it stops.

## Safety: what the path jail does and does not cover

Delegated jobs run with real filesystem and shell access. Two things
constrain them, and it is important to understand the limits of both.

**What is enforced.** The file tools (`read_file`, `write_file`,
`edit_file`) resolve every path against the job's `working_dir` and refuse
paths that escape it, including via symlinks and Windows directory
junctions.

**What is NOT enforced.** `run_bash` executes arbitrary shell commands. It
runs *with `working_dir` as its current directory*, but nothing stops a
command from using an absolute path, deleting files, making network
requests, or otherwise touching anything your user account can touch. The
path jail constrains the file tools, not the shell.

Treat a delegated job as running with your full user privileges, because it
does. Only delegate to a working directory you would be comfortable handing
to an unattended script, and read the task you are delegating.

## License

MIT — see `LICENSE`.
