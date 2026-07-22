# Delegating work

How to write tasks for a local model so they succeed. Written for an
agent (or person) calling `gemma_delegate`.

The model doing the work is small. It is fast, cheap, and private, but it
does exactly what it is told and infers nothing. Treat it as a capable
junior who cannot ask you questions.

## What to delegate

Good candidates:

- Mechanical file work — create, copy, move, bulk-rename
- Uniform transformations across many files
- Boilerplate and scaffolding
- Running commands and reporting output
- First drafts you intend to review anyway

Keep for yourself:

- Anything where a wrong answer is expensive or hard to detect
- Native/FFI calls, memory handling, cryptography, auth
- Design decisions and trade-offs
- Work needing context beyond the task text
- Changes you cannot verify afterwards

The dividing line is not difficulty. It is **whether you can check the
result**. Delegate work whose success is obvious; keep work whose failure
is subtle.

## Writing the task

**Be exhaustively specific.** Give exact paths, exact strings, exact
values. Never make the model search for what you meant.

**Keep it internally consistent.** If you say "three files", list three.
A contradiction is the single most expensive mistake you can make: the
model will try to reconcile it instead of working, and may never finish.

**Name the tool.** Say "use the write_file tool" or "run one shell
command". Left to choose, it may pick a slower or riskier route.

**State the finish line.** End with when the task is done and that it
should stop. Without this, it can keep going long after the work is
complete.

**Prefer whole files over edits.** "Write this entire file" is more
reliable than "change this part of that file", especially when the
replacement is indented inside a function.

**Split large work.** Several small tasks beat one large one. Each has a
better chance of finishing, and you find out sooner when something is
wrong.

## Choosing the approach

| Goal | Ask for |
|---|---|
| File must exist with content you already have | a shell copy command |
| Same change across many files | one shell command doing all of them |
| One-line change | a single targeted edit |
| New code | a whole new file, with tests you wrote |

Shell commands are the most reliable and fastest route. Prefer them for
anything mechanical. Note that shell behaviour differs between platforms,
so avoid quoting tricks, pipes and wildcards; when a command gets
complicated, put it in a script file and ask for the script to be run.

## Verifying

Always check the result yourself. Do not rely on the job's own summary —
it reports what the model believes it did.

- Diff against what you expected
- Run the tests
- Check the job's status and recent events, not just the final message

If you wrote tests first, they are both the specification and the check.
This is the most reliable way to delegate new code: you define the
contract, the model fills it in, the tests decide whether it worked.

## When something goes wrong

Expect to iterate. Two or three short rounds beat one perfect attempt.

When a result is wrong, **say exactly what is wrong and exactly what it
should be**. "Make the tests pass" rarely works. Naming the defect and
its fix usually does.

Fix every occurrence explicitly. If the same mistake appears in three
places, say so — correcting one will not correct the others.

Before blaming the model, check the failure. A surprising number of
"model errors" are unclear instructions, environment problems, or tool
failures. The job's recorded events and any command's exit code and error
output will usually tell you which.
