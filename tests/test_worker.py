from handyman import db
from handyman import worker
def _job(tmp_path):
    conn = db.connect(tmp_path / "jobs.db")
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    return conn, job_id


def test_run_job_completes_on_task_complete(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE all done here"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert "all done here" in job["result_summary"]


def test_run_job_executes_tool_call_then_completes(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    calls = []

    def chat_fn(messages):
        if len(calls) == 0:
            calls.append(1)
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "write_file", "arguments": '{"path": "x.txt", "content": "hi"}'},
                    }
                ],
            }
        return {"role": "assistant", "content": "TASK_COMPLETE wrote the file"}

    executed = []

    def execute_tool_fn(name, arguments):
        executed.append((name, arguments))
        return "wrote 2 bytes to x.txt"

    worker.run_job(
        conn, job_id, "write a file", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=execute_tool_fn,
    )

    assert executed == [("write_file", {"path": "x.txt", "content": "hi"})]
    assert db.get_job(conn, job_id)["status"] == "done"


def test_run_job_marks_incomplete_on_malformed_tool_arguments(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    # truncated/malformed JSON, as a local model might emit
                    "function": {"name": "read_file", "arguments": '{"path": '},
                }
            ],
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"
    assert "tool dispatch failed" in job["result_summary"]


def test_run_job_marks_incomplete_on_tool_call_missing_id(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                # no "id" field, as a local model might emit
                {"function": {"name": "read_file", "arguments": '{"path": "x"}'}}
            ],
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"
    assert "tool dispatch failed" in job["result_summary"]


def test_run_job_marks_incomplete_on_non_list_tool_calls(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {
            "role": "assistant",
            "content": None,
            # a local model might emit a bare scalar instead of a list here
            "tool_calls": 123,
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"
    assert "tool dispatch failed" in job["result_summary"]


def test_run_job_marks_incomplete_on_non_string_content(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        # a local model might emit a non-string content value (e.g. an int)
        return {"role": "assistant", "content": 5}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"


def test_run_job_marks_incomplete_on_none_message(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        # a local model's client wrapper might return None on a bad response
        return None

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"


def test_run_job_marks_incomplete_on_non_json_serializable_message(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        # a set is not JSON-serializable and could show up if a local model
        # wrapper does something unexpected with the parsed response
        return {"role": "assistant", "content": {1, 2, 3}}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"


def test_run_job_propagates_chat_fn_exception_uncaught(tmp_path):
    # chat_fn(messages) is deliberately OUTSIDE the guard, because in the
    # real (non-test) wiring it calls into ollama_client.chat(), which
    # raises OllamaError for genuine backend failures (e.g. "Ollama isn't
    # running"). Those must propagate out of run_job so worker.main() can
    # mark the job "error" instead of "incomplete". This test locks in
    # that boundary: any exception raised directly by chat_fn must NOT be
    # swallowed by run_job.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        raise RuntimeError("simulated ollama connection error")

    import pytest

    with pytest.raises(RuntimeError, match="simulated ollama connection error"):
        worker.run_job(
            conn, job_id, "do a thing", str(tmp_path), log_path,
            chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
            watchdog_max_retries=3,
        )

    # the job must still be in a non-terminal status: run_job did not
    # (and should not) touch db.update_status for this failure -- that's
    # worker.main()'s job.
    job = db.get_job(conn, job_id)
    assert job["status"] not in ("done", "incomplete", "canceled", "timeout")


def test_run_job_watchdog_nudges_then_completes(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    call_count = {"n": 0}
    seen_messages = []

    def chat_fn(messages):
        call_count["n"] += 1
        seen_messages.append(list(messages))
        if call_count["n"] == 1:
            return {"role": "assistant", "content": "still thinking about it"}
        return {"role": "assistant", "content": "TASK_COMPLETE done now"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    assert call_count["n"] == 2
    assert db.get_job(conn, job_id)["status"] == "done"
    # the nudge must have been appended to the conversation before the 2nd call
    second_call_messages = seen_messages[1]
    assert any(
        m.get("content") == worker.WATCHDOG_NUDGE for m in second_call_messages
    )


def test_run_job_marks_incomplete_after_watchdog_exhausted(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {"role": "assistant", "content": "not done yet, still working"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=10, max_wall_clock_seconds=60,
        watchdog_max_retries=2,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "incomplete"
    assert "not done yet" in job["result_summary"]


def test_run_job_watchdog_resets_on_alternating_tool_calls(tmp_path):
    # A model that alternates "no tool call" and "tool call" across many
    # iterations must never trip the watchdog, because every tool call
    # resets the retry counter to 0. With watchdog_max_retries=1, if the
    # counter were NOT reset per tool call, the cumulative count of
    # no-tool-call turns below (6, non-consecutive) would exceed the cap
    # and the job would end up "incomplete" long before it can finish.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    call_count = {"n": 0}
    ALTERNATIONS = 12  # 6 pairs of (no-tool-call, tool-call)

    def chat_fn(messages):
        call_count["n"] += 1
        n = call_count["n"]
        if n > ALTERNATIONS:
            return {"role": "assistant", "content": "TASK_COMPLETE finished after alternating"}
        if n % 2 == 1:
            return {"role": "assistant", "content": "still working on it"}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{n}", "function": {"name": "read_file", "arguments": '{"path": "x"}'}}
            ],
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=ALTERNATIONS + 5, max_wall_clock_seconds=60,
        watchdog_max_retries=1, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert call_count["n"] == ALTERNATIONS + 1


def test_run_job_marks_timeout_after_max_iterations(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "function": {"name": "read_file", "arguments": '{"path": "x"}'}}
            ],
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=3, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "timeout"
    assert "iterations" in job["result_summary"]


def test_run_job_respects_cancel_flag_before_first_call(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    db.request_cancel(conn, job_id)

    def chat_fn(messages):
        raise AssertionError("chat_fn should not be called when already canceled")

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    assert db.get_job(conn, job_id)["status"] == "canceled"


def test_run_job_respects_cancel_flag_requested_mid_run(tmp_path):
    # chat_fn always returns a tool call, so without mid-run cancellation
    # the loop would run until max_iterations. Simulate an external cancel
    # (e.g. from another process) arriving while the job is actively
    # iterating, by requesting it from inside chat_fn itself.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    call_count = {"n": 0}

    def chat_fn(messages):
        call_count["n"] += 1
        if call_count["n"] == 3:
            db.request_cancel(conn, job_id)
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{call_count['n']}",
                    "function": {"name": "read_file", "arguments": '{"path": "x"}'},
                }
            ],
        }

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=100, max_wall_clock_seconds=60,
        watchdog_max_retries=3, execute_tool_fn=lambda name, args: "ok",
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "canceled"
    # cancel is only checked at the top of the next iteration, so chat_fn
    # is called exactly 3 times (the call that requests cancel is the
    # last one) -- and nowhere near max_iterations=100.
    assert call_count["n"] == 3


def test_estimate_tokens_grows_with_message_size():
    small = [{"role": "user", "content": "hi"}]
    large = [{"role": "user", "content": "hi" * 1000}]
    assert worker.estimate_tokens(large) > worker.estimate_tokens(small)


def test_run_job_uses_base_model_below_all_thresholds(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE done"}

    def escalated_fn(messages):
        raise AssertionError("escalation tier must not be used below its threshold")

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
        escalation_tiers=[(1_000_000, "mid", escalated_fn)],
    )

    assert db.get_job(conn, job_id)["status"] == "done"


def test_run_job_escalates_to_tier_above_its_threshold(tmp_path):
    # No sibling job is running, so escalation must proceed immediately
    # (is_sole_runner is trivially true) with no waiting.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        raise AssertionError("base chat_fn must not be called once past the threshold")

    def escalated_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE used the escalated model"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
        escalation_tiers=[(0, "mid", escalated_fn)],
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert "used the escalated model" in job["result_summary"]
    assert "escalated to mid" in log_path.read_text(encoding="utf-8")
    assert job["current_tier"] == "mid"
    assert job["escalating"] == 0


def test_run_job_picks_the_highest_crossed_tier(tmp_path):
    # Both the mid (threshold 0) and big (threshold 0) tiers are crossed
    # immediately - the highest one (big, listed last) must win, not the
    # first one that happened to match.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        raise AssertionError("base chat_fn must not be called")

    def mid_fn(messages):
        raise AssertionError("mid tier must not win when big tier is also crossed")

    def big_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE used the big tier"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
        escalation_tiers=[(0, "mid", mid_fn), (0, "big", big_fn)],
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert "used the big tier" in job["result_summary"]
    assert job["current_tier"] == "big"


def test_run_job_ignores_growth_thresholds_when_no_escalation_tiers(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"

    def chat_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE no escalation configured"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
    )

    assert db.get_job(conn, job_id)["status"] == "done"


def test_run_job_waits_for_sole_runner_before_escalating(tmp_path):
    # A sibling job is running on the base tier when this job wants to
    # escalate. It must wait (polling via sleep_fn, not consuming
    # max_iterations) until the sibling finishes, rather than escalating
    # immediately and thrashing Ollama's single GPU-resident model slot.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    sibling_id = db.create_job(conn, "sibling task", str(tmp_path))
    db.try_claim_with_cap(conn, sibling_id, pid=999999, max_concurrent=3)
    assert db.get_job(conn, sibling_id)["status"] == "running"

    sleep_calls = {"n": 0}

    def fake_sleep(seconds):
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 2:
            # simulate the sibling finishing after a couple of poll attempts
            db.update_status(conn, sibling_id, "done", result_summary="sibling done")

    def chat_fn(messages):
        raise AssertionError("base chat_fn must not be called once past the threshold")

    def escalated_fn(messages):
        return {"role": "assistant", "content": "TASK_COMPLETE escalated after waiting"}

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=60,
        watchdog_max_retries=3,
        escalation_tiers=[(0, "mid", escalated_fn)],
        sleep_fn=fake_sleep,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert "escalated after waiting" in job["result_summary"]
    assert sleep_calls["n"] >= 2
    assert "waiting to escalate" in log_path.read_text(encoding="utf-8")


def test_run_job_escalation_wait_respects_wall_clock_timeout(tmp_path):
    # The sibling never finishes, so the wait must still be bounded by the
    # job's own wall-clock cap rather than blocking forever.
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    sibling_id = db.create_job(conn, "sibling task", str(tmp_path))
    db.try_claim_with_cap(conn, sibling_id, pid=999999, max_concurrent=3)

    ticks = iter([0.0, 0.0, 5.0, 15.0])

    def time_fn():
        return next(ticks)

    def chat_fn(messages):
        raise AssertionError("base chat_fn must not be called")

    def escalated_fn(messages):
        raise AssertionError("escalated chat_fn must not be reached if the sibling never finishes")

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=10,
        watchdog_max_retries=3, time_fn=time_fn,
        escalation_tiers=[(0, "mid", escalated_fn)],
        sleep_fn=lambda seconds: None,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "timeout"
    assert job["escalating"] == 0


def test_run_job_times_out_on_wall_clock(tmp_path):
    conn, job_id = _job(tmp_path)
    log_path = tmp_path / "job.log"
    ticks = iter([0.0, 100.0, 200.0])

    def time_fn():
        return next(ticks)

    def chat_fn(messages):
        raise AssertionError("chat_fn should not be reached once wall-clock cap is exceeded")

    worker.run_job(
        conn, job_id, "do a thing", str(tmp_path), log_path,
        chat_fn=chat_fn, max_iterations=5, max_wall_clock_seconds=10,
        watchdog_max_retries=3, time_fn=time_fn,
    )

    job = db.get_job(conn, job_id)
    assert job["status"] == "timeout"
    assert "wall-clock" in job["result_summary"]
