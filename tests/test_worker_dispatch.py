import subprocess
import uuid

from handyman import config
from handyman import db
from handyman import ollama_client
from handyman import worker
def _spawn_recorder(monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "spawn_worker", lambda job_id: calls.append(job_id))
    return calls


def _use_tmp_db(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setattr(config, "DB_PATH", db_path)
    return db.connect(db_path)


def test_execute_tool_call_write_and_read(tmp_path):
    result = worker.execute_tool_call(
        str(tmp_path), "write_file", {"path": "a.txt", "content": "hello"}
    )
    assert "wrote" in result
    result = worker.execute_tool_call(str(tmp_path), "read_file", {"path": "a.txt"})
    assert result == "hello"


def test_execute_tool_call_bash_returns_json(tmp_path):
    import sys

    result = worker.execute_tool_call(
        str(tmp_path), "bash", {"command": f'{sys.executable} -c "print(42)"'}
    )
    assert "42" in result


def test_execute_tool_call_unknown_tool_returns_error(tmp_path):
    result = worker.execute_tool_call(str(tmp_path), "not_a_real_tool", {})
    assert "error" in result
    assert "unknown tool" in result


def test_execute_tool_call_web_search_passes_configured_tavily_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TAVILY_API_KEY", "tvly-configured-key")
    captured = {}

    def fake_web_search(query, tavily_api_key=None):
        captured["query"] = query
        captured["tavily_api_key"] = tavily_api_key
        return [{"url": "https://example.com", "title": "Example"}]

    from handyman import tools
    monkeypatch.setattr(tools, "web_search", fake_web_search)

    worker.execute_tool_call(str(tmp_path), "web_search", {"query": "cats"})

    assert captured["query"] == "cats"
    assert captured["tavily_api_key"] == "tvly-configured-key"


def test_execute_tool_call_path_jail_violation_returns_error_not_raise(tmp_path):
    result = worker.execute_tool_call(
        str(tmp_path), "write_file", {"path": "../escape.txt", "content": "x"}
    )
    assert "error" in result
    assert "escapes" in result


def test_execute_tool_call_missing_file_returns_error_not_raise(tmp_path):
    result = worker.execute_tool_call(str(tmp_path), "read_file", {"path": "nope.txt"})
    assert "error" in result
    assert "no such file" in result


# --- spawn_worker subprocess stdout/stderr capture -------------------------


def test_spawn_worker_redirects_stdout_and_stderr_to_job_log(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "JOBS_LOG_DIR", tmp_path)
    job_id = uuid.uuid4().hex

    captured = {}

    def fake_popen(args, stdout=None, stderr=None, creationflags=None):
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        stdout.write("pre-crash output that must not be lost\n")

        class _FakeProcess:
            pass

        return _FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    worker.spawn_worker(job_id)

    expected_log_path = tmp_path / f"{job_id}.log"
    assert captured["stdout"].name == str(expected_log_path)
    assert captured["stderr"] == subprocess.STDOUT
    assert expected_log_path.read_text(encoding="utf-8") == (
        "pre-crash output that must not be lost\n"
    )


# --- main() hand-off and exception-handling tests -------------------------


def test_main_hands_off_to_next_queued_job_when_job_id_is_missing(tmp_path, monkeypatch):
    # job_id doesn't correspond to any row in the db.
    conn = _use_tmp_db(monkeypatch, tmp_path)
    missing_job_id = uuid.uuid4().hex

    second_job_id = db.create_job(conn, "real queued job", str(tmp_path))

    spawned = _spawn_recorder(monkeypatch)

    worker.main(missing_job_id)

    assert spawned == [second_job_id]
    assert db.get_job(conn, second_job_id)["status"] == "running"


def test_main_happy_path_reflects_run_job_status_and_hands_off(tmp_path, monkeypatch):
    conn = _use_tmp_db(monkeypatch, tmp_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    second_job_id = db.create_job(conn, "second queued job", str(tmp_path))

    monkeypatch.setattr(ollama_client, "model_is_pulled", lambda host, model: True)

    def fake_run_job(conn, job_id, *args, **kwargs):
        db.update_status(conn, job_id, "done", result_summary="ok")

    monkeypatch.setattr(worker, "run_job", fake_run_job)

    spawned = _spawn_recorder(monkeypatch)

    worker.main(job_id)

    job = db.get_job(conn, job_id)
    assert job["status"] == "done"
    assert job["result_summary"] == "ok"
    assert spawned == [second_job_id]
    assert db.get_job(conn, second_job_id)["status"] == "running"


def test_main_ollama_error_marks_job_error_and_hands_off(tmp_path, monkeypatch):
    conn = _use_tmp_db(monkeypatch, tmp_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    second_job_id = db.create_job(conn, "second queued job", str(tmp_path))

    error_message = "simulated: is it running?"

    def raise_ollama_error(host, model):
        raise ollama_client.OllamaError(error_message)

    monkeypatch.setattr(ollama_client, "model_is_pulled", raise_ollama_error)

    spawned = _spawn_recorder(monkeypatch)

    worker.main(job_id)

    job = db.get_job(conn, job_id)
    assert job["status"] == "error"
    assert error_message in job["result_summary"]
    assert spawned == [second_job_id]
    assert db.get_job(conn, second_job_id)["status"] == "running"


def test_main_generic_exception_marks_job_error_and_hands_off(tmp_path, monkeypatch):
    conn = _use_tmp_db(monkeypatch, tmp_path)
    job_id = db.create_job(conn, "do a thing", str(tmp_path))
    second_job_id = db.create_job(conn, "second queued job", str(tmp_path))

    def raise_runtime_error(host, model):
        raise RuntimeError("simulated unexpected failure")

    monkeypatch.setattr(ollama_client, "model_is_pulled", raise_runtime_error)

    spawned = _spawn_recorder(monkeypatch)

    worker.main(job_id)

    job = db.get_job(conn, job_id)
    assert job["status"] == "error"
    assert "RuntimeError" in job["result_summary"]
    assert "simulated unexpected failure" in job["result_summary"]
    assert spawned == [second_job_id]
    assert db.get_job(conn, second_job_id)["status"] == "running"
