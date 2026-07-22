import json

import requests

from handyman import setup


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.text = payload, status, json.dumps(payload)

    def json(self):
        return self._p


def _msg(message):
    return _Resp({"choices": [{"message": message}]})


def _call(name="write_file", args='{"path": "probe.txt", "content": "hello"}'):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": "1", "function": {"name": name, "arguments": args}}]}


def test_accepts_a_model_that_really_calls_the_tool(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _msg(_call()))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert ok and why == "ok"


def test_rejects_a_tool_call_returned_as_plain_text(monkeypatch):
    """The failure that motivates this whole check: a 200 response, the
    metadata advertising tool support, and the call written into the
    message body with tool_calls empty."""
    monkeypatch.setattr(requests, "post", lambda *a, **k: _msg(
        {"role": "assistant",
         "content": '{"name": "write_file", "arguments": {"path": "probe.txt"}}',
         "tool_calls": None}))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "plain text" in why


def test_rejects_a_model_that_only_describes_the_tool(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _msg(
        {"role": "assistant",
         "content": "You could run: writefile --name=probe.txt --content=hello",
         "tool_calls": None}))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "did not call the tool" in why


def test_rejects_a_server_that_refuses_tools(monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _Resp({"error": "does not support tools"}, 400))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "400" in why


def test_rejects_malformed_tool_arguments(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _msg(_call(args="not json")))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "malformed" in why


def test_rejects_the_wrong_tool(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _msg(_call(name="read_file")))
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "instead of" in why


def test_unreachable_server_is_reported_plainly(monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", boom)
    ok, why = setup.verify_tool_calling("http://h", "m")
    assert not ok
    assert "could not reach" in why


def test_recommendations_fit_the_reported_vram():
    for model, size, _ in setup.recommend(8.0):
        assert size < 8.0, f"{model} would not fit in 8 GB"


def test_a_small_gpu_still_gets_an_option():
    assert setup.recommend(4.0)


def test_unknown_vram_offers_everything_rather_than_guessing():
    """A wrong guess silently picks a model that will not fit, so an
    unknown answer has to become a question."""
    assert len(setup.recommend(None)) == len(setup.CANDIDATES)


def test_recommendations_are_largest_first():
    sizes = [size for _, size, _ in setup.recommend(24.0)]
    assert sizes == sorted(sizes, reverse=True)


def test_build_config_is_loadable(tmp_path):
    import yaml

    from handyman import config

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(setup.build_config("qwen3:8b")), encoding="utf-8")
    cfg = config.load(path)
    assert cfg.tiers[0].model == "qwen3:8b"
    assert cfg.tiers[0].name == config.BASE_TIER_NAME
