import json
from types import SimpleNamespace

import pytest

from hodor import agent


def _make_tool_call(name: str, arguments: dict, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def test_execute_tools_parallel_injects_context(monkeypatch):
    captured = {}

    def fake_execute_tool(tool_name, arguments, platform, token, gitlab_url):
        captured["tool_name"] = tool_name
        captured["arguments"] = arguments
        return {"ok": True}

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    ctx = agent.ReviewContext(owner="foo", repo="bar", pr_number=7)
    call = _make_tool_call("search_tests", {"file_path": "api.py"})

    results = agent.execute_tools_parallel(
        [call],
        platform="github",
        token=None,
        gitlab_url=None,
        max_workers=1,
        review_context=ctx,
    )

    assert captured["tool_name"] == "search_tests"
    assert captured["arguments"]["owner"] == "foo"
    assert captured["arguments"]["repo"] == "bar"
    assert captured["arguments"]["pr_number"] == 7
    assert results and results[0]["role"] == "tool"


def test_execute_tools_parallel_blocks_unknown_file(monkeypatch):
    called = False

    def fake_execute_tool(tool_name, arguments, platform, token, gitlab_url):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(agent, "execute_tool", fake_execute_tool)

    ctx = agent.ReviewContext(owner="foo", repo="bar", pr_number=7)
    ctx.files_to_cover = {"src/app.py"}
    call = _make_tool_call("fetch_file_diff", {"file_path": "missing.py"})

    results = agent.execute_tools_parallel(
        [call],
        platform="github",
        token=None,
        gitlab_url=None,
        max_workers=1,
        review_context=ctx,
    )

    assert not called  # ensure execute_tool never ran
    content = json.loads(results[0]["content"])
    assert "not part of the tracked PR files" in content["error"]
