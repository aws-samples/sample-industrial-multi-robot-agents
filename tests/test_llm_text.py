"""Content-block flattening — the dashboard chat route depends on it.

The bug this guards: ``/api/chat`` assigned ``AIMessage.content`` straight into
``reply_text`` and then ran ``re.search`` over it. Under Bedrock Converse that
content is a *list* of blocks, so the route raised
``TypeError: expected string or bytes-like object, got 'list'``, Flask returned
its HTML traceback page, and the browser reported the real symptom the operator
saw: ``SyntaxError: Unexpected token '<'``.
"""
import re

import pytest

from utils.llm_text import extract_text

# The regex the chat route runs over the flattened reply.
ACTION_RE = r'\{[^{}]*"action"[^{}]*\}'


def test_plain_string_passes_through():
    assert extract_text("hello") == "hello"


def test_none_becomes_empty_string():
    assert extract_text(None) == ""


def test_single_text_block():
    assert extract_text([{"type": "text", "text": "hello"}]) == "hello"


def test_bare_strings_in_list():
    assert extract_text(["a", "b"]) == "a\nb"


def test_reasoning_content_nested_dict():
    content = [
        {"type": "reasoning_content",
         "reasoning_content": {"text": "thinking out loud", "signature": "sig"}},
        {"type": "text", "text": "the answer"},
    ]
    assert extract_text(content) == "thinking out loud\nthe answer"


def test_reasoning_content_flat_string():
    assert extract_text([{"reasoning_content": "just text"}]) == "just text"


def test_tool_use_blocks_are_skipped():
    content = [
        {"type": "text", "text": "calling a tool"},
        {"type": "tool_use", "id": "t1", "name": "plan_trajectories", "input": {}},
    ]
    assert extract_text(content) == "calling a tool"


def test_empty_fragments_dropped():
    assert extract_text([{"type": "text", "text": ""},
                         {"type": "text", "text": "kept"}]) == "kept"


def test_non_str_non_list_is_stringified():
    assert extract_text(42) == "42"


def test_blocks_without_text_yield_empty_string():
    assert extract_text([{"type": "tool_use", "name": "x", "input": {}}]) == ""


def test_flattened_content_is_regex_searchable():
    """The exact failure mode: re.search over raw block content raises."""
    content = [{"type": "text", "text": 'ok {"action": "reset"} done'}]

    with pytest.raises(TypeError):
        re.search(ACTION_RE, content)          # what the route used to do

    match = re.search(ACTION_RE, extract_text(content))   # what it does now
    assert match and match.group(0) == '{"action": "reset"}'
