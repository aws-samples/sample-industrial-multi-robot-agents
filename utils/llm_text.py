"""Flatten LangChain message content to plain text.

``AIMessage.content`` is typed ``str | list[str | dict]``. Which one you get is a
property of the *provider*, not of the call: OpenAI-style backends return a bare
string, while Bedrock Converse (this repo's default ``LLM_PROVIDER``) returns a
list of blocks as soon as the model emits anything besides one plain answer —
extended reasoning, or a tool call alongside prose.

Every caller that treats ``.content`` as a string is therefore one provider
switch (or one reasoning-enabled model) away from a ``TypeError``. Route reads
through ``extract_text()`` instead, so the caller holds a ``str`` by invariant.
"""
from __future__ import annotations


def extract_text(content) -> str:
    """Flatten LangChain message content (str or list of blocks) to plain text.

    Handles Bedrock Converse block shapes:
      - {"type": "text", "text": "..."}
      - {"type": "reasoning_content", "reasoning_content": {"text": "...", ...}}
    Only string fragments are collected; non-text blocks (e.g. tool_use) are skipped.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
            rc = block.get("reasoning_content")
            if isinstance(rc, str):
                parts.append(rc)
            elif isinstance(rc, dict) and isinstance(rc.get("text"), str):
                parts.append(rc["text"])
    return "\n".join(p for p in parts if p)
