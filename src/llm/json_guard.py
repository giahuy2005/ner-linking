"""Tolerant but bounded JSON extraction for LLM responses.

The parser accepts fenced JSON, surrounding prose, and trailing commas. It does
not silently manufacture an empty object: callers receive ``None`` when every
candidate fails, so they can retry or fall back explicitly.
"""

from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _balanced_json_blocks(text: str) -> list[str]:
    """Return balanced top-level object/array substrings in source order."""
    blocks: list[str] = []
    start: int | None = None
    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"}": "{", "]": "["}

    for index, char in enumerate(text):
        if start is None:
            if char in "{[":
                start = index
                stack = [char]
                in_string = False
                escaped = False
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack or stack[-1] != pairs[char]:
                start = None
                stack = []
                in_string = False
                escaped = False
                continue
            stack.pop()
            if not stack:
                blocks.append(text[start:index + 1])
                start = None
    return blocks


def extract_json(text: str) -> dict | list | None:
    if not isinstance(text, str):
        return None

    candidates: list[str] = []
    for match in _FENCE_RE.finditer(text):
        fenced = match.group(1)
        candidates.extend(_balanced_json_blocks(fenced))
        candidates.append(fenced)
    candidates.extend(_balanced_json_blocks(text))
    candidates.append(text)

    seen: set[str] = set()
    for candidate in candidates:
        cleaned = _TRAILING_COMMA_RE.sub(r"\1", candidate.strip())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None