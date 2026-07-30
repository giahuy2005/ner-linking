"""Offset-preserving EMR/QA sectioning from the final prediction notebook."""

from __future__ import annotations

import re
from typing import Any

SECTION_TITLES = {
    0: "KHÔNG_XÁC_ĐỊNH",
    1: "Tiền sử bệnh",
    2: "Tiền sử bệnh hiện tại",
    3: "Đánh giá tại bệnh viện",
    4: "Hỏi đáp (QA)",
}

_HEADER_PREFIX = r"^[ \t]*(?:(?:[0-9]{1,2}[.)])|[-•])?[ \t]*"
_HEADER_BOUNDARY = r"(?=[ \t]*(?::|\.|•|[-–—]|\r?\n|\Z))"
_SECTION_PATTERNS = [
    (1, re.compile(
        rf"{_HEADER_PREFIX}(?:tiền\s+sử\s+bệnh(?:\s+lý)?|tiền\s+sử\s+y\s+khoa|"
        rf"bệnh\s+sử\s+trước\s+đây){_HEADER_BOUNDARY}",
        re.IGNORECASE | re.MULTILINE,
    )),
    (2, re.compile(
        rf"{_HEADER_PREFIX}(?:tiền\s+sử\s+bệnh\s+hiện\s+tại|"
        rf"bệnh\s+sử(?:\s+hiện\s+tại)?|diễn\s+biến\s+bệnh\s+hiện\s+tại|"
        rf"quá\s+trình\s+bệnh\s+lý\s+hiện\s+tại){_HEADER_BOUNDARY}",
        re.IGNORECASE | re.MULTILINE,
    )),
    (3, re.compile(
        rf"{_HEADER_PREFIX}(?:đánh\s+giá\s+tại\s+bệnh\s+viện|"
        rf"khám\s+tại\s+bệnh\s+viện|thăm\s+khám\s+tại\s+bệnh\s+viện|"
        rf"kết\s+quả\s+đánh\s+giá\s+ban\s+đầu|đánh\s+giá\s+ban\s+đầu)"
        rf"{_HEADER_BOUNDARY}",
        re.IGNORECASE | re.MULTILINE,
    )),
]

_QA_QUESTION_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:câu\s+hỏi(?:\s+(?:từ|của)\s+người\s+dùng)?"
    r"(?:\s+gửi\s+đến\s+hệ\s+thống)?)[ \t]*(?::|\r?$)|hỏi[ \t]*:)"
)
_QA_ANSWER_RE = re.compile(
    r"(?im)^[ \t]*(?:(?:câu\s+trả\s+lời\s+của\s+bác\s+s[iĩ]|"
    r"bác\s+s[iĩ]\s+trả\s+lời)[ \t]*(?::|\r?$)|trả\s+lời[ \t]*:)"
)


def is_qa_document(raw_text: str) -> bool:
    """QA is predicted as one complete document, never split question/answer."""
    return bool(_QA_QUESTION_RE.search(raw_text) and _QA_ANSWER_RE.search(raw_text))


def _deduplicate_header_matches(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(matches, key=lambda item: (
        item["start"], -(item["end"] - item["start"]), item["section_no"],
    ))
    kept = []
    for item in ordered:
        if any(item["start"] >= old["start"] and item["end"] <= old["end"] for old in kept):
            continue
        kept.append(item)
    return sorted(kept, key=lambda item: item["start"])


def split_sections_by_header(raw_text: str) -> dict[int, dict[str, Any]]:
    """Return ordered blocks; body is always exactly raw_text[start:end]."""
    if is_qa_document(raw_text):
        return {0: {
            "section_no": 4, "title": SECTION_TITLES[4],
            "start": 0, "end": len(raw_text),
            "header_start": None, "header_end": None,
            "matched_heading": None, "body": raw_text,
        }}

    matches = []
    for section_no, pattern in _SECTION_PATTERNS:
        for match in pattern.finditer(raw_text):
            matches.append({
                "section_no": section_no,
                "start": match.start(),
                "end": match.end(),
                "matched_heading": raw_text[match.start():match.end()],
            })
    matches = _deduplicate_header_matches(matches)
    if not matches:
        return {0: {
            "section_no": 0, "title": SECTION_TITLES[0],
            "start": 0, "end": len(raw_text),
            "header_start": None, "header_end": None,
            "matched_heading": None, "body": raw_text,
        }}

    blocks = []
    if raw_text[:matches[0]["start"]].strip():
        end = matches[0]["start"]
        blocks.append({
            "section_no": 0,
            "title": "KHÔNG_XÁC_ĐỊNH (trước header đầu tiên)",
            "start": 0, "end": end,
            "header_start": None, "header_end": None,
            "matched_heading": None, "body": raw_text[:end],
        })
    for index, match in enumerate(matches):
        start = match["start"]
        end = matches[index + 1]["start"] if index + 1 < len(matches) else len(raw_text)
        if not raw_text[start:end].strip():
            continue
        section_no = match["section_no"]
        blocks.append({
            "section_no": section_no, "title": SECTION_TITLES[section_no],
            "start": start, "end": end,
            "header_start": match["start"], "header_end": match["end"],
            "matched_heading": match["matched_heading"],
            "body": raw_text[start:end],
        })
    return {block_id: block for block_id, block in enumerate(blocks)}
