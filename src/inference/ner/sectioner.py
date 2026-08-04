"""Offset-preserving EMR/QA sectioning from the final prediction notebook."""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
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


# ---------------------------------------------------------------------------
# Assertion-only section resolver
# ---------------------------------------------------------------------------
# This resolver is intentionally separate from ``split_sections_by_header``.
# It never changes NER chunk boundaries; it only supplies deterministic scope
# metadata to the assertion validator.
ASSERTION_SECTION_UNKNOWN = "unknown"
ASSERTION_SECTION_HISTORICAL = "historical"
ASSERTION_SECTION_CURRENT = "current"
ASSERTION_SECTION_FAMILY = "family_history"
ASSERTION_SECTION_GENERAL = "general"

_ASSERTION_LINE_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:(?P<number>\d{1,2})[.)][ \t]*)?"
    r"(?P<body>[^\r\n]{1,5000})$"
)
_ASSERTION_NUMBERED_CONTENT_RE = re.compile(
    r"(?iu)\b(?:\d+(?:[.,]\d+)?\s*(?:mg|mcg|g|ml|l|iu|đơn\s+vị)|"
    r"po|iv|im|sc|sq|prn|bid|tid|qid|qhs|qam|daily)\b"
)
_INLINE_ASSERTION_HEADING_RE = re.compile(
    r"(?iu)(?:[.:][ \t]*|[ \t]{2,})"
    r"(?P<body>(?:\d{1,2}[.)][ \t]*)?(?:"
    r"tiền\s+sử\s+bệnh\s+hiện\s+tại|tiền\s+sử\s+bệnh(?:\s+lý)?|"
    r"tiền\s+sử\s+phẫu\s+thuật(?:\s*/\s*thủ\s+thuật)?|"
    r"thuốc\s+trước\s+(?:khi\s+)?nhập\s+viện|"
    r"bệnh\s+sử(?:\s+hiện\s+tại)?|diễn\s+biến\s+bệnh|"
    r"triệu\s+chứng\s+(?:hiện\s+tại|khi\s+nhập\s+viện)|"
    r"tình\s+trạng\s+(?:hiện\s+tại|ngay\s+trước\s+khi\s+nhập\s+viện)|"
    r"câu\s+hỏi(?:\s+từ\s+người\s+dùng)?|câu\s+trả\s+lời\s+của\s+bác\s+sĩ"
    r"))"
)


def _normalize_assertion_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"^[ \t]*[-•*]+[ \t]*", "", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized).strip()
    return normalized.strip(" \t.:;,-–—()[]{}")


def _classify_assertion_heading(body: str, *, numbered: bool) -> str | None:
    heading = _normalize_assertion_heading(body)
    if not heading:
        return None

    # Current-state headings must be tested before the broader historical
    # prefixes because "tiền sử bệnh hiện tại" starts with "tiền sử bệnh".
    current_patterns = (
        r"^tiền sử bệnh hiện tại(?:\b|$)",
        r"^bệnh sử(?: hiện tại)?(?:\b|$)",
        r"^diễn biến(?: bệnh)?(?: hiện tại)?(?:\b|$)",
        r"^quá trình bệnh lý hiện tại(?:\b|$)",
        r"^lý do (?:vào|nhập) viện(?:\b|$)",
        r"^thời điểm khởi phát(?: triệu chứng)?(?:\b|$)",
        r"^(?:các )?triệu chứng (?:hiện tại|khi nhập viện)(?:\b|$)",
        r"^đặc điểm triệu chứng(?:\b|$)",
        r"^tình trạng (?:hiện tại|khi nhập viện|ngay trước khi nhập viện)(?:\b|$)",
        r"^(?:đánh giá|khám|thăm khám) tại bệnh viện(?:\b|$)",
        r"^kết quả đánh giá ban đầu(?:\b|$)",
        r"^kết quả (?:xét nghiệm|chẩn đoán hình ảnh)(?:\b|$)",
        r"^các phát hiện chẩn đoán khác(?:\b|$)",
    )
    if any(re.search(pattern, heading, flags=re.I) for pattern in current_patterns):
        return ASSERTION_SECTION_CURRENT

    family_exact = {
        "tiền sử gia đình", "bệnh sử gia đình", "family history",
    }
    if (
        heading in family_exact
        or re.match(r"^(?:tiền sử|bệnh sử) gia đình\s*:", body.casefold().strip())
    ):
        return ASSERTION_SECTION_FAMILY

    historical_patterns = (
        r"^tiền sử bệnh(?: lý)?(?:\b|$)",
        r"^tiền sử y khoa(?:\b|$)",
        r"^bệnh sử trước đây(?:\b|$)",
        r"^tiền sử bản thân(?:\b|$)",
        r"^tiền sử phẫu thuật(?:\b|$)",
        r"^(?:các )?bệnh lý mạn tính(?:\b|$)",
        r"^(?:các )?bệnh lý mãn tính(?:\b|$)",
        r"^thuốc trước (?:khi )?nhập viện(?:\b|$)",
        r"^danh sách thuốc trước nhập viện(?:\b|$)",
        r"^(?:các )?sự kiện trước khi nhập viện(?:\b|$)",
        r"^các tập (?:phát bệnh )?tương tự trước đây(?:\b|$)",
    )
    if any(re.search(pattern, heading, flags=re.I) for pattern in historical_patterns):
        return ASSERTION_SECTION_HISTORICAL

    general_patterns = (
        r"^(?:yếu tố nguy cơ|nguyên nhân|cơ chế|dịch tễ|tiền sử dịch tễ)(?:\b|$)",
        r"^(?:chẩn đoán|tiêu chuẩn chẩn đoán|điều trị|phòng ngừa|biến chứng)(?:\b|$)",
        r"^(?:không thay đổi được|có thể thay đổi)(?:\b|$)",
        r"^(?:cần làm gì|lưu ý|khuyến cáo|tổng quan|định nghĩa)(?:\b|$)",
        r"^(?:câu hỏi|câu trả lời|bác sĩ trả lời|trả lời|lời khuyên)(?:\b|$)",
        r"^(?:chào bạn|như bạn đã mô tả|cảm ơn bạn đã gửi câu hỏi)(?:\b|$)",
        r"^lưu ý quan trọng(?:\b|$)",
    )
    if any(re.search(pattern, heading, flags=re.I) for pattern in general_patterns):
        return ASSERTION_SECTION_GENERAL

    # A numbered line usually starts a new top-level section and therefore
    # safely resets a prior historical scope. Do not treat medication list
    # rows such as "1. amlodipine 10 mg po daily" as headings.
    if numbered and not _ASSERTION_NUMBERED_CONTENT_RE.search(heading):
        return ASSERTION_SECTION_GENERAL
    return None


_ASSERTION_GENERAL_BULLET_RE = re.compile(
    r"(?iu)^[ \t]*[-•*][ \t]*(?:"
    r"(?:không\s+nên|nên|cần|hãy|khuyến\s+cáo)\b|"
    r"(?:uống|thực\s+hiện|ghép|súc|chải|tái\s+khám|đi\s+khám)\b"
    r"(?=[^\n]{0,180}\b(?:để|giúp|nhằm|hỗ\s+trợ|ngăn|phòng|"
    r"loại\s+bỏ|khử|làm\s+dịu|tái\s+tạo)\b)|"
    r"trà\s+[^:;,.]{1,80}\b(?:giúp|để|nhằm)\b"
    r")"
)


def build_assertion_section_blocks(raw_text: str) -> list[dict[str, Any]]:
    """Build offset-preserving assertion scope blocks.

    The output is independent from inference sectioning and therefore cannot
    change tokenization, model chunks, or entity offsets.
    """
    headings: list[dict[str, Any]] = []
    cursor = 0
    for line in raw_text.splitlines(keepends=True):
        visible = line.rstrip("\r\n")
        match = _ASSERTION_LINE_RE.match(visible)
        line_heading_added = False
        if _ASSERTION_GENERAL_BULLET_RE.search(visible):
            headings.append({
                "start": cursor,
                "heading_end": cursor + len(visible),
                "kind": ASSERTION_SECTION_GENERAL,
                "heading": visible,
            })
            line_heading_added = True
        if match is not None and not line_heading_added:
            # Bullet rows are content, not section headings. Known unbulleted
            # subheadings and numbered top-level headings are considered.
            stripped = visible.lstrip()
            if not stripped.startswith(("-", "•", "*")):
                numbered = match.group("number") is not None
                body = match.group("body")
                kind = _classify_assertion_heading(body, numbered=numbered)
                if kind is not None:
                    headings.append({
                        "start": cursor,
                        "heading_end": cursor + len(visible),
                        "kind": kind,
                        "heading": visible,
                    })
                    line_heading_added = True

        # Some synthetic/dirty records concatenate the next heading onto the
        # preceding sentence. Detect only a closed set of heading prefixes
        # after punctuation or a wide whitespace gap; ordinary prose such as
        # "bệnh nhân có tiền sử..." is not treated as a section boundary.
        for inline in (() if line_heading_added else _INLINE_ASSERTION_HEADING_RE.finditer(visible)):
            body = inline.group("body")
            body_start = inline.start("body")
            if body_start == 0:
                continue
            numbered = bool(re.match(r"\d{1,2}[.)]", body))
            kind = _classify_assertion_heading(body, numbered=numbered)
            if kind is None:
                continue
            headings.append({
                "start": cursor + body_start,
                "heading_end": cursor + body_start + len(body),
                "kind": kind,
                "heading": body,
            })
        cursor += len(line)

    if headings:
        deduplicated: dict[tuple[int, str], dict[str, Any]] = {}
        for item in headings:
            deduplicated[(int(item["start"]), str(item["kind"]))] = item
        headings = sorted(deduplicated.values(), key=lambda item: (item["start"], item["heading_end"]))

    if not headings:
        return [{
            "start": 0, "end": len(raw_text),
            "kind": ASSERTION_SECTION_UNKNOWN,
            "heading": None, "heading_end": None,
        }]

    blocks: list[dict[str, Any]] = []
    if headings[0]["start"] > 0:
        blocks.append({
            "start": 0, "end": headings[0]["start"],
            "kind": ASSERTION_SECTION_UNKNOWN,
            "heading": None, "heading_end": None,
        })
    for index, heading in enumerate(headings):
        end = headings[index + 1]["start"] if index + 1 < len(headings) else len(raw_text)
        blocks.append({**heading, "end": end})
    return blocks


def assertion_section_at(
    blocks: list[dict[str, Any]], position: int,
) -> dict[str, Any]:
    """Return the assertion block containing ``position``."""
    if not blocks:
        return {
            "start": 0, "end": 0, "kind": ASSERTION_SECTION_UNKNOWN,
            "heading": None, "heading_end": None,
        }
    starts = [int(block["start"]) for block in blocks]
    index = max(0, bisect_right(starts, int(position)) - 1)
    return blocks[index]