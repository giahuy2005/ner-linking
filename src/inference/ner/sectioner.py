"""Tách raw_text .txt thành các section (EMR 3 header + QA 2 marker).

Header nhận diện theo NỘI DUNG (không dựa số thứ tự đầu dòng) vì data
thật lẫn cả mục lục QA không liên quan (vd "1. Bệnh dại có lây không?").
"""

from __future__ import annotations

import re

# Section 2 chấp nhận cả "Bệnh sử" trần (không có "hiện tại").
_SECTION_PATTERNS = [
    (1, re.compile(r'^\s*tiền\s*sử\s*bệnh(?:\s*lý)?\s*$', re.IGNORECASE)),
    (2, re.compile(r'^\s*(?:tiền\s*sử\s*bệnh\s*hiện\s*tại|bệnh\s*sử(?:\s*hiện\s*tại)?)\s*$', re.IGNORECASE)),
    (3, re.compile(r'^\s*(?:đánh\s*giá|khám)\s*tại\s*bệnh\s*viện\s*$', re.IGNORECASE)),
]

SECTION_TITLES = {
    0: "KHÔNG_XÁC_ĐỊNH",
    1: "Tiền sử bệnh",
    2: "Tiền sử bệnh hiện tại",
    3: "Đánh giá tại bệnh viện",
    4: "Hỏi (QA)",
    5: "Trả lời (QA)",
}

# Dòng ứng viên header EMR phải NGẮN (<=40 ký tự) và không chứa dấu câu
# của câu hỏi/liệt kê ('?', '!', ':', '•') -> tự loại mục lục kiểu
# "1. Bệnh dại có lây không?" và dòng nhiễm bẩn kiểu
# "3. Đánh giá tại bệnh viện • Tim đập nhanh..." mà không tạo split sai.
_CANDIDATE_LINE_RE = re.compile(
    r'(?m)^[ \t]*(?:[0-9]{1,2}[.)]|[-•])?[ \t]*([^\n?!:•]{1,40}?)[ \t]*$'
)

# Marker QA match ngay đầu dòng + bắt buộc có ':' theo sau, KHÔNG yêu cầu
# cả dòng chỉ có mỗi nhãn (nội dung câu hỏi thường dính ngay sau ':').
_QA_PATTERNS = [
    (4, re.compile(r'(?m)^[ \t]*(?:câu\s*hỏi\s*từ\s*người\s*dùng|hỏi)[ \t]*:', re.IGNORECASE)),
    (5, re.compile(r'(?m)^[ \t]*(?:câu\s*trả\s*lời\s*của\s*bác\s*s[iĩ]|trả\s*lời)[ \t]*:', re.IGNORECASE)),
]


def _drop_nested_matches(matches):
    """Loại match có span nằm hoàn toàn trong 1 match khác đã giữ lại."""
    ordered = sorted(matches, key=lambda t: (t[1], -(t[2] - t[1])))
    kept = []
    for sec_no, start, end in ordered:
        if kept and start >= kept[-1][1] and end <= kept[-1][2]:
            continue
        kept.append((sec_no, start, end))
    return kept


def split_sections_by_header(raw_text: str) -> dict[int, dict[str, str]]:
    """Trả về {section_no: {"title":..., "body":...}}."""
    matches = []
    for m in _CANDIDATE_LINE_RE.finditer(raw_text):
        line = m.group(1).strip()
        if not line:
            continue
        for sec_no, pat in _SECTION_PATTERNS:
            if pat.match(line):
                matches.append((sec_no, m.start(), m.end()))
                break

    for sec_no, pat in _QA_PATTERNS:
        for m in pat.finditer(raw_text):
            matches.append((sec_no, m.start(), m.end()))

    if not matches:
        return {0: {"title": SECTION_TITLES[0] + " (trước header đầu tiên)", "body": raw_text.strip()}}

    matches = _drop_nested_matches(matches)
    matches.sort(key=lambda t: t[1])

    sections: dict[int, dict[str, str]] = {}
    for i, (sec_no, header_start, header_end) in enumerate(matches):
        body_start = header_end
        body_end = matches[i + 1][1] if i + 1 < len(matches) else len(raw_text)
        body = raw_text[body_start:body_end].strip()
        if not body:
            continue
        if sec_no in sections:
            sections[sec_no]["body"] += "\n" + body
        else:
            sections[sec_no] = {"title": SECTION_TITLES[sec_no], "body": body}

    preamble = raw_text[:matches[0][1]].strip()
    if preamble:
        sections[0] = {"title": SECTION_TITLES[0] + " (trước header đầu tiên)", "body": preamble}

    return sections