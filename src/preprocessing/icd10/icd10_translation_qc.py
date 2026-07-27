#!/usr/bin/env python3
"""
src/preprocessing/icd10/icd10_translation_qc.py

Logic QC dùng chung, được gọi ở 2 chỗ:
  1. merge_icd.py -- NGAY sau mỗi batch dịch alias, TRƯỚC khi lưu checkpoint.
     Mục đích: checkpoint luôn ở trạng thái đã lọc sạch, không phải raw LLM
     output; nếu job bị dừng giữa chừng thì phần đã xong không cần hậu kiểm
     lại từ đầu.
  2. validate_and_filter_translations.py -- lượt tổng hợp cuối cùng, chủ yếu
     để build file embedding + lấy mẫu QA theo chương. Gọi lại qc_aliases()
     một lần nữa (rẻ, không tốn API) để áp dụng TERM_CORRECTIONS mới nếu
     Ghuy bổ sung thêm sau khi phần lớn dữ liệu đã dịch xong.

Không có state, không I/O -- import và gọi trực tiếp.
"""

import re

# ---------------------------------------------------------------------------
# Chương ICD-10 (suy trực tiếp từ code, không phụ thuộc field ngoài)
# ---------------------------------------------------------------------------

ICD10_CHAPTERS = [
    ("I",     "A00", "B99", "Bệnh nhiễm trùng và ký sinh trùng"),
    ("II",    "C00", "D48", "Khối u (bướu tân sinh)"),
    ("III",   "D50", "D89", "Bệnh máu, cơ quan tạo máu, miễn dịch"),
    ("IV",    "E00", "E90", "Bệnh nội tiết, dinh dưỡng, chuyển hoá"),
    ("V",     "F00", "F99", "Rối loạn tâm thần và hành vi"),
    ("VI",    "G00", "G99", "Bệnh hệ thần kinh"),
    ("VII",   "H00", "H59", "Bệnh mắt và phần phụ"),
    ("VIII",  "H60", "H95", "Bệnh tai và xương chũm"),
    ("IX",    "I00", "I99", "Bệnh hệ tuần hoàn"),
    ("X",     "J00", "J99", "Bệnh hệ hô hấp"),
    ("XI",    "K00", "K93", "Bệnh hệ tiêu hoá"),
    ("XII",   "L00", "L99", "Bệnh da và mô dưới da"),
    ("XIII",  "M00", "M99", "Bệnh hệ cơ xương khớp"),
    ("XIV",   "N00", "N99", "Bệnh hệ sinh dục tiết niệu"),
    ("XV",    "O00", "O99", "Thai nghén, sinh đẻ, hậu sản"),
    ("XVI",   "P00", "P96", "Bệnh chu sinh"),
    ("XVII",  "Q00", "Q99", "Dị tật bẩm sinh"),
    ("XVIII", "R00", "R99", "Triệu chứng, dấu hiệu bất thường"),
    ("XIX",   "S00", "T98", "Chấn thương, ngộ độc"),
    ("XX",    "V01", "Y98", "Nguyên nhân ngoại sinh"),
    ("XXI",   "Z00", "Z99", "Yếu tố ảnh hưởng sức khoẻ"),
    ("XXII",  "U00", "U99", "Mã dự phòng / đặc biệt"),
]


def _code_key(code: str):
    m = re.match(r"^([A-Z])(\d{2})", code)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def chapter_of(code: str) -> str:
    key = _code_key(code)
    if key is None:
        return "?"
    letter, num = key
    for roman, lo, hi, _label in ICD10_CHAPTERS:
        lo_l, lo_n = lo[0], int(lo[1:])
        hi_l, hi_n = hi[0], int(hi[1:])
        if lo_l == hi_l:
            if letter == lo_l and lo_n <= num <= hi_n:
                return roman
        else:
            if letter == lo_l and num >= lo_n:
                return roman
            if letter == hi_l and num <= hi_n:
                return roman
            if lo_l < letter < hi_l:
                return roman
    return "?"


def is_provisional_code(code: str) -> bool:
    """U13-U49: mã dự phòng / 'Emergency use of Uxx', không phải chẩn đoán thật."""
    key = _code_key(code)
    if key is None:
        return False
    letter, num = key
    return letter == "U" and 13 <= num <= 49


# ---------------------------------------------------------------------------
# Phân loại alias: đồng nghĩa thật vs coding note / mô tả phạm vi mã.
# Nguồn sự thật ưu tiên tiếng Anh (cấu trúc ICD-10 gốc chuẩn hoá hơn bản dịch).
# ---------------------------------------------------------------------------

CODING_NOTE_PATTERNS_EN = [
    r"^conditions? (classifiable|listed|specified|described)\b",
    r"^diseases? (classifiable|listed|specified)\b",
    r"^use additional code\b",
    r"^code (first|also)\b",
    r"\bclassifiable to [A-Z]\d{2}\b",
    r"^the conditions? in\b",
]

CODING_NOTE_PATTERNS_VI = [
    r"^các (tình trạng|bệnh|trường hợp)(\s+\S+){0,3}\s+(liệt kê|được liệt kê|nêu|đã nêu|mô tả)\s+(trong|ở)\s+[A-Z]\d",
    r"^tình trạng(\s+\S+){0,3}\s+(liệt kê|nêu)\s+(trong|ở)\s+[A-Z]\d",
    r"^(sử dụng|dùng)\s+thêm\s+mã\b",
    r"^mã\s+hoá\s+trước\b",
]

_EN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CODING_NOTE_PATTERNS_EN]
_VI_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CODING_NOTE_PATTERNS_VI]

# Câu mô tả phạm vi dài (vd A02: "Nhiễm trùng hoặc ngộ độc thực phẩm do
# Salmonella, không phải viêm dạ dày ruột...") thường không khớp pattern cố
# định, nhưng có đặc điểm: dài + nhiều mệnh đề nối bằng or/and/due to. Đây là
# tín hiệu phụ để ĐẨY SANG REVIEW (không tự loại, vì tên hội chứng dài vẫn
# có thể hợp lệ).
_CLAUSE_WORDS_EN = re.compile(r"\b(or|and|due to|except|excluding|not classified|without)\b", re.IGNORECASE)


def classify_alias(alias_en: str, alias_vi: str) -> str:
    """"coding_note" | "review" | "alias" """
    en = (alias_en or "").strip()
    vi = (alias_vi or "").strip()

    for pat in _EN_PATTERNS:
        if pat.search(en):
            return "coding_note"
    for pat in _VI_PATTERNS:
        if pat.search(vi):
            return "coding_note"

    word_count_en = len(en.split())
    clause_hits = len(_CLAUSE_WORDS_EN.findall(en))
    if word_count_en >= 9 and clause_hits >= 2:
        return "review"
    if word_count_en >= 14:
        return "review"

    return "alias"


# ---------------------------------------------------------------------------
# Sửa lỗi thuật ngữ đã biết -- khớp theo TỪ KHOÁ (mọi mã có alias_en này),
# không phải theo từng mã, vì LLM lặp lại cùng lỗi mỗi lần gặp thuật ngữ đó.
# Word-boundary substring match (KHÔNG exact-match): aliases_en ICD-10 hay có
# hậu tố như "NOS" (vd "Lupus vulgaris NOS"), exact-match sẽ bỏ sót phần lớn.
# Verify nguồn: y văn da liễu VN + Wikipedia tiếng Việt.
# ---------------------------------------------------------------------------

TERM_CORRECTIONS = {
    "scrofuloderma": "Lao tầng (scrofuloderma)",
    "lupus vulgaris": "Lupus vulgaris (lao da dạng lupus)",
    # Thêm cặp (alias_en lowercase -> bản dịch đúng) khi phát hiện lỗi hệ
    # thống mới qua batch QC log hoặc random-QA theo chương. Vì hàm này
    # được gọi lại ở validate_and_filter_translations.py, thêm entry mới ở
    # đây sẽ tự động áp dụng lại cho TOÀN BỘ dữ liệu đã dịch mà không cần
    # gọi lại OpenRouter.
}

_TERM_CORRECTION_PATTERNS = [
    (term, re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), fix)
    for term, fix in TERM_CORRECTIONS.items()
]


def apply_term_correction(alias_en: str, alias_vi: str):
    text = (alias_en or "").strip()
    for _term, pattern, fix in _TERM_CORRECTION_PATTERNS:
        if pattern.search(text):
            return fix, True
    return alias_vi, False


# ---------------------------------------------------------------------------
# Hàm tổng hợp: gọi 1 lần cho toàn bộ aliases_en/aliases_vi của 1 record.
# Đây là hàm merge_icd.py gọi NGAY sau khi 1 batch dịch xong, trước khi lưu
# checkpoint; và validate_and_filter_translations.py gọi lại ở lượt cuối.
# ---------------------------------------------------------------------------

def qc_aliases(code: str, aliases_en: list, aliases_vi_raw: list, source: str = "llm_openrouter"):
    """
    Trả về dict:
      {
        "aliases_en": [...],       # subset đã lọc (khớp 1:1 với aliases_vi)
        "aliases_vi": [...],       # đã sửa lỗi thuật ngữ + lọc coding_note/review
        "inclusion_notes_vi": [...],
        "aliases_vi_source": [...],# song song với "aliases_vi"
        "review_flags": [...],     # list dict, KHÔNG có "code" (caller tự thêm)
        "term_corrections_applied": [...],
      }
    """
    clean_en, clean_vi, clean_src = [], [], []
    inclusion_notes_vi = []
    review_flags = []
    corrections = []

    for en, vi in zip(aliases_en, aliases_vi_raw):
        vi_fixed, was_corrected = apply_term_correction(en, vi)
        if was_corrected:
            corrections.append({"alias_en": en, "before": vi, "after": vi_fixed})
        vi = vi_fixed

        label = classify_alias(en, vi)

        if label == "coding_note":
            inclusion_notes_vi.append(vi)
            continue

        if label == "review":
            inclusion_notes_vi.append(vi)  # an toàn hơn là lẫn vào alias
            review_flags.append({
                "type": "long_or_ambiguous", "alias_en": en, "alias_vi": vi, "source": source,
            })
            continue

        clean_en.append(en)
        clean_vi.append(vi)
        clean_src.append(source)
        if source == "llm_openrouter":
            review_flags.append({
                "type": "llm_translated_alias_needs_spotcheck",
                "alias_en": en, "alias_vi": vi, "source": source,
            })

    return {
        "aliases_en": clean_en,
        "aliases_vi": clean_vi,
        "aliases_vi_source": clean_src,
        "inclusion_notes_vi": inclusion_notes_vi,
        "review_flags": review_flags,
        "term_corrections_applied": corrections,
    }


def qc_preferred(preferred_en: str, preferred_vi_raw: str, source: str = "llm_openrouter"):
    """Áp term-correction cho preferred_vi (đơn lẻ, không phân loại coding_note)."""
    vi_fixed, was_corrected = apply_term_correction(preferred_en, preferred_vi_raw)
    review_flags = []
    if source == "llm_openrouter":
        review_flags.append({
            "type": "llm_translated_preferred_needs_spotcheck",
            "preferred_en": preferred_en, "preferred_vi": vi_fixed, "source": source,
        })
    return vi_fixed, was_corrected, review_flags