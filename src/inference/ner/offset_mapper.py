"""Word-segment + char-offset mapping + chunking cho input dài.

Port nguyên logic từ predict_ner_crf_final.ipynb (đã test qua các case
bẩn thật trong data). KHÔNG đổi thuật toán match syllable ở đây nếu
chưa test lại trên cùng bộ .txt — offset lệch 1 ký tự sẽ làm sai cả
position lẫn text trả về, và WER/candidates score BTC chấm sẽ tụt theo.
"""

from __future__ import annotations

import re
import unicodedata

_TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}

_STUCK_PUNCT_RE = re.compile(
    r"(?<=[^\W\d_])[.!?](?=[^\W\d_])|"
    r"(?<=\S)[:;](?=\S)|"
    r"(?<=\S),(?=[^\s\d])|(?<=[^\d\s]),(?=\S)"
)


def _strip_tone(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in nfd if ch not in _TONE_MARKS)
    return unicodedata.normalize("NFC", stripped).lower()


def segment_with_offsets(text: str, rdr) -> tuple[list[str], list[tuple[int | None, int | None]], list[int]]:
    """Tokenize bằng VnCoreNLP rồi map lại char offset trên `text` gốc.

    Trả về (tokens, offsets, line_ids). offsets[i] = (None, None) nếu
    token không map được (KHÔNG raise — 1 token lạ không được crash cả
    request lúc inference); entity chứa token này bị is_valid_char_span()
    loại ở bước sau.
    """
    sentences = rdr.tokenize(text)
    word_tokens = [tok for sent in sentences for tok in sent]

    n = len(text)
    offsets: list[tuple[int | None, int | None]] = []
    mapped_tokens: list[str] = []
    cursor = 0
    for tok in word_tokens:
        mapped_tok = tok.strip("_") if tok.strip("_") else tok
        syllables = [part for part in mapped_tok.split("_") if part]
        if not syllables:
            syllables = [mapped_tok]

        tok_start, tok_end = None, None
        ok = True
        for syl in syllables:
            while cursor < n and text[cursor].isspace():
                cursor += 1
            target = _strip_tone(syl)
            matched = False
            for length in (len(syl), len(syl) + 1, len(syl) - 1, len(syl) + 2):
                if length <= 0 or cursor + length > n:
                    continue
                candidate = text[cursor:cursor + length]
                if _strip_tone(candidate) == target:
                    if tok_start is None:
                        tok_start = cursor
                    cursor += length
                    tok_end = cursor
                    matched = True
                    break
            if not matched:
                ok = False
                break
        if not ok:
            tok_start, tok_end = None, None

        offsets.append((tok_start, tok_end))
        mapped_tokens.append(mapped_tok)

    tokens, offsets = _split_stuck_sentence_punctuation(text, mapped_tokens, offsets)
    line_ids = [text.count("\n", 0, s) if s is not None else -1 for s, _ in offsets]
    return tokens, offsets, line_ids


def _split_stuck_sentence_punctuation(raw_text, tokens, offsets):
    """Tách dấu câu kết câu dính chữ (vd '...áp.Rối loạn...') ra token riêng,
    để 2 entity liền kề không tranh nhau 1 token BIO."""
    split_tokens = []
    split_offsets = []

    for token, (start, end) in zip(tokens, offsets):
        if start is None or end is None:
            split_tokens.append(token)
            split_offsets.append((start, end))
            continue

        raw_piece = raw_text[start:end]
        boundaries = list(_STUCK_PUNCT_RE.finditer(raw_piece))
        if not boundaries:
            split_tokens.append(token)
            split_offsets.append((start, end))
            continue

        cursor = 0
        for match in boundaries:
            punct_at = match.start()
            if cursor < punct_at:
                piece = raw_piece[cursor:punct_at]
                split_tokens.append(re.sub(r"\s+", "_", piece))
                split_offsets.append((start + cursor, start + punct_at))

            split_tokens.append(raw_piece[punct_at:punct_at + 1])
            split_offsets.append((start + punct_at, start + punct_at + 1))
            cursor = punct_at + 1

        if cursor < len(raw_piece):
            piece = raw_piece[cursor:]
            split_tokens.append(re.sub(r"\s+", "_", piece))
            split_offsets.append((start + cursor, end))

    return split_tokens, split_offsets


def encode_words_for_inference(tokenizer, words: list[str], max_len: int = 256):
    """Subword-encode danh sách word token, trả word_to_subword_start để
    map ngược tag CRF (predict per-subword) về word-level."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    unk_tok = tokenizer.unk_token

    input_ids = [cls_id]
    word_to_subword_start = []
    word_num_subtokens = []

    for word in words:
        sub_tokens = tokenizer.tokenize(word)
        if not sub_tokens:
            sub_tokens = [unk_tok]

        sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)

        if len(input_ids) + len(sub_ids) + 1 > max_len:
            break

        word_to_subword_start.append(len(input_ids))
        word_num_subtokens.append(len(sub_ids))
        input_ids.extend(sub_ids)

    input_ids.append(sep_id)
    attention_mask = [1] * len(input_ids)

    pad_len = max_len - len(input_ids)
    if pad_len > 0:
        input_ids += [pad_id] * pad_len
        attention_mask += [0] * pad_len

    return input_ids, attention_mask, word_to_subword_start, word_num_subtokens


def make_word_chunks(tokens: list[str], tokenizer, max_len: int = 256, overlap_words: int = 32):
    """Chia token dài thành các chunk vừa max_len subword, có overlap để
    entity nằm ngay biên chunk không bị cắt cụt."""
    budget = max_len - 2

    sub_lens = [max(1, len(tokenizer.tokenize(tok))) for tok in tokens]

    chunks = []
    n = len(tokens)
    start = 0

    while start < n:
        total = 0
        end = start
        while end < n and total + sub_lens[end] <= budget:
            total += sub_lens[end]
            end += 1
        if end == start:
            end = start + 1
        chunks.append((start, end))
        if end >= n:
            break
        start = max(start + 1, end - overlap_words)

    return chunks


def is_valid_char_span(offsets, ws: int, we: int) -> bool:
    """Require every word offset in the entity to be valid and monotonic."""
    if ws < 0 or we <= ws or we > len(offsets):
        return False
    spans = offsets[ws:we]
    for start, end in spans:
        if start is None or end is None or start < 0 or end <= start:
            return False
    for previous, current in zip(spans, spans[1:]):
        if current[0] < previous[1]:
            return False
    return True
