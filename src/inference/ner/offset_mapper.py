"""Word-segment + char-offset mapping + chunking cho input dài.

Port nguyên logic từ predict_ner_crf_final.ipynb (đã test qua các case
bẩn thật trong data). KHÔNG đổi thuật toán match syllable ở đây nếu
chưa test lại trên cùng bộ .txt — offset lệch 1 ký tự sẽ làm sai cả
position lẫn text trả về, và WER/candidates score BTC chấm sẽ tụt theo.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}

# Chỉ tìm lại token trong một vùng ngắn phía trước. Mục đích là phục hồi
# sau một token OCR/lạ, không được phép nhảy tùy ý tới một occurrence xa.
_RESYNC_MAX_CHARS = 128

_STUCK_PUNCT_RE = re.compile(
    r"(?<=[^\W\d_])[.!?](?=[^\W\d_])|"
    r"(?<=\S)[:;](?=\S)|"
    r"(?<=\S),(?=[^\s\d])|(?<=[^\d\s]),(?=\S)"
)


def _strip_tone(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in nfd if ch not in _TONE_MARKS)
    return unicodedata.normalize("NFC", stripped).lower()


def _try_match_token_from(
    text: str,
    start_cursor: int,
    syllables: list[str],
) -> tuple[tuple[int, int, int] | None, int]:
    """Thử map toàn bộ token từ ``start_cursor`` mà không sửa cursor bên ngoài.

    Trả về:
    - ``((token_start, token_end, next_cursor), matched_syllables)`` nếu thành công;
    - ``(None, matched_syllables)`` nếu thất bại.

    ``matched_syllables`` giúp phân biệt hai trường hợp:
    1. chưa match được syllable nào: cursor có thể đang lệch và được phép resync;
    2. đã match một phần token rồi mới fail: phải rollback, không được nhảy tới
       một occurrence cùng surface ở xa vì có thể gắn sai token hiện tại.
    """
    n = len(text)
    local_cursor = start_cursor
    tok_start: int | None = None
    tok_end: int | None = None
    matched_syllables = 0

    for syl in syllables:
        while local_cursor < n and text[local_cursor].isspace():
            local_cursor += 1

        target = _strip_tone(syl)
        matched = False

        # Giữ nguyên thứ tự/logic length tolerance của code cũ.
        for length in (len(syl), len(syl) + 1, len(syl) - 1, len(syl) + 2):
            if length <= 0 or local_cursor + length > n:
                continue

            candidate = text[local_cursor:local_cursor + length]
            if _strip_tone(candidate) == target:
                if tok_start is None:
                    tok_start = local_cursor
                local_cursor += length
                tok_end = local_cursor
                matched_syllables += 1
                matched = True
                break

        if not matched:
            return None, matched_syllables

    if tok_start is None or tok_end is None:
        return None, matched_syllables

    return (tok_start, tok_end, local_cursor), matched_syllables


def _is_resync_boundary(text: str, position: int) -> bool:
    """Chỉ bắt đầu resync ở đầu text hoặc sau whitespace/dấu câu.

    Điều này giảm nguy cơ map nhầm một token vào giữa một từ dài hơn.
    """
    if position <= 0:
        return True

    previous = text[position - 1]
    return previous.isspace() or (not previous.isalnum() and previous != "_")


def _bounded_resync(
    text: str,
    start_cursor: int,
    syllables: list[str],
    max_chars: int = _RESYNC_MAX_CHARS,
) -> tuple[int, int, int] | None:
    """Tìm lại token trong cửa sổ ngắn phía trước bằng đúng matcher hiện tại.

    Hàm chỉ được gọi khi lần match tuần tự chưa khớp syllable nào. Nếu token đã
    match một phần rồi fail, caller phải rollback và không resync token đó.
    """
    n = len(text)
    if start_cursor >= n or max_chars <= 0:
        return None

    search_stop = min(n, start_cursor + max_chars)
    probe = start_cursor + 1  # vị trí start_cursor vừa được thử ở fast path

    while probe < search_stop:
        while probe < search_stop and text[probe].isspace():
            probe += 1
        if probe >= search_stop:
            break

        if _is_resync_boundary(text, probe):
            result, _ = _try_match_token_from(text, probe, syllables)
            if result is not None:
                return result

        probe += 1

    return None


def segment_with_offsets(
    text: str,
    rdr,
) -> tuple[list[str], list[tuple[int | None, int | None]], list[int]]:
    """Tokenize bằng VnCoreNLP rồi map lại char offset trên ``text`` gốc.

    Trả về ``(tokens, offsets, line_ids)``. ``offsets[i] = (None, None)`` nếu
    token không map được (KHÔNG raise — một token lạ không được crash cả
    request lúc inference); entity chứa token này bị ``is_valid_char_span()``
    loại ở bước sau.

    Invariant quan trọng: nếu một token map thất bại, cursor luôn rollback về
    vị trí trước token. Một bounded resync chỉ được dùng khi token chưa match
    được syllable nào, nhằm tránh một lỗi OCR làm lệch toàn bộ token phía sau.
    """
    # VnCoreNLP may join syllables on adjacent physical lines into one
    # underscore token when the whole block is submitted at once. Such a token
    # has a character span containing ``\n`` and can make BIO extraction create
    # an illegal cross-line entity. Tokenize each physical line independently;
    # the global offset mapper below still maps every token against ``text``.
    word_tokens: list[str] = []
    for physical_line in text.splitlines():
        if not physical_line.strip():
            continue
        sentences = rdr.tokenize(physical_line)
        word_tokens.extend(tok for sent in sentences for tok in sent)

    offsets: list[tuple[int | None, int | None]] = []
    mapped_tokens: list[str] = []
    cursor = 0

    for tok in word_tokens:
        mapped_tok = tok.strip("_") if tok.strip("_") else tok
        syllables = [part for part in mapped_tok.split("_") if part]
        if not syllables:
            syllables = [mapped_tok]

        # PHẦN FIX QUAN TRỌNG:
        # Cursor chỉ được commit khi TOÀN BỘ token map thành công.
        token_cursor_start = cursor
        match_result, matched_syllables = _try_match_token_from(
            text=text,
            start_cursor=token_cursor_start,
            syllables=syllables,
        )

        resync_attempted = False
        resync_result: tuple[int, int, int] | None = None

        if match_result is None:
            # Rollback bắt buộc. Code cũ đã để cursor tiến sau các syllable
            # map được trước khi một syllable sau thất bại.
            cursor = token_cursor_start

            # Chỉ resync khi chưa match được phần nào của token. Nếu đã match
            # một phần, việc tìm occurrence tiếp theo có thể gắn token hiện tại
            # vào một mention lặp ở xa và làm sai alignment.
            if matched_syllables == 0:
                resync_attempted = True
                resync_result = _bounded_resync(
                    text=text,
                    start_cursor=token_cursor_start,
                    syllables=syllables,
                )

            if resync_result is not None:
                tok_start, tok_end, cursor = resync_result
                logger.debug(
                    "Offset mapper resynced token=%r from cursor=%d to span=(%d, %d)",
                    mapped_tok,
                    token_cursor_start,
                    tok_start,
                    tok_end,
                )
            else:
                tok_start, tok_end = None, None
                logger.warning(
                    "Offset mapping failed for token=%r at cursor=%d; "
                    "matched_syllables=%d/%d; cursor rolled back; "
                    "bounded_resync_attempted=%s",
                    mapped_tok,
                    token_cursor_start,
                    matched_syllables,
                    len(syllables),
                    resync_attempted,
                )
        else:
            tok_start, tok_end, cursor = match_result

        offsets.append((tok_start, tok_end))
        mapped_tokens.append(mapped_tok)

    tokens, offsets = _split_stuck_sentence_punctuation(
        text,
        mapped_tokens,
        offsets,
    )
    line_ids = [
        text.count("\n", 0, start) if start is not None else -1
        for start, _ in offsets
    ]
    return tokens, offsets, line_ids


def _split_stuck_sentence_punctuation(raw_text, tokens, offsets):
    """Tách dấu câu kết câu dính chữ (vd ``'...áp.Rối loạn...'``) ra token
    riêng, để hai entity liền kề không tranh nhau một token BIO.
    """
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
    """Subword-encode danh sách word token, trả ``word_to_subword_start`` để
    map ngược tag CRF (predict per-subword) về word-level.
    """
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


def make_word_chunks(
    tokens: list[str],
    tokenizer,
    max_len: int = 256,
    overlap_words: int = 32,
):
    """Chia token dài thành các chunk vừa ``max_len`` subword, có overlap để
    entity nằm ngay biên chunk không bị cắt cụt.
    """
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
