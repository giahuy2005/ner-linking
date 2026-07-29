"""
JSONToBioConverter: chuẩn hóa dữ liệu NER tiếng Việt (entity theo text) sang
định dạng BIO word-level cho ViHealthBERT bản `-word`.

Yêu cầu: ViHealthBERT-word cần input đã word-segmented (RDRSegmenter/VnCoreNLP),
dạng "buồn_nôn" thay vì "buồn nôn". Class này:
  1. Word-segment câu bằng VnCoreNLP (RDRSegmenter - đúng segmenter dùng để
     pretrain ViHealthBERT, không dùng underthesea/pyvi để tránh lệch phân phối).
  2. Map lại char-offset của raw text sang từng word-token đã segment.
  3. Tìm char span của entity trên RAW TEXT, đảm bảo KHÔNG overlap giữa các
     entity khác nhau (fix lỗi "buồn nôn" nuốt luôn "nôn" đứng riêng).
  4. Tách dấu kết câu bị dính chữ thành token riêng trước khi gán BIO. Ví dụ
     `ngực.Hồi` trở thành `ngực`, `.`, `Hồi`, nên hai entity không tranh cùng token.
  5. Gán BIO tag theo overlap token<->entity và báo lỗi nếu một token vẫn bị hai
     entity tranh chấp; không âm thầm ghi đè gold.
  6. Suy ra token_start/token_end của assertion_spans TRỰC TIẾP từ chính
     bio_tags đã build ở bước 4 (thay vì tính lại độc lập bằng overlap char),
     để đảm bảo assertion_spans luôn khớp 100% với ner_tags -- kể cả khi
     2 entity liền kề nhau khiến 1 token bị "tranh chấp" giữa 2 entity.

Yêu cầu Java (JDK) đã cài, vì VnCoreNLP chạy qua JVM.

Cài đặt (chạy 1 lần, ví dụ trên Colab):
    !pip install vncorenlp -q
    !mkdir -p vncorenlp/models/wordsegmenter
    !wget -q -O vncorenlp/VnCoreNLP-1.1.1.jar \
        https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
    !wget -q -O vncorenlp/models/wordsegmenter/vi-vocab \
        https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
    !wget -q -O vncorenlp/models/wordsegmenter/wordsegmenter.rdr \
        https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

Sử dụng:
    from src.preprocessing.json_to_bio import JSONToBioConverter

    converter = JSONToBioConverter()
    converter.convert_file("train_1.jsonl", "train_hf_ner.jsonl")
    converter.close()

    # hoặc dùng context manager để tự đóng JVM:
    with JSONToBioConverter() as converter:
        converter.convert_file("train_1.jsonl", "train_hf_ner.jsonl")
"""

import re
import json
import unicodedata


_TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def _strip_tone(s: str) -> str:
    """
    Bỏ dấu THANH (sắc/huyền/hỏi/ngã/nặng), giữ nguyên các ký tự biến âm
    (ă, â, ê, ô, ơ, ư, đ). Dùng để so sánh 2 chuỗi bất chấp việc VnCoreNLP
    tự ý đổi vị trí dấu thanh trong nhóm nguyên âm đôi (vd "hóa" <-> "hoá",
    "thủy" <-> "thuỷ", "tỏa" <-> "toả") -- đây là 2 quy ước chính tả cũ/mới
    của tiếng Việt, cùng 1 từ, cùng phát âm, khác Unicode.
    """
    nfd = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in nfd if ch not in _TONE_MARKS)
    return unicodedata.normalize("NFC", stripped).lower()


class JSONToBioConverter:
    """
    Convert dữ liệu dạng {"input_text": ..., "entities": [{"text", "type",
    "assertions"}, ...]} sang {"tokens": [...], "ner_tags": [...],
    "assertion_spans": [{"token_start", "token_end", "type", "assertions"}]}.

    QUAN TRỌNG: VnCoreNLP giữ JVM sống suốt vòng đời object. Khởi tạo
    converter MỘT LẦN ở đầu batch/notebook, không tạo mới cho mỗi sample hay
    mỗi lần gọi convert_file (rất chậm vì mỗi lần khởi tạo là 1 lần start
    JVM). Gọi close() (hoặc dùng `with`) khi xử lý xong toàn bộ để giải
    phóng process JVM.
    """

    def __init__(
        self,
        jar_path: str = "vncorenlp/VnCoreNLP-1.1.1.jar",
        annotators: str = "wseg",
        max_heap_size: str = "-Xmx2g",
    ):
        from vncorenlp import VnCoreNLP

        self.rdr = VnCoreNLP(jar_path, annotators=annotators, max_heap_size=max_heap_size)

    # ------------------------------------------------------------------
    # 1. Word segmentation + map offset raw_text -> word-token
    # ------------------------------------------------------------------
    def segment_with_offsets(self, raw_text: str):
        """
        Trả về (word_tokens, offsets) với offsets là char span TRÊN RAW TEXT
        GỐC (chưa nối "_") cho từng word-token đã segment.

        Cách làm: RDRSegmenter chỉ NHÓM các syllable liền kề lại bằng "_",
        không đổi thứ tự syllable. Nhưng nó CÓ THỂ đổi chính tả dấu thanh của
        syllable (vd input "hóa" -> output "hoá") do dùng 2 quy ước chính tả
        cũ/mới khác nhau -- nên KHÔNG match bằng substring tuyệt đối.

        Thay vào đó: với mỗi syllable, chỉ bỏ qua khoảng trắng từ cursor rồi
        so sánh trực tiếp đoạn ký tự tiếp theo (không search xa, tránh false
        positive khi 1 syllable ngắn trùng ngẫu nhiên với 1 đoạn con nằm giữa
        từ khác, vd "hoá" trùng 3 ký tự giữa của "thoái") -- so sánh sau khi
        bỏ dấu thanh để chấp nhận sai khác chính tả cũ/mới.
        """
        sentences = self.rdr.tokenize(raw_text)
        word_tokens = [tok for sent in sentences for tok in sent]

        n = len(raw_text)
        offsets = []
        mapped_tokens = []
        cursor = 0
        for tok in word_tokens:
            # RDRSegmenter đôi khi trả `_Thống` ngay sau newline. Dấu `_` đầu/cuối
            # là artefact ghép từ, không tồn tại trong raw text; bỏ component rỗng để
            # tránh cố map syllable ''. Riêng token chỉ gồm `_` có thể là ký tự thật
            # trong raw, nên vẫn map nguyên token đó.
            mapped_tok = tok.strip("_") if tok.strip("_") else tok
            syllables = [part for part in mapped_tok.split("_") if part]
            if not syllables:
                syllables = [mapped_tok]
            tok_start, tok_end = None, None
            for syl in syllables:
                while cursor < n and raw_text[cursor].isspace():
                    cursor += 1

                target = _strip_tone(syl)
                matched = False
                # Ưu tiên đúng độ dài gốc; thử lệch +-1/+2 phòng hờ trường hợp
                # đổi chính tả hiếm khi làm lệch số ký tự.
                for length in (len(syl), len(syl) + 1, len(syl) - 1, len(syl) + 2):
                    if length <= 0 or cursor + length > n:
                        continue
                    candidate = raw_text[cursor:cursor + length]
                    if _strip_tone(candidate) == target:
                        if tok_start is None:
                            tok_start = cursor
                        cursor += length
                        tok_end = cursor
                        matched = True
                        break

                if not matched:
                    context = raw_text[cursor:cursor + 15]
                    raise ValueError(
                        f"Không map được syllable '{syl}' (thuộc token '{tok}') "
                        f"tại vị trí {cursor}. raw_text ở đó là: '{context}...'. "
                        f"Kiểm tra lại raw_text có ký tự đặc biệt bất thường không."
                    )
            offsets.append((tok_start, tok_end))
            mapped_tokens.append(mapped_tok)

        return self._split_stuck_sentence_punctuation(raw_text, mapped_tokens, offsets)

    @staticmethod
    def _split_stuck_sentence_punctuation(raw_text: str, tokens: list, offsets: list):
        """Tách `.?!` đang dính chữ ở hai phía mà vẫn giữ offset trên raw text.

        RDRSegmenter đôi khi trả một token như ``ngực.Hồi`` khi nguồn thiếu khoảng
        trắng. Một token BIO không thể đồng thời là phần cuối của entity thứ nhất và
        phần đầu của entity thứ hai. Vì vậy chỉ tại các ranh giới ``chữ.dính`` ta tách
        token thành phần trái, dấu câu và phần phải. Các từ ghép VnCoreNLP bình thường
        vẫn giữ nguyên.
        """
        split_tokens = []
        split_offsets = []

        for token, (start, end) in zip(tokens, offsets):
            raw_piece = raw_text[start:end]
            # Dấu kết câu chỉ tách khi hai phía là CHỮ nên không phá ``5.2``/``37.8``.
            # Dấu :; và dấu phẩy ghi nhanh cũng tách khi dính, nhưng dấu phẩy thập phân
            # ``12,5`` vẫn được giữ nguyên.
            boundaries = list(
                re.finditer(
                    r"(?<=[^\W\d_])[.!?](?=[^\W\d_])|"
                    r"(?<=\S)[:;](?=\S)|"
                    r"(?<=\S),(?=[^\s\d])|(?<=[^\d\s]),(?=\S)",
                    raw_piece,
                )
            )
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

    # ------------------------------------------------------------------
    # 2. Tìm char span của entity trên raw text, không overlap giữa các entity
    # ------------------------------------------------------------------
    @staticmethod
    def find_entity_char_spans(input_text: str, entities: list):
        """
        Duyệt mọi occurrence của entity["text"] bằng finditer, chọn occurrence
        đầu tiên KHÔNG đè lên bất kỳ span nào đã dùng trước đó -- kể cả khi 2
        entity có text khác nhau nhưng 1 cái là substring của cái kia
        (vd "nôn" nằm trong "buồn nôn").

        Lưu ý: thứ tự entity trong list ảnh hưởng tới việc span nào được ưu
        tiên trước -- nên xử lý entity dài/cụ thể hơn trước nếu data có
        nhiều trường hợp lồng nhau phức tạp hơn.
        """
        used_spans = []
        spans = []
        skipped = []
        for ent in entities:
            chosen = None
            for m in re.finditer(re.escape(ent["text"]), input_text):
                s, e = m.start(), m.end()
                if all(e <= us or s >= ue for us, ue in used_spans):
                    chosen = (s, e)
                    break
            if chosen is None:
                skipped.append(ent["text"])
                continue
            s, e = chosen
            used_spans.append((s, e))
            spans.append({**ent, "char_start": s, "char_end": e})
        return spans, skipped

    @staticmethod
    def resolve_entity_char_spans(input_text: str, entities: list, explicit_spans=None):
        """Prefer generator-provided offsets and strictly validate alignment.

        Legacy files without ``entity_spans`` still use text matching. New data
        must not lose occurrence identity between generation/QC and BIO export.
        """
        if explicit_spans is None:
            return JSONToBioConverter.find_entity_char_spans(input_text, entities)
        if not isinstance(explicit_spans, list) or len(explicit_spans) != len(entities):
            raise ValueError("entity_spans phải là list cùng độ dài với entities")

        resolved = []
        previous_end = -1
        for index, (entity, span) in enumerate(zip(entities, explicit_spans)):
            if not isinstance(span, dict):
                raise ValueError(f"entity_spans[{index}] không phải object")
            start = span.get("char_start")
            end = span.get("char_end")
            if not isinstance(start, int) or isinstance(start, bool):
                raise ValueError(f"entity_spans[{index}].char_start không phải int")
            if not isinstance(end, int) or isinstance(end, bool):
                raise ValueError(f"entity_spans[{index}].char_end không phải int")
            if not (0 <= start < end <= len(input_text)):
                raise ValueError(f"entity_spans[{index}] ngoài phạm vi input_text: {(start, end)}")
            if start < previous_end:
                raise ValueError(f"entity_spans[{index}] overlap hoặc không theo thứ tự")
            if input_text[start:end] != entity["text"]:
                raise ValueError(
                    f"entity_spans[{index}] không khớp text: "
                    f"{input_text[start:end]!r} != {entity['text']!r}"
                )
            if span.get("text", entity["text"]) != entity["text"]:
                raise ValueError(f"entity_spans[{index}].text lệch entities[{index}].text")
            if span.get("type", entity["type"]) != entity["type"]:
                raise ValueError(f"entity_spans[{index}].type lệch entities[{index}].type")
            resolved.append({**entity, "char_start": start, "char_end": end})
            previous_end = end
        return resolved, []

    # ------------------------------------------------------------------
    # 3. Map char span -> word-token index -> BIO
    # ------------------------------------------------------------------
    @staticmethod
    def build_bio_tags(tokens: list, offsets: list, entity_spans: list):
        """
        Dùng OVERLAP (tok_start < char_end AND tok_end > char_start), không
        dùng match tuyệt đối, vì dấu câu dính liền token cuối có thể làm
        char_end lệch.

        Nếu sau bước tách dấu câu mà một token vẫn overlap hai entity, sample không
        biểu diễn được bằng BIO word-level. Báo lỗi để caller skip/log sample thay vì
        ghi đè một entity và tạo gold sai.
        """
        tags = ["O"] * len(tokens)
        for ent in entity_spans:
            first_token = True
            for i, (tok_start, tok_end) in enumerate(offsets):
                if tok_start < ent["char_end"] and tok_end > ent["char_start"]:
                    if tags[i] != "O":
                        raise ValueError(
                            "Một token BIO overlap nhiều entity: "
                            f"token={tokens[i]!r}, tag_cũ={tags[i]!r}, "
                            f"entity_mới={ent['text']!r}/{ent['type']}. "
                            "Cần tách token hoặc sửa ranh giới entity."
                        )
                    tags[i] = ("B-" if first_token else "I-") + ent["type"]
                    first_token = False
        return tags

    # ------------------------------------------------------------------
    # 4. Suy assertion_spans TRỰC TIẾP từ bio_tags (nguồn chân lý duy nhất)
    # ------------------------------------------------------------------
    @staticmethod
    def _entity_ranges_from_bio(bio_tags: list) -> dict:
        """Quét lại bio_tags đã build, trả về {(token_start, token_end): entity_type}.

        Đây là NGUỒN DUY NHẤT xác định ranh giới thật của mỗi entity sau khi
        đã xử lý ghi đè -- khớp 100% với những gì mô hình NER sẽ thực sự học,
        vì chính bio_tags này được dùng làm label huấn luyện.
        """
        ranges = {}
        i, n = 0, len(bio_tags)
        while i < n:
            tag = bio_tags[i]
            if tag.startswith("B-"):
                ent_type = tag[2:]
                start = i
                j = i + 1
                while j < n and bio_tags[j] == f"I-{ent_type}":
                    j += 1
                ranges[(start, j)] = ent_type
                i = j
            else:
                i += 1
        return ranges

    @classmethod
    def token_range_for_span(cls, offsets: list, char_start: int, char_end: int):
        """Giữ lại để tương thích ngược / debug thủ công. token_end EXCLUSIVE.

        KHÔNG còn được dùng để build assertion_spans chính thức (xem
        process_sample) vì cách tính overlap-char độc lập này có thể lệch so
        với ner_tags khi 2 entity liền kề tranh chấp 1 token. Dùng
        `_entity_ranges_from_bio` làm nguồn chính thức thay thế.
        """
        idxs = [i for i, (s, e) in enumerate(offsets) if s < char_end and e > char_start]
        return idxs[0], idxs[-1] + 1

    def _match_entities_to_bio_ranges(self, entity_spans: list, offsets: list, bio_tags: list):
        """
        Với mỗi entity gốc (theo char span + type), tìm range token thực sự
        thuộc entity đó SAU KHI đã build xong bio_tags (tức là sau khi mọi
        tranh chấp ghi đè giữa các entity liền kề đã được giải quyết).

        Match theo: cùng `type`, còn "trống" (chưa bị entity khác trong cùng
        sample nhận), và có overlap char lớn nhất với entity gốc. Trả về
        (assertion_spans, dropped) -- dropped là các entity bị 1 entity khác
        ghi đè hoàn toàn (không còn B-/I- nào sau khi build_bio_tags), cần
        log lại để soát thủ công thay vì âm thầm mất.
        """
        available = self._entity_ranges_from_bio(bio_tags)  # {(s,e): type}
        assertion_spans = []
        dropped = []

        for ent in entity_spans:
            candidates = []
            for (s, e), t in available.items():
                if t != ent["type"]:
                    continue
                tok_char_start, tok_char_end = offsets[s][0], offsets[e - 1][1]
                overlap = min(tok_char_end, ent["char_end"]) - max(tok_char_start, ent["char_start"])
                if overlap > 0:
                    candidates.append(((s, e), overlap))

            if not candidates:
                dropped.append(ent)
                continue

            # entity có overlap lớn nhất thắng; nếu bằng nhau, ưu tiên range
            # đứng trước (start nhỏ hơn) để kết quả ổn định, dễ debug
            best_range, _ = max(candidates, key=lambda c: (c[1], -c[0][0]))
            available.pop(best_range)  # dùng rồi thì bỏ, tránh 2 entity match cùng 1 range

            s, e = best_range
            assertion_spans.append({
                "token_start": s,
                "token_end": e,
                "type": ent["type"],
                "assertions": ent["assertions"],
            })

        return assertion_spans, dropped

    # ------------------------------------------------------------------
    # Xử lý 1 sample
    # ------------------------------------------------------------------
    def process_sample(self, sample: dict) -> dict:
        input_text = sample["input_text"]
        tokens, offsets = self.segment_with_offsets(input_text)
        entity_spans, skipped = self.resolve_entity_char_spans(
            input_text,
            sample["entities"],
            sample.get("entity_spans"),
        )
        bio_tags = self.build_bio_tags(tokens, offsets, entity_spans)

        assertion_spans, dropped = self._match_entities_to_bio_ranges(entity_spans, offsets, bio_tags)

        if dropped:
            skipped = skipped + [
                f"dropped(bị entity khác ghi đè hoàn toàn trong bio_tags): {d['text']}"
                for d in dropped
            ]

        result = {
            "tokens": tokens,
            "ner_tags": bio_tags,
            "assertion_spans": assertion_spans,
        }
        return result, skipped

    # ------------------------------------------------------------------
    # Xử lý cả file jsonl -> jsonl
    # ------------------------------------------------------------------
    def convert_file(self, input_path: str, output_path: str, verbose: bool = True) -> dict:
        """
        Đọc từng dòng jsonl input, convert, ghi từng dòng ra output_path.
        Sample nào lỗi (segment lỗi, không map được token) sẽ bị SKIP toàn bộ
        sample đó (không ghi ra output) và log lại, không làm crash cả batch.

        Trả về dict thống kê: {"total", "converted", "sample_failed",
        "entities_skipped"} để bạn kiểm tra chất lượng convert ngay sau khi chạy.
        """
        stats = {"total": 0, "converted": 0, "sample_failed": 0, "entities_skipped": 0}

        with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line_no, line in enumerate(fin):
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                try:
                    sample = json.loads(line)
                    result, skipped = self.process_sample(sample)
                except Exception as e:
                    stats["sample_failed"] += 1
                    if verbose:
                        print(f"[!] Bỏ qua dòng {line_no + 1}: {e}")
                    continue

                if skipped:
                    stats["entities_skipped"] += len(skipped)
                    if verbose:
                        print(f"[!] Dòng {line_no}: {len(skipped)} entity không map được span -> {skipped}")

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                stats["converted"] += 1

        if verbose:
            print()
            print("=== Tổng kết convert ===")
            print(f"  Tổng số dòng input:      {stats['total']}")
            print(f"  Convert thành công:      {stats['converted']}")
            print(f"  Sample bị bỏ qua hoàn toàn: {stats['sample_failed']}")
            print(f"  Entity bị bỏ qua (nhưng sample vẫn giữ): {stats['entities_skipped']}")

        return stats

    # ------------------------------------------------------------------
    # Dọn dẹp JVM
    # ------------------------------------------------------------------
    def close(self):
        self.rdr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.preprocess_json_to_bio import main

    main()
