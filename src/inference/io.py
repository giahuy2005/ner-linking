"""Đọc input, remap offset về raw text và ghi output đúng cấu trúc BTC."""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Sequence, TypeAlias

from .schemas import NerEntity


# Opcode của difflib:
# (tag, raw_start, raw_end, cleaned_start, cleaned_end)
Opcode: TypeAlias = tuple[str, int, int, int, int]

# Chỉ các loại này cần ontology linking.
LINKING_TYPES = frozenset({
    "THUỐC",
    "CHẨN_ĐOÁN",
})

# Thứ tự trường giống output mẫu của BTC.
BTC_FIELD_ORDER = (
    "text",
    "type",
    "candidates",
    "assertions",
    "position",
)


def read_text_file(path: str | Path) -> str:
    """Đọc nguyên văn file, không tự đổi CRLF thành LF.

    newline="" rất quan trọng trên Windows. Nếu không truyền tham số này,
    Python mặc định đổi "\\r\\n" thành "\\n", khiến offset lệch thêm 1 ký tự
    sau mỗi lần xuống dòng.
    """
    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",  # giữ nguyên \r\n, \n hoặc \r
    ) as f:
        return f.read()


def build_raw_to_cleaned_map(
    raw_text: str,
    cleaned_text: str,
) -> list[Opcode]:
    """Tạo opcodes để map offset từ cleaned_text về raw_text.

    SequenceMatcher chỉ được chạy một lần cho mỗi record, thay vì chạy lại
    với từng entity.
    """
    matcher = difflib.SequenceMatcher(
        None,
        raw_text,
        cleaned_text,
        autojunk=False,
    )
    return matcher.get_opcodes()


def map_cleaned_span_to_raw(
    opcodes: Sequence[Opcode],
    cleaned_start: int,
    cleaned_end: int,
) -> tuple[int, int] | None:
    """Map span [cleaned_start, cleaned_end) về raw text.

    Quy ước offset là start-inclusive, end-exclusive.

    Trả về None nếu entity chỉ nằm trong phần được chèn thêm vào cleaned_text
    và hoàn toàn không có ký tự tương ứng trong raw_text.
    """
    if cleaned_start < 0:
        return None

    if cleaned_end <= cleaned_start:
        return None

    raw_start: int | None = None
    raw_end: int | None = None
    has_raw_content = False

    for tag, a1, a2, b1, b2 in opcodes:
        # Opcode không giao với entity.
        if b2 <= cleaned_start or b1 >= cleaned_end:
            continue

        overlap_start = max(cleaned_start, b1)
        overlap_end = min(cleaned_end, b2)

        if overlap_start >= overlap_end:
            continue

        if tag == "equal":
            # Hai đoạn có cùng độ dài và cùng nội dung.
            candidate_start = a1 + (overlap_start - b1)
            candidate_end = a1 + (overlap_end - b1)

            if raw_start is None:
                raw_start = candidate_start

            raw_end = candidate_end
            has_raw_content = True

        elif tag == "replace":
            # Nội suy trong đoạn bị thay thế.
            cleaned_block_length = b2 - b1
            raw_block_length = a2 - a1

            if cleaned_block_length <= 0:
                continue

            start_ratio = (overlap_start - b1) / cleaned_block_length
            end_ratio = (overlap_end - b1) / cleaned_block_length

            candidate_start = a1 + round(start_ratio * raw_block_length)
            candidate_end = a1 + round(end_ratio * raw_block_length)

            candidate_start = max(a1, min(candidate_start, a2))
            candidate_end = max(candidate_start, min(candidate_end, a2))

            if raw_start is None:
                raw_start = candidate_start

            raw_end = candidate_end

            if raw_block_length > 0:
                has_raw_content = True

        elif tag == "insert":
            # Đoạn tồn tại ở cleaned_text nhưng không tồn tại ở raw_text,
            # ví dụ khoảng trắng được chèn thêm.
            #
            # Dùng điểm a1 làm mốc để span vẫn có thể nối với các đoạn raw
            # nằm trước hoặc sau nó.
            if raw_start is None:
                raw_start = a1

            raw_end = a1

        # tag == "delete" có b1 == b2 nên không có ký tự cleaned giao với span.

    if not has_raw_content:
        return None

    if raw_start is None or raw_end is None:
        return None

    if raw_end < raw_start:
        return None

    return raw_start, raw_end


def _find_nearest_exact_span(
    raw_text: str,
    target_text: str,
    approximate_start: int,
    approximate_end: int,
    *,
    search_radius: int = 128,
) -> tuple[int, int] | None:
    """Tìm target_text gần span dự đoán để sửa các lỗi lệch 1-2 ký tự.

    Chỉ tìm exact match, không fuzzy match, nhằm tránh gán nhầm khi văn bản
    có nhiều entity giống nhau.
    """
    if not target_text:
        return None

    # Nếu span hiện tại đã đúng thì giữ nguyên.
    if raw_text[approximate_start:approximate_end] == target_text:
        return approximate_start, approximate_end

    window_start = max(0, approximate_start - search_radius)
    window_end = min(
        len(raw_text),
        approximate_end + search_radius + len(target_text),
    )

    matches: list[tuple[int, int]] = []
    search_from = window_start

    while True:
        found_start = raw_text.find(
            target_text,
            search_from,
            window_end,
        )

        if found_start == -1:
            break

        found_end = found_start + len(target_text)
        matches.append((found_start, found_end))
        search_from = found_start + 1

    if matches:
        return min(
            matches,
            key=lambda span: (
                abs(span[0] - approximate_start)
                + abs(span[1] - approximate_end)
            ),
        )

    # Nếu trong vùng gần không tìm thấy nhưng toàn văn chỉ có đúng một lần,
    # vẫn có thể dùng occurrence duy nhất đó.
    first_occurrence = raw_text.find(target_text)

    if first_occurrence == -1:
        return None

    second_occurrence = raw_text.find(
        target_text,
        first_occurrence + 1,
    )

    if second_occurrence == -1:
        return (
            first_occurrence,
            first_occurrence + len(target_text),
        )

    return None


def remap_entities_to_raw(
    raw_text: str,
    cleaned_text: str,
    entities: list[NerEntity],
    *,
    strict: bool = False,
    snap_to_exact_text: bool = True,
) -> list[NerEntity]:
    """Map lại text và position từ cleaned_text về raw_text.

    Args:
        raw_text:
            Nội dung nguyên bản, phải được đọc bằng read_text_file().
        cleaned_text:
            Nội dung sau preprocessing.
        entities:
            Entity có offset tính trên cleaned_text.
        strict:
            True: ném lỗi nếu entity không thể map.
            False: giữ entity cũ và in cảnh báo ra stderr.
        snap_to_exact_text:
            Tìm exact match gần offset đã map để sửa lệch nhỏ.

    Returns:
        Danh sách NerEntity có text và position tính trên raw_text.
    """
    if raw_text == cleaned_text:
        return entities

    opcodes = build_raw_to_cleaned_map(
        raw_text,
        cleaned_text,
    )

    remapped_entities: list[NerEntity] = []

    for entity_index, entity in enumerate(entities):
        cleaned_start = int(entity.position[0])
        cleaned_end = int(entity.position[1])

        if not (
            0 <= cleaned_start < cleaned_end <= len(cleaned_text)
        ):
            message = (
                f"Entity {entity_index} có offset cleaned không hợp lệ: "
                f"{entity.position}; cleaned length={len(cleaned_text)}; "
                f"text={entity.text!r}"
            )

            if strict:
                raise ValueError(message)

            print(f"[WARNING] {message}", file=sys.stderr)
            remapped_entities.append(entity)
            continue

        mapped_span = map_cleaned_span_to_raw(
            opcodes,
            cleaned_start,
            cleaned_end,
        )

        if mapped_span is None:
            message = (
                f"Không thể map entity {entity_index}: "
                f"position={entity.position}, text={entity.text!r}"
            )

            if strict:
                raise ValueError(message)

            print(f"[WARNING] {message}", file=sys.stderr)
            remapped_entities.append(entity)
            continue

        raw_start, raw_end = mapped_span

        raw_start = max(
            0,
            min(raw_start, len(raw_text)),
        )
        raw_end = max(
            raw_start,
            min(raw_end, len(raw_text)),
        )

        if snap_to_exact_text:
            cleaned_entity_text = cleaned_text[
                cleaned_start:cleaned_end
            ]

            exact_span = _find_nearest_exact_span(
                raw_text=raw_text,
                target_text=cleaned_entity_text,
                approximate_start=raw_start,
                approximate_end=raw_end,
            )

            # Nếu slice cleaned không khớp entity.text, thử entity.text.
            if exact_span is None and entity.text != cleaned_entity_text:
                exact_span = _find_nearest_exact_span(
                    raw_text=raw_text,
                    target_text=entity.text,
                    approximate_start=raw_start,
                    approximate_end=raw_end,
                )

            if exact_span is not None:
                raw_start, raw_end = exact_span

        remapped_entities.append(
            NerEntity(
                text=raw_text[raw_start:raw_end],
                type=entity.type,
                assertions=list(entity.assertions),
                position=(raw_start, raw_end),
                score=entity.score,
                flag=entity.flag,
            )
        )

    return remapped_entities


def _normalize_string_list(
    values: Sequence[object] | None,
) -> list[str]:
    """Đổi thành list[str] và loại phần tử trùng, giữ nguyên thứ tự."""
    if not values:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        normalized = str(value)

        if normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

    return result


def build_record_output(
    entities: list[NerEntity],
    candidates_by_entity: dict[int, list[str]] | None = None,
) -> list[dict]:
    """Chuyển list[NerEntity] thành output đúng cấu trúc BTC.

    candidates_by_entity có dạng:

        {
            0: ["308135"],
            1: ["243670"],
        }

    Chỉ THUỐC và CHẨN_ĐOÁN có trường candidates.
    """
    candidates_by_entity = candidates_by_entity or {}

    output: list[dict] = []

    for entity_index, entity in enumerate(entities):
        start = int(entity.position[0])
        end = int(entity.position[1])

        item: dict = {
            "text": str(entity.text),
            "type": str(entity.type),
        }

        if entity.type in LINKING_TYPES:
            item["candidates"] = _normalize_string_list(
                candidates_by_entity.get(entity_index, [])
            )

        item["assertions"] = _normalize_string_list(
            entity.assertions
        )
        item["position"] = [start, end]

        output.append(item)

    return output


def validate_record_output(
    record_output: list[dict],
    *,
    raw_text: str | None = None,
) -> None:
    """Kiểm tra output trước khi ghi file.

    Khi truyền raw_text, hàm còn kiểm tra:

        item["text"] == raw_text[start:end]

    Nhờ đó có thể phát hiện ngay lỗi offset do CRLF/LF.
    """
    if not isinstance(record_output, list):
        raise TypeError(
            "record_output phải là list[dict]."
        )

    required_fields = {
        "text",
        "type",
        "assertions",
        "position",
    }
    allowed_fields = {
        "text",
        "type",
        "candidates",
        "assertions",
        "position",
    }

    for entity_index, item in enumerate(record_output):
        if not isinstance(item, dict):
            raise TypeError(
                f"Entity {entity_index} phải là dict, "
                f"nhận được {type(item).__name__}."
            )

        missing_fields = required_fields - item.keys()
        if missing_fields:
            raise ValueError(
                f"Entity {entity_index} thiếu trường: "
                f"{sorted(missing_fields)}"
            )

        extra_fields = item.keys() - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Entity {entity_index} có trường không hợp lệ: "
                f"{sorted(extra_fields)}"
            )

        entity_text = item["text"]
        entity_type = item["type"]
        assertions = item["assertions"]
        position = item["position"]

        if not isinstance(entity_text, str):
            raise TypeError(
                f"Entity {entity_index}: text phải là str."
            )

        if not isinstance(entity_type, str):
            raise TypeError(
                f"Entity {entity_index}: type phải là str."
            )

        if not isinstance(assertions, list):
            raise TypeError(
                f"Entity {entity_index}: assertions phải là list."
            )

        if not all(
            isinstance(assertion, str)
            for assertion in assertions
        ):
            raise TypeError(
                f"Entity {entity_index}: mọi assertion phải là str."
            )

        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(
                f"Entity {entity_index}: position phải có dạng [start, end]."
            )

        start, end = position

        if type(start) is not int or type(end) is not int:
            raise TypeError(
                f"Entity {entity_index}: start/end phải là int."
            )

        if start < 0 or end < start:
            raise ValueError(
                f"Entity {entity_index}: offset không hợp lệ "
                f"{position}."
            )

        if entity_type in LINKING_TYPES:
            if "candidates" not in item:
                raise ValueError(
                    f"Entity {entity_index} loại {entity_type!r} "
                    "phải có trường candidates."
                )

            candidates = item["candidates"]

            if not isinstance(candidates, list):
                raise TypeError(
                    f"Entity {entity_index}: candidates phải là list."
                )

            if not all(
                isinstance(candidate, str)
                for candidate in candidates
            ):
                raise TypeError(
                    f"Entity {entity_index}: mọi candidate phải là str."
                )

            max_candidates = 1 if entity_type == "THUỐC" else 3
            if len(candidates) > max_candidates:
                raise ValueError(
                    f"Entity {entity_index} loại {entity_type!r} chỉ được có "
                    f"tối đa {max_candidates} candidate."
                )
            if len(candidates) != len(set(candidates)):
                raise ValueError(
                    f"Entity {entity_index}: candidates không được trùng nhau."
                )

        elif "candidates" in item:
            raise ValueError(
                f"Entity {entity_index} loại {entity_type!r} "
                "không được có trường candidates."
            )

        if raw_text is not None:
            if end > len(raw_text):
                raise ValueError(
                    f"Entity {entity_index}: end={end} vượt quá "
                    f"độ dài raw_text={len(raw_text)}."
                )

            raw_slice = raw_text[start:end]

            if entity_text != raw_slice:
                raise ValueError(
                    f"Entity {entity_index} sai text hoặc offset:\n"
                    f"  text output : {entity_text!r}\n"
                    f"  raw[{start}:{end}]: {raw_slice!r}\n"
                    f"  position    : {position}"
                )


def _dumps_btc_output(
    record_output: list[dict],
) -> str:
    """Format JSON giống mẫu BTC.

    Mỗi object được indent 2 spaces, trong khi candidates, assertions và
    position nằm trên một dòng.
    """
    if not record_output:
        return "[]"

    lines: list[str] = ["["]

    for item_index, item in enumerate(record_output):
        lines.append("  {")

        ordered_fields = [
            field
            for field in BTC_FIELD_ORDER
            if field in item
        ]

        for field_index, field in enumerate(ordered_fields):
            value = json.dumps(
                item[field],
                ensure_ascii=False,
            )

            comma = (
                ","
                if field_index < len(ordered_fields) - 1
                else ""
            )

            lines.append(
                f'    "{field}": {value}{comma}'
            )

        item_comma = (
            ","
            if item_index < len(record_output) - 1
            else ""
        )

        lines.append(f"  }}{item_comma}")

    lines.append("]")
    return "\n".join(lines)


def write_output_json(
    record_output: list[dict],
    path: str | Path,
    *,
    raw_text: str | None = None,
) -> None:
    """Validate và ghi output JSON theo format BTC."""
    validate_record_output(
        record_output,
        raw_text=raw_text,
    )

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = _dumps_btc_output(record_output)

    with open(
        path,
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(content)
        f.write("\n")


def _natural_path_sort_key(
    path: Path,
) -> tuple[tuple[int, int | str], ...]:
    """Sắp xếp 1.txt, 2.txt, 10.txt theo thứ tự số."""
    parts = re.split(r"(\d+)", path.stem)

    return tuple(
        (0, int(part))
        if part.isdigit()
        else (1, part.lower())
        for part in parts
        if part
    )


def run_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    process_fn: Callable[[str], list[dict]],
    *,
    pattern: str = "*.txt",
) -> dict[str, str]:
    """Chạy inference cho toàn bộ file input.

    process_fn nhận raw_text nguyên bản và phải trả về list[dict] theo
    cấu trúc BTC.

    Input:
        input/1.txt
        input/2.txt
        ...

    Output:
        output/1.json
        output/2.json
        ...

    Một file lỗi không làm dừng toàn bộ batch.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy input_dir: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"input_dir không phải thư mục: {input_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_files = sorted(
        (
            path
            for path in input_dir.glob(pattern)
            if path.is_file()
        ),
        key=_natural_path_sort_key,
    )

    status: dict[str, str] = {}

    for input_path in input_files:
        try:
            # Giữ nguyên CRLF để offset không bị lệch.
            raw_text = read_text_file(input_path)

            record_output = process_fn(raw_text)

            output_path = (
                output_dir
                / f"{input_path.stem}.json"
            )

            write_output_json(
                record_output,
                output_path,
                raw_text=raw_text,
            )

            status[input_path.name] = "ok"

        except Exception as exc:
            status[input_path.name] = (
                f"error: {type(exc).__name__}: {exc}"
            )

    return status
