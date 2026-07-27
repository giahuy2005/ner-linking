#!/usr/bin/env python3
"""
src/preprocessing/icd10/merge_icd.py

Chạy sau parse_xml.py. Đọc:
  - data/processed/icd10_clean.jsonl   (output của parse_xml.py)
  - data/raw/icd10/vi_icd10.xlsx       (danh mục ICD-10 tiếng Việt, QĐ 4469/TT06)

Ghi ra:
  - data/processed/icd10_merged.jsonl

Cách chạy:
  1) Chỉ join bản dịch chính thức, KHÔNG gọi LLM (nhanh, không cần API key):
       python src/preprocessing/icd10/merge_icd.py --skip-llm

  2) Join + dịch nốt phần còn thiếu bằng OpenRouter:
       export OPENROUTER_API_KEY="sk-or-..."
       python src/preprocessing/icd10/merge_icd.py

  3) Đổi path nếu cần:
       python src/preprocessing/icd10/merge_icd.py --xlsx data/raw/icd10/vi_icd10_v2.xlsx

Chạy lại nhiều lần AN TOÀN: bước join luôn re-derive từ xlsx (không tốn
API), bước LLM có checkpoint (data/processed/translate_progress.json) và
tự bỏ qua record đã có preferred_vi/aliases_vi hợp lệ.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from icd10_translation_qc import qc_aliases, qc_preferred
load_dotenv()
# src/preprocessing/icd10/merge_icd.py -> parents[3] = thư mục gốc repo
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEAN_JSONL = PROJECT_ROOT / "data" / "processed" /"icd10"/ "icd10_clean.jsonl"
DEFAULT_XLSX = PROJECT_ROOT / "data" / "raw" / "icd10" / "vi_icd10.xlsx"
DEFAULT_OUTPUT_JSONL = PROJECT_ROOT / "data" / "processed" /"icd10"/ "icd10_merged.jsonl"
DEFAULT_PROGRESS_FILE = PROJECT_ROOT / "data" / "processed"  /"icd10"/ "translate_progress.json"
DEFAULT_MANUAL_ALIASES = PROJECT_ROOT / "data" / "raw" / "icd10" / "manual_aliases_vi.json"
DEFAULT_REVIEW_LIVE = PROJECT_ROOT / "data" / "processed" / "icd10" / "icd10_review_manifest_live.jsonl"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("GEN_MODEL", "deepseek/deepseek-chat")
BATCH_SIZE = 20
MAX_RETRIES = 5
MAX_VALIDATION_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.0

MARK_RE = re.compile(r"[†\*]")
CODE_TOKEN = r"[A-Z]\d{2}(?:\.[0-9A-Z]+)?[†\*]?"
REF_SUFFIX_RE = re.compile(rf"\s*\(\s*{CODE_TOKEN}(?:\s*[,\-]\s*{CODE_TOKEN})*\s*\)\s*$")

SYSTEM_PROMPT_PREFERRED = """
Bạn là chuyên gia dịch thuật ngữ ICD-10 Anh-Việt.

Dịch tên bệnh tiếng Anh sang tên bệnh tiếng Việt chuẩn, ngắn gọn.
Không thêm hoặc bỏ các thuộc tính quan trọng như có/không, cấp/mạn,
týp, vị trí, bên trái/phải, nguyên phát/thứ phát, xác định/không xác định.
Giữ nguyên code.
Chỉ trả về JSON hợp lệ, không dùng markdown.
"""

SYSTEM_PROMPT_ALIASES = """
Bạn là chuyên gia chuẩn hóa thuật ngữ y khoa Anh-Việt theo ICD-10.

Quy tắc:
- preferred_vi được cung cấp là tên chuẩn, KHÔNG được sửa hoặc dịch lại.
- Dịch aliases_en sao cho nhất quán với preferred_vi.
- Không thêm hoặc làm mất các thuộc tính có ý nghĩa phân loại như:
  có/không, cấp/mạn, týp 1/týp 2, trái/phải,
  nguyên phát/thứ phát, xác định/không xác định.
- Không thêm bệnh, nguyên nhân, vị trí hoặc biến chứng không có trong bản gốc.
- Mỗi alias đầu ra phải tương ứng đúng với alias đầu vào ở cùng vị trí.
- Alias phải là tên bệnh ngắn gọn, không phải định nghĩa hay lời giải thích.
- Chỉ trả về JSON hợp lệ, không dùng markdown.
-Không được loại bỏ hoặc gộp các alias gần nghĩa nhau.
Ví dụ catarrhal, exudative, secretory, serous và transudative
phải được dịch thành các phần tử riêng biệt.
"""


# ---------- Bước 1: join bản dịch chính thức từ xlsx ----------

def clean_code(c):
    return MARK_RE.sub("", str(c)).strip()


def load_vn_map(xlsx_path: Path):
    df = pd.read_excel(xlsx_path, sheet_name="ICD10", header=2)
    df["code_clean"] = df["MÃ BỆNH"].apply(clean_code)
    df["has_mark"] = df["MÃ BỆNH"].astype(str).str.contains(r"[†\*]")
    df = df.drop_duplicates(subset=["code_clean", "TÊN BỆNH"])
    df = df.sort_values("has_mark")  # ưu tiên dòng không dấu † / *
    df = df.drop_duplicates(subset=["code_clean"], keep="first")

    vn_map = {}
    for _, row in df.iterrows():
        code = row["code_clean"]
        raw = str(row["TÊN BỆNH"]).strip()
        clean = REF_SUFFIX_RE.sub("", raw).strip()
        vn_map[code] = (clean, raw if raw != clean else None)
    return vn_map


def join_vn_official(records, xlsx_path: Path):
    vn_map = load_vn_map(xlsx_path)
    n_matched = 0
    for rec in records:
        code = rec["code"]
        if code in vn_map:
            clean, raw = vn_map[code]
            rec["preferred_vi"] = clean
            if raw:
                rec["preferred_vi_raw"] = raw
            else:
                rec.pop("preferred_vi_raw", None)
            rec["preferred_vi_source"] = "byt_official"
            n_matched += 1
    print(f"Join VN chính thức: {n_matched}/{len(records)} record có preferred_vi")
    return records


# ---------- Bước 2: dịch phần còn thiếu qua OpenRouter ----------

def has_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_unique_codes(records):
    seen, duplicated = set(), set()
    for r in records:
        if r["code"] in seen:
            duplicated.add(r["code"])
        seen.add(r["code"])
    if duplicated:
        raise ValueError(f"Phát hiện code bị trùng trong input: {sorted(duplicated)}")


def load_progress(progress_file: Path):
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            return json.load(f)
    return {"preferred_vi": {}, "aliases_vi": {}}


def save_progress(progress, progress_file: Path):
    temp_path = progress_file.with_suffix(progress_file.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, progress_file)


def append_jsonl(path: Path, items):
    """Append review-flag items to a live-growing jsonl so Ghuy can `tail -f`
    it while the (potentially hours-long) LLM translation loop is running,
    instead of waiting for a full post-hoc pass to see problems."""
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def call_openrouter(session, api_key, user_prompt, system_prompt):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.post(OPENROUTER_URL, headers=headers, json=body, timeout=(20, 180))
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            return json.loads(text)
        except Exception as e:
            print(f"  [lỗi lần {attempt}/{MAX_RETRIES}] {e}", file=sys.stderr)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(min(60, (2 ** attempt) + random.uniform(0, 1)))


def validate_preferred_result(batch, result):
    if not isinstance(result, dict):
        raise ValueError("Kết quả preferred không phải object JSON")
    expected = {r["code"] for r in batch}
    returned = set(result)
    if returned != expected:
        raise ValueError(f"Sai danh sách code. Thiếu={expected - returned}, thừa={returned - expected}")
    cleaned = {}
    for r in batch:
        v = result[r["code"]]
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"preferred_vi không hợp lệ tại {r['code']}: {v!r}")
        cleaned[r["code"]] = v.strip()
    return cleaned


def validate_aliases_result(batch, result):
    if not isinstance(result, dict):
        raise ValueError("Kết quả alias không phải object JSON")
    expected = {r["code"] for r in batch}
    returned = set(result)
    if returned != expected:
        raise ValueError(f"Sai danh sách code. Thiếu={expected - returned}, thừa={returned - expected}")
    cleaned = {}
    for r in batch:
        code = r["code"]
        src = r["aliases_en"]
        out = result[code]
        if not isinstance(out, list):
            raise ValueError(f"aliases_vi của {code} không phải list")
        if len(out) != len(src):
            print(f"\n[DEBUG] Code lỗi: {code}")
            print("aliases_en:")
            for i, text in enumerate(src):
                print(f"  EN[{i}]: {text}")

            print("aliases_vi:")
            for i, text in enumerate(out):
                print(f"  VI[{i}]: {text}")
            raise ValueError(f"{code}: aliases_en có {len(src)} phần tử nhưng aliases_vi có {len(out)}")
        normalized = []
        for alias in out:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(f"Alias không hợp lệ tại {code}: {alias!r}")
            normalized.append(alias.strip())
        cleaned[code] = normalized
    return cleaned


def partition_aliases_result(batch, result):
    """Giữ các code hợp lệ và chỉ trả lại các code cần dịch lại.

    LLM đôi khi gộp hai alias gần nghĩa và làm một code bị thiếu phần tử. Không
    nên vì một code như vậy mà bỏ toàn bộ kết quả hợp lệ của batch.
    """
    if not isinstance(result, dict):
        print("  [WARN] Kết quả alias không phải object JSON; sẽ dịch lại cả batch")
        return {}, list(batch)

    cleaned = {}
    retry = []
    for r in batch:
        code = r["code"]
        if code not in result:
            print(f"  [WARN] Kết quả thiếu code {code}; sẽ dịch lại riêng")
            retry.append(r)
            continue
        try:
            # Validate từng code để một code lỗi không làm mất 19 code đúng.
            cleaned.update(validate_aliases_result([r], {code: result[code]}))
        except ValueError as exc:
            print(f"  [WARN] {exc}; sẽ dịch lại riêng")
            retry.append(r)

    extra = set(result) - {r["code"] for r in batch}
    if extra:
        print(f"  [WARN] Bỏ qua code thừa trong kết quả: {sorted(extra)}")
    return cleaned, retry


def translate_preferred_batch(session, api_key, batch):
    items = [{"code": r["code"], "en": r["preferred_en"]} for r in batch]
    prompt = (
        "Dịch trường \"en\" (tên bệnh ICD-10 tiếng Anh) sang tiếng Việt cho từng "
        "item dưới đây. Trả về JSON là 1 object, key là \"code\", value là bản "
        "dịch tiếng Việt (string).\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output format: {\"<code>\": \"<bản dịch>\", ...}"
    )
    return call_openrouter(session, api_key, prompt, SYSTEM_PROMPT_PREFERRED)


def translate_aliases_batch(session, api_key, batch):
    items = [
        {
            "code": r["code"],
            "preferred_en": r.get("preferred_en", ""),
            "preferred_vi": r.get("preferred_vi", ""),
            "aliases_en": r["aliases_en"],
        }
        for r in batch
    ]
    prompt = (
        "Dịch từng chuỗi trong \"aliases_en\" sang tiếng Việt, dùng \"preferred_vi\" "
        "(KHÔNG được sửa/dịch lại field này) làm ngữ cảnh để thuật ngữ nhất quán, và "
        "\"preferred_en\" để hiểu rõ khái niệm khi alias tiếng Anh quá ngắn/mơ hồ. "
        "Giữ đúng thứ tự, số lượng phần tử trong mảng dịch phải khớp với aliases_en gốc.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Output format: {\"<code>\": [\"<alias dịch 1>\", \"<alias dịch 2>\", ...], ...}"
    )
    return call_openrouter(session, api_key, prompt, SYSTEM_PROMPT_ALIASES)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def migrate_legacy_aliases_checkpoint(progress, records_by_code, review_live_path: Path):
    """
    Checkpoint cũ (trước khi có QC-in-loop) lưu progress["aliases_vi"][code]
    là list[str] thô, chưa lọc coding_note / chưa sửa term. Hàm này chạy 1
    lần khi load progress: phát hiện entry dạng cũ và chạy qc_aliases() lên
    ngay, không cần gọi lại LLM. Sau khi migrate xong, ghi lại checkpoint ở
    format mới (dict) để lần chạy sau không phải migrate lại.
    """
    legacy_codes = [
        code for code, val in progress["aliases_vi"].items() if isinstance(val, list)
    ]
    if not legacy_codes:
        return progress, False

    print(f"Phát hiện {len(legacy_codes)} entry aliases_vi ở checkpoint cũ (chưa QC) -- đang migrate...")
    all_flags = []
    for code in legacy_codes:
        rec = records_by_code.get(code)
        if rec is None:
            continue  # code không còn trong input hiện tại, bỏ qua an toàn
        raw_vi = progress["aliases_vi"][code]
        qc = qc_aliases(code, rec.get("aliases_en", []), raw_vi, source="llm_openrouter")
        progress["aliases_vi"][code] = qc
        for flag in qc["review_flags"]:
            all_flags.append({**flag, "code": code, "batch_context": "checkpoint_migration"})
    append_jsonl(review_live_path, all_flags)
    print(f"Migrate xong: {len(legacy_codes)} entry, {len(all_flags)} flag cần review -> {review_live_path}")
    return progress, True


def translate_missing(records, api_key, progress_file: Path, review_live_path: Path = DEFAULT_REVIEW_LIVE):
    import requests

    validate_unique_codes(records)
    session = requests.Session()
    progress = load_progress(progress_file)
    records_by_code = {r["code"]: r for r in records}

    progress, migrated = migrate_legacy_aliases_checkpoint(progress, records_by_code, review_live_path)
    if migrated:
        save_progress(progress, progress_file)

    need_preferred = [
        r for r in records
        if not has_text(r.get("preferred_vi")) and r["code"] not in progress["preferred_vi"]
    ]
    print(f"Cần dịch preferred_vi: {len(need_preferred)} record")
    for batch in chunks(need_preferred, BATCH_SIZE):
        codes = [r["code"] for r in batch]
        print(f"  preferred_vi batch: {codes[0]}..{codes[-1]} ({len(batch)} record)")
        result = translate_preferred_batch(session, api_key, batch)
        result = validate_preferred_result(batch, result)

        # QC ngay tại đây, trước khi lưu checkpoint: sửa lỗi thuật ngữ đã
        # biết (vd nếu preferred_en trùng 1 từ khoá trong TERM_CORRECTIONS)
        # và lưu bản ĐÃ SỬA vào checkpoint, không lưu bản raw.
        batch_flags = []
        n_corrected = 0
        for r in batch:
            code = r["code"]
            fixed, was_corrected, flags = qc_preferred(r.get("preferred_en", ""), result[code], source="llm_openrouter")
            result[code] = fixed
            if was_corrected:
                n_corrected += 1
            batch_flags.extend({**f, "code": code} for f in flags)
        print(f"    QC: {n_corrected} term correction áp dụng, {len(batch_flags)} flag cần review")
        append_jsonl(review_live_path, batch_flags)

        progress["preferred_vi"].update(result)
        save_progress(progress, progress_file)
        time.sleep(SLEEP_BETWEEN_CALLS)

    for r in records:
        if not has_text(r.get("preferred_vi")) and r["code"] in progress["preferred_vi"]:
            r["preferred_vi"] = progress["preferred_vi"][r["code"]]
            r["preferred_vi_source"] = "llm_openrouter"

    need_aliases = [
        r for r in records
        if r.get("aliases_en") and not r.get("aliases_vi") and r["code"] not in progress["aliases_vi"]
    ]
    print(f"Cần dịch aliases_vi: {len(need_aliases)} record")
    for batch in chunks(need_aliases, BATCH_SIZE):
        pending = list(batch)
        validation_attempts = {r["code"]: 0 for r in batch}
        while pending:
            codes = [r["code"] for r in pending]
            label = "batch" if len(pending) > 1 else "retry riêng"
            print(f"  aliases_vi {label}: {codes[0]}..{codes[-1]} ({len(pending)} record)")
            raw_result = translate_aliases_batch(session, api_key, pending)
            result, retry = partition_aliases_result(pending, raw_result)

            # QC và checkpoint ngay các code hợp lệ, trước khi gọi lại code lỗi.
            # Nhờ vậy kết quả đúng trong batch vẫn được giữ nếu lần retry sau lỗi.
            batch_flags = []
            n_coding_note = n_corrected = n_review = 0
            qc_by_code = {}
            records_by_pending_code = {r["code"]: r for r in pending}
            for code, aliases_vi in result.items():
                r = records_by_pending_code[code]
                qc = qc_aliases(code, r["aliases_en"], aliases_vi, source="llm_openrouter")
                qc_by_code[code] = qc
                n_coding_note += len(qc["inclusion_notes_vi"]) - sum(
                    1 for f in qc["review_flags"] if f["type"] == "long_or_ambiguous"
                )
                n_review += sum(1 for f in qc["review_flags"] if f["type"] == "long_or_ambiguous")
                n_corrected += len(qc["term_corrections_applied"])
                batch_flags.extend({**f, "code": code} for f in qc["review_flags"])
            if qc_by_code:
                print(
                    f"    QC + checkpoint {len(qc_by_code)} code: "
                    f"{n_coding_note} tách coding_note, {n_corrected} term correction, "
                    f"{n_review} câu dài cần review"
                )
                append_jsonl(review_live_path, batch_flags)
                progress["aliases_vi"].update(qc_by_code)
                save_progress(progress, progress_file)

            for r in retry:
                validation_attempts[r["code"]] += 1
            exhausted = [
                r["code"] for r in retry
                if validation_attempts[r["code"]] >= MAX_VALIDATION_RETRIES
            ]
            if exhausted:
                raise ValueError(
                    "Không nhận được đủ aliases_vi sau "
                    f"{MAX_VALIDATION_RETRIES} lần cho code: {exhausted}. "
                    "Các code hợp lệ trong batch đã được lưu checkpoint."
                )
            pending = retry
            time.sleep(SLEEP_BETWEEN_CALLS)

    for r in records:
        if not has_text(r.get("preferred_vi")) and r["code"] in progress["preferred_vi"]:
            r["preferred_vi"] = progress["preferred_vi"][r["code"]]
            r["preferred_vi_source"] = "llm_openrouter"
        if r.get("aliases_en") and not r.get("aliases_vi") and r["code"] in progress["aliases_vi"]:
            qc = progress["aliases_vi"][r["code"]]
            r["aliases_en_raw"] = r["aliases_en"]  # giữ nguyên gốc XML để audit
            r["aliases_en"] = qc["aliases_en"]
            r["aliases_vi"] = qc["aliases_vi"]
            r["aliases_vi_source"] = qc["aliases_vi_source"]
            if qc["inclusion_notes_vi"]:
                r["inclusion_notes_vi"] = qc["inclusion_notes_vi"]
    return records


# ---------- Bước 3: gộp alias thủ công (mã khó / có viết tắt) ----------

def merge_manual_aliases(records, manual_path: Path):
    """
    Đọc file JSON dạng {"code": ["alias 1", "alias 2", ...], ...} chứa
    alias được curate tay cho các mã khó (viết tắt như IBS-M, tên hiếm
    gặp như locked-in syndrome...). KHÔNG bắt buộc khớp số lượng với
    aliases_en (khác với aliases_vi dịch từ inclusion), vì đây là bổ
    sung thêm để tăng recall cho embedding, không phải bản dịch 1-1.

    Nối thêm (dedup) vào aliases_vi đã có sẵn -- không đè.
    """
    if not manual_path.exists():
        print(f"Không có file alias thủ công ({manual_path}), bỏ qua bước này.")
        return records

    with open(manual_path, encoding="utf-8") as f:
        manual_map = json.load(f)

    n_applied = 0
    for rec in records:
        extra = manual_map.get(rec["code"])
        if not extra:
            continue
        existing = rec.get("aliases_vi", [])
        # aliases_vi_source may be missing (e.g. record came straight from
        # byt_official join with no LLM step) -- backfill as "unknown" so the
        # array stays the same length as aliases_vi before we append.
        existing_source = rec.get("aliases_vi_source", [])
        if len(existing_source) != len(existing):
            existing_source = ["unknown"] * len(existing)
        merged = list(existing)
        merged_source = list(existing_source)
        for alias in extra:
            if alias not in merged:
                merged.append(alias)
                merged_source.append("manual")
        rec["aliases_vi"] = merged
        rec["aliases_vi_source"] = merged_source
        n_applied += 1

    print(f"Gộp alias thủ công: {n_applied}/{len(manual_map)} mã trong {manual_path.name} khớp với record hiện có")
    return records


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Join bản dịch VN chính thức + dịch LLM phần còn thiếu")
    ap.add_argument("--clean", type=Path, default=DEFAULT_CLEAN_JSONL, help="Input icd10_clean.jsonl (từ parse_xml.py)")
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="File danh mục ICD-10 tiếng Việt (.xlsx)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSONL, help="File jsonl kết quả cuối")
    ap.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS_FILE, help="File checkpoint dịch LLM")
    ap.add_argument("--review-live", type=Path, default=DEFAULT_REVIEW_LIVE,
                     help="File jsonl được append liên tục trong lúc dịch (tail -f để theo dõi coding_note/term-correction/review theo từng batch)")
    ap.add_argument("--skip-llm", action="store_true", help="Chỉ join bản dịch chính thức, không gọi OpenRouter")
    args = ap.parse_args()

    if not args.clean.exists():
        raise SystemExit(f"Không tìm thấy {args.clean} -- chạy parse_xml.py trước.")
    if not args.xlsx.exists():
        raise SystemExit(f"Không tìm thấy {args.xlsx} -- thả file xlsx vào đây rồi chạy lại.")

    with open(args.clean, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    print(f"Tổng số record: {len(records)}")

    records = join_vn_official(records, args.xlsx)

    if args.skip_llm:
        print("Bỏ qua bước dịch LLM (--skip-llm).")
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "Thiếu OPENROUTER_API_KEY trong biến môi trường.\n"
                "  export OPENROUTER_API_KEY=\"sk-or-...\"\n"
                "Hoặc chạy với --skip-llm để chỉ lấy phần join chính thức."
            )
        records = translate_missing(records, api_key, args.progress, args.review_live)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            r.pop("needs_vi_translation", None)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã ghi {args.output}")

    n_missing_pref = sum(1 for r in records if not has_text(r.get("preferred_vi")))
    n_missing_alias = sum(1 for r in records if r.get("aliases_en") and not r.get("aliases_vi"))
    print(f"Còn thiếu preferred_vi: {n_missing_pref}")
    print(f"Còn thiếu aliases_vi: {n_missing_alias}")


if __name__ == "__main__":
    main()
