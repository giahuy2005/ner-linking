# Kiến trúc model và luồng dữ liệu hiện tại

> Cập nhật 2026-07-30: notebook `predict_ner_crf_and__llm_fixed.ipynb`
> là nguồn chuẩn cho phần NER. Luồng production hiện tại là:

```text
NER Pass 1
  -> rule audit + suspicious regions
  -> NER Pass 2 trên các region
  -> exact dedup + conflict resolution
  -> deterministic cleanup/rule recovery
  -> grouped REVIEW_REGION + RECOVER_MISSING_ENTITIES
  -> Qwen2.5-7B NER batch (retry riêng request/batch lỗi)
  -> exact-span/type/assertion/overlap validation
  -> RxNorm + ICD-10 retriever hiện có
  -> Qwen2.5-7B chọn/rerank trong candidate linking
  -> BTC JSON
```

Trong task NER, 7B không được sinh mã. Ở stage linking riêng, 7B được chọn
RxNorm/ICD-10 nhưng chỉ trong candidate do retriever hiện tại trả về; code ngoài
danh sách bị validator từ chối. Khi response lỗi, NER trước 7B được giữ nguyên
và linking fallback theo thứ tự retriever. Các rule deterministic nằm trong
`src/inference/rule/`.


## 1. Luồng xử lý inference tổng thể

```text
Nhiều file .txt
  -> clean text + ánh xạ offset về raw text
  -> NER Pass 1: ViHealthBERT + CRF + assertion
  -> rule audit + phát hiện suspicious regions
  -> NER Pass 2 chỉ trên các region
  -> exact dedup + merge + conflict resolution
  -> deterministic cleanup/rule recovery
  -> grouped REVIEW_REGION + RECOVER_MISSING_ENTITIES
  -> Qwen2.5-7B review/recover NER theo batch
  -> validator exact span/type/assertion/overlap
  -> danh sách NER cuối cùng
  -> RxNorm retrieval cho THUỐC / ICD-10 retrieval cho CHẨN_ĐOÁN
  -> Qwen2.5-7B rerank candidate linking theo batch
  -> validate code, fallback linker nếu cần
  -> assemble BTC JSON
```

Văn bản được ánh xạ offset về raw text ngay sau mỗi lượt NER. Mọi stage sau đó
dùng half-open position `[start, end)` trên raw text và phải thỏa:

```python
0 <= start < end <= len(raw_text)
raw_text[start:end] == entity.text
```

Ba type `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM` và
`KẾT_QUẢ_XÉT_NGHIỆM` không chạy linking.

## 2. Stage 1 — Two-pass NER

Điểm vào là `InferencePipeline.run_ner_stage()`.

### 2.1. NER Pass 1

`NerEngine` dùng VnCoreNLP để word-segment, ViHealthBERT làm encoder, linear
head tạo BIO emission và CRF để decode chuỗi tag. Assertion head pool entity
cùng context xung quanh để dự đoán `isHistorical`, `isNegated`, `isFamily`.

Năm type hợp lệ:

```text
TRIỆU_CHỨNG
CHẨN_ĐOÁN
THUỐC
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
```

### 2.2. Suspicious-region detection và Pass 2

`detect_suspicious_regions()` trong `src/inference/ner/two_pass.py` route vùng
có confidence thấp, repair flag, boundary đáng ngờ, token lặp, occurrence bị
sót, dòng y tế không có entity hoặc long medical gap. Các hit gần nhau được
gộp thành `SuspiciousRegion`; mặc định tối đa 24 region mỗi document.

Pass 2 dùng lại chính `NerEngine` đã load và chỉ predict context của các region.
Offset local được đổi về global raw offset. Region lỗi sẽ log `pass2_error` và
không làm mất candidate Pass 1.

### 2.3. Merge và deterministic cleanup

Candidate Pass 1 và Pass 2 được exact-dedup theo `(start, end, type)`, hợp
assertions, rồi resolve overlap deterministic theo score, độ dài và vị trí.
Rule production nằm trong `src/inference/rule/clinical.py`, gồm:

- Boundary thừa/thiếu như `sốt bn`, `bn vàng da`, ngoặc/newline/connector thừa.
- Repeated token/cụm như `chụp chụp...`, `Chụp lại chụp...`.
- Specimen boundary `hầu họng` -> `dịch hầu họng` khi context hỗ trợ.
- Hard negatives: `◦ 8`, `đứng dậy`, `đánh răng không`, `ăn ngủ`,
  `tĩnh mạch L giọt/phút`, `cấp tính`, `Tomisaku Kawasaki`.
- Giải phẫu trần bị gán chẩn đoán và `G6PD` bị gán sai thành xét nghiệm.
- Danh sách thuốc trước nhập viện: recover regimen; chỉ thuốc nhận
  `isHistorical`, triệu chứng chỉ định không kế thừa assertion của thuốc.

## 3. Handoff và 7B NER

`build_handoff_requests()` trong `src/inference/rule/routing.py` tạo schema
`7b_handoff_v2_grouped` với hai task.

### 3.1. `REVIEW_REGION`

Nhiều target gần nhau được gom trong cùng context. Request giữ `request_id`,
`candidate_id`, global/relative position, assertions và allowed actions.

```json
{
  "task": "REVIEW_REGION",
  "request_id": "record-review-region-0-100-150",
  "context_global_position": [20, 250],
  "target_candidate_ids": [3, 4],
  "targets": [{
    "candidate_id": 3,
    "text": "sốt bn",
    "type": "TRIỆU_CHỨNG",
    "global_position": [100, 106],
    "relative_position": [80, 86],
    "assertions": [],
    "allowed_actions": ["KEEP", "DROP", "REPAIR_SPAN", "RETYPE"]
  }]
}
```

7B phải trả đúng một decision cho mỗi target. `KEEP` giữ nguyên; `DROP` xóa;
`REPAIR_SPAN` sửa boundary; `RETYPE` đổi sang một trong năm type hợp lệ.

### 3.2. `RECOVER_MISSING_ENTITIES`

Request chứa suspicious region, focus span và entity đã tồn tại trong context.
7B chỉ được trả entity thật sự bị bỏ sót hoặc boundary rộng hơn cho fragment
cùng type. `relative_position` luôn là `[start, end)` trên context request.

`--no-llm-recall-audit` chỉ tắt recovery request; review target đã có và 7B
linking vẫn chạy.

### 3.3. Batch, retry và validator

`review_entities_batch()` gom request của nhiều document và gọi Qwen2.5-7B qua
`generate_batch()`. Chỉ request lỗi được retry; request hợp lệ không chạy lại.

Validator kiểm tra tối thiểu:

- `request_id` khớp và mỗi target có đúng một decision.
- Không chỉnh candidate ngoài `target_candidate_ids`.
- Action/type/assertion thuộc allow-list.
- Text khớp chính xác `raw_text[start:end]`.
- Span sửa nằm trong context và gần/overlap span gốc.
- Relative position không âm, không vượt context.
- Không tạo exact duplicate hoặc overlap không an toàn.
- Recovery chỉ được thay fragment cùng type khi span mới bao trọn fragment.

Parse/schema/generation lỗi sau retry sẽ fallback về output NER trước 7B. Log
accepted/rejected/fallback được lưu ở `InferencePipeline.last_7b_logs`.

## 4. Retrieval và 7B linking

Linking chỉ chạy sau khi NER đã hoàn tất:

```text
THUỐC      -> RxNormLinker
CHẨN_ĐOÁN -> Icd10Linker
```

Index, embedding, FAISS, retrieval và ranking hiện có không bị thay đổi. Nếu 7B
đã sửa text/type NER, linker chạy lại trên entity cuối; candidate cũ không được
tái sử dụng.

Khi không bật 7B, pipeline lấy theo ranking của linker: tối đa 1 RxNorm code
cho thuốc và 3 ICD-10 code cho chẩn đoán.

Khi bật 7B, `select_candidates_many()` gom các entity mơ hồ của nhiều document
thành batch. Prompt linking nhận entity text/type, raw context và candidate kèm
metadata/ranking feature. 7B có quyền chọn/rerank candidate, với các giới hạn:

- Chỉ được chọn code có trong candidate do retriever trả về.
- Không được bịa code mới.
- THUỐC luôn tối đa 1 RxNorm code.
- CHẨN_ĐOÁN tối đa 2 ICD-10 code; exact alias trả 1 mã, semantic candidate
  dưới ngưỡng bị loại, và khi top-1 cách top-2 đủ xa chỉ giữ top-1.
- JSON lỗi, code không hợp lệ hoặc lỗi generation sẽ fallback về top candidate
  của linker.

Exact match chắc chắn có thể bypass 7B để tiết kiệm generation: ICD-10 khi text
khớp normalized `matched_term`; RxNorm khi ingredient exact và
strength/form/release không mismatch.

## 5. Lifecycle model và cấu hình

CLI quản lý model theo trình tự:

```text
1. Load NER engine và các linker được bật.
2. Chạy two-pass NER cho toàn bộ input batch.
3. Load Qwen2.5-7B đúng một lần.
4. Chạy tất cả NER review/recovery request theo batch.
5. Giữ 7B trên GPU.
6. Chạy retrieval và 7B linking rerank theo batch.
7. Unload 7B và giải phóng VRAM.
8. Assemble và ghi BTC JSON.
```

Pipeline không load/unload 7B theo candidate hoặc document.

`NER_REVIEWER_7B_CONFIG` mặc định:

```text
model_id           = Qwen/Qwen2.5-7B-Instruct
load_in_4bit       = true
batch_size         = 4
max_new_tokens     = 512
temperature        = 0.0
max_context_length = 8192
retry_rounds       = 1
```

## 6. Output

```json
{
  "text": "amlodipine 10 mg po daily",
  "type": "THUỐC",
  "candidates": ["308135"],
  "assertions": ["isHistorical"],
  "position": [58, 83]
}
```

`candidates` chỉ được gắn ở stage linking sau khi NER hoàn tất. Entity không
thuộc type cần linking không nhận code.

## 7. CLI

Chỉ NER:

```bash
python -m src.inference.cli --input data/1.txt --print
```

NER + linker, không dùng 7B:

```bash
python -m src.inference.cli \
  --input-dir data/public_test \
  --output-dir output \
  --with-rxnorm --with-icd10
```

Pipeline đầy đủ, gồm 7B NER và 7B linking:

```bash
python -m src.inference.cli \
  --input-dir data/public_test \
  --output-dir output \
  --with-rxnorm --with-icd10 --with-llm-7b
```

Cờ tương thích:

- `--with-llm-fixer`: alias của `--with-llm-7b`.
- `--with-llm-selector`: alias của `--with-llm-7b`.
- `--no-llm-recall-audit`: tắt recovery NER, vẫn review NER và rerank linking.
- `--no-repair-gate`: tắt repair gate để A/B test.

## 8. Module chịu trách nhiệm

| Module | Trách nhiệm |
|---|---|
| `src/inference/pipeline.py` | Điều phối stage và dữ liệu batch |
| `src/inference/ner/engine.py` | ViHealthBERT + CRF + assertion inference |
| `src/inference/ner/two_pass.py` | Suspicious region, Pass 2, merge |
| `src/inference/rule/clinical.py` | Cleanup, boundary, hard-negative, medication rules |
| `src/inference/rule/routing.py` | Tạo grouped 7B NER requests |
| `src/inference/ner/reviewer_7b.py` | Batch/retry/validate/apply NER response |
| `src/inference/selection/candidate_selector.py` | 7B linking rerank và code validation |
| `src/linking/rxnorm/` | RxNorm retrieval/ranking hiện có |
| `src/linking/icd10/` | ICD-10 retrieval/ranking hiện có |
| `src/llm/backend.py` | Local model load, batch generation, unload |
| `src/inference/io.py` | Raw-offset mapping, output assembly/validation |

## 9. Test và invariant chính

`tests/test_new_ner_pipeline.py` kiểm tra gold 11 thuốc trước nhập viện,
assertion/offset, boundary, repeated token, false positive, recovery,
batch/retry/fallback và thời điểm gắn RxNorm ID.

`tests/test_inference_regressions.py` kiểm tra candidate selector theo batch,
code allow-list, fallback linker và giới hạn số code output.
