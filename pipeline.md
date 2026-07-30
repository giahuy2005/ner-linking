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

## 4. Assertion head học gì?

Assertion không phải token classification. Sau khi có entity span, model lấy
mean pooling hidden states của các subword nằm trong span đó:

```text
hidden_states[token_start : token_end]
        |
        v
mean pooling
        |
        v
span vector
        |
        v
Linear(hidden_size, num_assert_labels)
```

Assertion là bài toán multi-label:

```text
isHistorical
isNegated
isFamily
```

Một entity có thể có nhiều assertion cùng lúc, ví dụ:

```json
{
  "text": "cường giáp",
  "type": "CHAN_DOAN",
  "assertions": ["isFamily", "isHistorical"]
}
```

Vì vậy assertion head dùng:

```text
BCEWithLogitsLoss
```

Không cần class `NONE`. Entity không có assertion được biểu diễn bằng vector
toàn 0:

```text
[0, 0, 0]
```

## 5. Luồng dữ liệu train cho khối NER

```text
data/synthetic/train.jsonl
        |
        v
input_text + entities
        |
        v
VnCoreNLP word segmentation + char span mapping
        |
        v
tokens + ner_tags + assertion_spans
        |
        v
NerDataset
        |
        +--> chunk tokens: chunk_size=200, overlap=50
        +--> tokenize with ViHealthBERT tokenizer
        +--> align BIO label to first subword
        +--> remap assertion spans to subword index
        |
        v
DataLoader
        |
        v
NerAssertionModel
        |
        +--> ner_loss
        +--> assertion_loss
        |
        v
total_loss = ner_loss + assertion_loss_weight * assertion_loss
```

Tóm lại, phần NER hiện tại gồm hai nhiệm vụ chạy chung một encoder:

- Token-level BIO tagging để tìm ranh giới và loại entity.
- Span-level assertion classification để gắn trạng thái cho entity đã tìm được.

Sau khối này mới đến validate span, linking ICD10/RxNorm và reconstruct offset
theo sơ đồ inference ở trên.
