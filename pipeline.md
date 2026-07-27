# Kiến trúc model và luồng dữ liệu hiện tại


## 1. Luồng xử lý inference tổng thể

tách văn bản thành section, xử lý tuần tự từng section, section nào NER xong thì đi tiếp sang validate/linking/offset.

```text
                 +----------------------+
 file.txt  ----> |  Section Splitter    |
                 |  + Offset Map Builder|
                 +----------+-----------+
                            |
                            v
                 for section in sections:
                            |
                            v
                 +----------------------+
                 | 1. NER + Assertion  |
                 |    current section  |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | 2. Span Validation  |
                 |    match raw text   |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | 3. Candidate        |
                 |    Retrieval/Rerank |
                 +----------+-----------+
                            |
          +-----------------+-----------------+
          |                                   |
          v                                   v
   THUOC entities                      CHAN_DOAN entities
   -> RxNorm                           -> ICD10
                            |
                            v
                 +----------------------+
                 | 4. Offset           |
                 |    Reconstruction   |
                 +----------+-----------+
                            |
                            v
                 entities_section.append(...)
                            |
                            v
                 +----------------------+
                 | Assembler           |
                 | + schema validation |
                 +----------+-----------+
                            |
                            v
                    output/{id}.json
```

Điểm chính của luồng này:

- `Section Splitter` chia note dài thành các đoạn nhỏ hơn để model dễ xử lý.
- `Offset Map Builder` lưu mapping từ text đã làm sạch/đã tách section về text
  gốc.
- `NER + Assertion` là khối model ner multitask, nhận một section và trả về entity
  span kèm assertion.
- `Span Validation` loại entity mà text dự đoán không match lại được với
  section gốc.
- `Candidate Retrieval/Rerank` chỉ áp dụng cho entity cần linking, ví dụ thuốc
  và chẩn đoán.
- `Offset Reconstruction` đổi offset trong section về offset toàn file.
- `Assembler` gom entity của các section theo đúng thứ tự văn bản.

## 2. Khối NER hiện tại

Khối NER hiện tại không còn là LLM generate JSON trực tiếp. Notebook demo đang
train một model multi-task dựa trên ViHealthBERT:

```text
section text
    |
    v
VnCoreNLP word segmentation
    |
    v
word tokens
    |
    v
ViHealthBERT tokenizer
    |
    v
input_ids + attention_mask
    |
    v
+--------------------------------------------------+
| ViHealthBERT encoder                             |
| demdecuong/vihealthbert-base-word                |
+----------------------+---------------------------+
                       |
                       v
              hidden_states
                       |
        +--------------+--------------+
        |                             |
        v                             v
+---------------+             +--------------------+
| NER head      |             | Span mean pooling  |
| token linear  |             | over entity spans  |
+-------+-------+             +---------+----------+
        |                               |
        v                               v
 BIO logits                    span representations
        |                               |
        v                               v
 decode BIO                    +--------------------+
        |                      | Assertion head     |
        v                      | span linear        |
 entity spans                  +---------+----------+
                                        |
                                        v
                              assertion logits
```

Output logic:

```text
BIO logits
  -> argmax per token
  -> B-/I-/O tags
  -> decode thành entity spans

assertion logits
  -> sigmoid
  -> multi-label assertion per span
  -> isHistorical / isNegated / isFamily
```

## 3. NER head học gì?

NER head là một lớp `Linear(hidden_size, num_ner_tags)` đặt trên từng token của
ViHealthBERT.

Ví dụ label BIO:

```text
Tokens:
["đang", "dùng", "atorvastatin", "20", "mg", "điều_trị", "rối_loạn", "lipid", "máu"]

NER tags:
["O", "O", "B-THUOC", "I-THUOC", "I-THUOC", "O",
 "B-CHAN_DOAN", "I-CHAN_DOAN", "I-CHAN_DOAN"]
```

Ý nghĩa:

- `B-*`: token bắt đầu một entity.
- `I-*`: token tiếp tục entity cùng loại.
- `O`: token không thuộc entity nào.

Khi inference, model dự đoán BIO tag cho từng token, sau đó decode các chuỗi
`B-*` + `I-*` liền nhau thành entity span. Ví dụ:

```text
B-THUOC I-THUOC I-THUOC
=> entity: "atorvastatin 20 mg", type = THUOC
```

NER loss dùng:

```text
CrossEntropyLoss(ignore_index = -100)
```

`-100` được dùng cho padding và các subword phụ, để loss chỉ tính trên token
đầu của mỗi word-token.

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
