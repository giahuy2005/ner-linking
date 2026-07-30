# RunPod Guide — chạy thử `ner-linking` pipeline

Guide này dùng cho pod mới trên RunPod, thư mục làm việc mặc định là:

```bash
/workspace/ner-linking
```

Repo code:

```text
https://github.com/giahuy2005/ner-linking.git
```

Model/data/index trên HuggingFace:

```text
AIwho/ner_llm
```

---

## 0. Tạo pod

Nên chọn image PyTorch có CUDA sẵn. Sau khi vào terminal pod, kiểm tra GPU:

```bash
nvidia-smi
```

---

## 1. Clone code

```bash
cd /workspace

git clone https://github.com/giahuy2005/ner-linking.git
cd /workspace/ner-linking
```

Nếu repo private thì cần clone bằng token GitHub.

---

## 2. Cài package hệ thống và Python

```bash
apt-get update
apt-get install -y git wget curl unzip zip openjdk-17-jre-headless

python -m pip install -U pip setuptools wheel
pip install -r requirements.txt

pip install -U \
  accelerate \
  transformers \
  safetensors \
  sentencepiece \
  protobuf \
  huggingface_hub \
  faiss-cpu \
  rapidfuzz \
  pytorch-crf \
  numpy \
  pandas \
  scikit-learn \
  tqdm \
  lxml \
  requests \
  openai
```

Kiểm tra các package chính:

```bash
python - <<'PY'
import torch, transformers, faiss
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("faiss ok")
PY
```

---

## 3. Login HuggingFace nếu repo HF private

Nếu repo `AIwho/ner_llm` private:

```bash
hf auth login
hf auth whoami
```

Nếu repo public thì có thể bỏ qua bước này.

---

## 4. Tải model/data/index từ HuggingFace

Quan trọng: repo HF đang có prefix `model_ner/...`, nên phải include đúng prefix này.

```bash
cd /workspace/ner-linking

rm -rf models data/processed model_ner

hf download AIwho/ner_llm \
  --repo-type model \
  --local-dir . \
  --include "model_ner/models/ner/**" \
  --include "model_ner/models/icd10/**" \
  --include "model_ner/models/rxnorm/**" \
  --include "model_ner/models/sapbert/config.json" \
  --include "model_ner/models/sapbert/model.safetensors" \
  --include "model_ner/models/sapbert/special_tokens_map.json" \
  --include "model_ner/models/sapbert/tokenizer_config.json" \
  --include "model_ner/models/sapbert/vocab.txt" \
  --include "model_ner/processed/**"

mkdir -p data

mv model_ner/models ./models
mv model_ner/processed ./data/processed

rm -rf model_ner
```

Kiểm tra file:

```bash
ls -lah models
ls -lah models/ner
ls -lah models/icd10
ls -lah models/rxnorm
ls -lah models/sapbert
ls -lah data/processed
```

Cần thấy ít nhất:

```text
models/ner/best_ner_assertion_model.pth
models/ner/label_dicts_crf.json

models/sapbert/config.json
models/sapbert/model.safetensors
models/sapbert/vocab.txt

models/icd10/icd10_faiss.index
models/icd10/icd10_metadata.jsonl
models/icd10/icd10_index_config.json

models/rxnorm/rxnorm_index_config.json
models/rxnorm/product_sapbert.index
models/rxnorm/product_metadata.jsonl
models/rxnorm/product_sapbert_embeddings.npy
models/rxnorm/support_sapbert.index
models/rxnorm/support_metadata.jsonl
models/rxnorm/support_sapbert_embeddings.npy
models/rxnorm/historical_sapbert.index
models/rxnorm/historical_metadata.jsonl
models/rxnorm/historical_sapbert_embeddings.npy
```

---

## 5. Tải VnCoreNLP

Không dùng dấu `!` trong terminal Linux.

```bash
cd /workspace/ner-linking

mkdir -p vncorenlp/models/wordsegmenter

wget -O vncorenlp/VnCoreNLP-1.1.1.jar \
  https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar

wget -O vncorenlp/models/wordsegmenter/vi-vocab \
  https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab

wget -O vncorenlp/models/wordsegmenter/wordsegmenter.rdr \
  https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

java -version
ls -lah vncorenlp
ls -lah vncorenlp/models/wordsegmenter
```

---

## 6. Fix nóng path Windows trong config nếu còn lỗi

Nếu chạy mà thấy lỗi kiểu:

```text
Repo id ... 'D:\Viettel_AI\viettel_ai_ner\models\sapbert'
```

thì chạy block này:

```bash
cd /workspace/ner-linking

python - <<'PY'
from pathlib import Path
import json

ROOT = Path("/workspace/ner-linking")

targets = [
    ROOT / "models/icd10/icd10_index_config.json",
    ROOT / "models/rxnorm/rxnorm_index_config.json",
]

OLD_ROOTS = [
    "D:/Viettel_AI/viettel_ai_ner/",
    "D:/Viettel_AI/ner-linking/",
    "/workspace/ner-linking/",
]

def fix_str(x: str) -> str:
    s = str(x).replace("\\", "/")

    for old in OLD_ROOTS:
        if s.startswith(old):
            s = s[len(old):]

    return s

def walk(obj):
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v) for v in obj]
    if isinstance(obj, str):
        return fix_str(obj)
    return obj

for p in targets:
    if not p.exists():
        print("missing:", p)
        continue

    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg = walk(cfg)

    if "model" in cfg and "model_id" in cfg["model"]:
        cfg["model"]["model_id"] = "models/sapbert"
    elif "model_id" in cfg:
        cfg["model_id"] = "models/sapbert"

    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print("fixed:", p)

    if "model" in cfg:
        print("model_id =", cfg["model"].get("model_id"))
    else:
        print("model_id =", cfg.get("model_id"))
PY
```

Check lại:

```bash
cat models/icd10/icd10_index_config.json | grep model_id
cat models/rxnorm/rxnorm_index_config.json | grep model_id
```

Kỳ vọng:

```text
"model_id": "models/sapbert"
```

---

## 7. Fix nóng lỗi `assertions` do LLM trả dict

Nếu gặp lỗi:

```text
TypeError: unhashable type: 'dict'
File ".../src/inference/ner/reviewer_7b.py", line ...
```

thì LLM reviewer trả `assertions` dạng dict. Chạy block này để patch hàm `_assertions`:

```bash
cd /workspace/ner-linking

python - <<'PY'
from pathlib import Path

p = Path("src/inference/ner/reviewer_7b.py")
lines = p.read_text(encoding="utf-8").splitlines()

start = None
for i, line in enumerate(lines):
    if line.startswith("def _assertions("):
        start = i
        break

if start is None:
    raise SystemExit("Không tìm thấy def _assertions")

end = len(lines)
for j in range(start + 1, len(lines)):
    if lines[j].startswith("def ") or lines[j].startswith("class "):
        end = j
        break

new_func = r'''
def _assertions(value, fallback=None):
    """Chuẩn hóa assertions từ output LLM.

    Chấp nhận:
    - ["isNegated"]
    - "isNegated"
    - [{"label": "isNegated"}]
    - [{"assertion": "isNegated"}]
    - [{"isNegated": true}]
    - nested list/dict lỗi nhẹ từ LLM

    Luôn trả list[str] hợp lệ, không crash.
    """
    if fallback is None:
        fallback = []

    allowed = set(ALLOWED_ASSERTIONS)
    out = []

    alias = {
        "negated": "isNegated",
        "negative": "isNegated",
        "is_negated": "isNegated",
        "historical": "isHistorical",
        "history": "isHistorical",
        "past": "isHistorical",
        "is_historical": "isHistorical",
        "family": "isFamily",
        "family_history": "isFamily",
        "is_family": "isFamily",
    }

    def add_one(x):
        if x is None:
            return

        if isinstance(x, str):
            s = x.strip()
            s = alias.get(s, s)
            if s in allowed and s not in out:
                out.append(s)
            return

        if isinstance(x, dict):
            for key in ("assertion", "label", "type", "name", "value"):
                if key in x:
                    add_one(x.get(key))

            for key, val in x.items():
                if key in allowed and bool(val):
                    add_one(key)
            return

        if isinstance(x, (list, tuple, set)):
            for item in x:
                add_one(item)
            return

    add_one(value)

    priority = ["isNegated", "isFamily", "isHistorical"]
    out = [a for a in priority if a in out]

    if out:
        return out

    if value == []:
        return []

    return list(fallback or [])
'''.strip("\n").splitlines()

new_lines = lines[:start] + new_func + lines[end:]
p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

print(f"fixed {p}")
PY
```

---

## 8. Tạo input test nhanh

```bash
cd /workspace/ner-linking

mkdir -p data/input_ output

cat > data/input_/1.txt <<'TXT'
Bệnh nhân nam 65 tuổi, tiền sử tăng huyết áp và đái tháo đường type 2, đang dùng amlodipine 5 mg uống mỗi ngày và metformin 500 mg uống hai lần mỗi ngày. Hiện vào viện vì đau ngực, khó thở, không sốt. Xét nghiệm glucose máu 12.5 mmol/L, creatinin 110 umol/L. Chẩn đoán: cơn đau thắt ngực không ổn định.
TXT
```

---

## 9. Chạy thử pipeline

Xem CLI help:

```bash
python -m src.inference.cli --help
```

Chạy full ICD10 + RxNorm:

```bash
python -m src.inference.cli \
  --input data/input_/1.txt \
  --output-dir output \
  --with-rxnorm \
  --with-icd10
```

Xem output:

```bash
ls -lah output
cat output/1.json
```

---

## 10. Chạy cả folder 100 file

Đặt input theo dạng:

```text
data/input_/1.txt
data/input_/2.txt
...
data/input_/100.txt
```

Chạy:

```bash
cd /workspace/ner-linking

rm -rf output
mkdir -p output

python -m src.inference.cli \
  --input data/input_ \
  --output-dir output \
  --with-rxnorm \
  --with-icd10
```

Check số file:

```bash
find output -name "*.json" | wc -l
ls -lah output | head
```

Zip nộp:

```bash
cd /workspace/ner-linking
rm -f output.zip
zip -r output.zip output
ls -lh output.zip
```

Nếu BTC yêu cầu zip giải nén ra trực tiếp các file `.json`, dùng:

```bash
cd /workspace/ner-linking/output
zip -r ../output.zip .
cd ..
ls -lh output.zip
```

---

## 11. Lệnh debug nhanh

### Kiểm tra còn path Windows không

```bash
cd /workspace/ner-linking

grep -R --exclude="*.pth" --exclude="*.npy" --exclude="*.index" --exclude="*.safetensors" \
  "D:\\|D:/|Viettel_AI" -n src models data | head -80
```

### Kiểm tra model_id

```bash
cat models/icd10/icd10_index_config.json | grep model_id
cat models/rxnorm/rxnorm_index_config.json | grep model_id
```

### Kiểm tra SapBERT local đủ file

```bash
ls -lah models/sapbert
```

Cần có:

```text
config.json
model.safetensors
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

Nếu thiếu `pytorch_model.bin` nhưng có `model.safetensors` thì thường vẫn OK.

### Kiểm tra checkpoint NER

```bash
ls -lah models/ner
```

Cần có:

```text
best_ner_assertion_model.pth
label_dicts_crf.json
```

### Xem GPU RAM

```bash
watch -n 1 nvidia-smi
```

---

## 12. Sau khi chạy xong

Nếu chỉ test và không muốn tốn tiền nữa:

- `Stop` pod: thường vẫn có thể còn tính tiền storage volume.
- `Terminate` pod: dừng compute.
- Xóa volume nếu không cần giữ dữ liệu để tránh phí storage.

Trước khi terminate, nhớ tải `output.zip` về.


## bash run 
python -m src.inference.cli \
  --input-dir data/input \
  --output-dir output_15b \
  --with-llm-fixer \
  2>&1 | tee run_15b.log

  python -m src.inference.cli \
  --input-dir data/input \
  --output-dir output_7b \
  --with-llm-fixer \
  --with-llm-7b \
  --with-rxnorm \
  --with-icd10 \
  2>&1 | tee run_7b.log