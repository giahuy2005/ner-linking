# RunPod Guide â€” cháº¡y thá»­ `ner-linking` pipeline

Guide nÃ y dÃ¹ng cho pod má»›i trÃªn RunPod, thÆ° má»¥c lÃ m viá»‡c máº·c Ä‘á»‹nh lÃ :

```bash
/workspace/ner-linking
```
Repo code:

```text
https://github.com/giahuy2005/ner-linking.git
```

Model/data/index trÃªn HuggingFace:

```text
AIwho/ner_llm
```

---

## 0. Táº¡o pod

NÃªn chá»n image PyTorch cÃ³ CUDA sáºµn. Sau khi vÃ o terminal pod, kiá»ƒm tra GPU:

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

Náº¿u repo private thÃ¬ cáº§n clone báº±ng token GitHub.

---

## 2. CÃ i package há»‡ thá»‘ng vÃ  Python

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

Kiá»ƒm tra cÃ¡c package chÃ­nh:

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

## 3. Login HuggingFace náº¿u repo HF private

Náº¿u repo `AIwho/ner_llm` private:

```bash
hf auth login
hf auth whoami
```

Náº¿u repo public thÃ¬ cÃ³ thá»ƒ bá» qua bÆ°á»›c nÃ y.

---

## 4. Táº£i model/data/index tá»« HuggingFace

Quan trá»ng: repo HF Ä‘ang cÃ³ prefix `model_ner/...`, nÃªn pháº£i include Ä‘Ãºng prefix nÃ y.

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

Kiá»ƒm tra file:

```bash
ls -lah models
ls -lah models/ner
ls -lah models/icd10
ls -lah models/rxnorm
ls -lah models/sapbert
ls -lah data/processed
```

Cáº§n tháº¥y Ã­t nháº¥t:

```text
models/ner/best_ner_assertion_span_model.pth
models/ner/label_dicts.json

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

## 5. Táº£i VnCoreNLP

KhÃ´ng dÃ¹ng dáº¥u `!` trong terminal Linux.

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

## 6. Fix nÃ³ng path Windows trong config náº¿u cÃ²n lá»—i

Náº¿u cháº¡y mÃ  tháº¥y lá»—i kiá»ƒu:

```text
Repo id ... 'D:\Viettel_AI\viettel_ai_ner\models\sapbert'
```

thÃ¬ cháº¡y block nÃ y:

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

Check láº¡i:

```bash
cat models/icd10/icd10_index_config.json | grep model_id
cat models/rxnorm/rxnorm_index_config.json | grep model_id
```

Ká»³ vá»ng:

```text
"model_id": "models/sapbert"
```

---

## 8. Táº¡o input test nhanh

```bash
cd /workspace/ner-linking

mkdir -p data/input_ output

cat > data/input_/1.txt <<'TXT'
Bá»‡nh nhÃ¢n nam 65 tuá»•i, tiá»n sá»­ tÄƒng huyáº¿t Ã¡p vÃ  Ä‘Ã¡i thÃ¡o Ä‘Æ°á»ng type 2, Ä‘ang dÃ¹ng amlodipine 5 mg uá»‘ng má»—i ngÃ y vÃ  metformin 500 mg uá»‘ng hai láº§n má»—i ngÃ y. Hiá»‡n vÃ o viá»‡n vÃ¬ Ä‘au ngá»±c, khÃ³ thá»Ÿ, khÃ´ng sá»‘t. XÃ©t nghiá»‡m glucose mÃ¡u 12.5 mmol/L, creatinin 110 umol/L. Cháº©n Ä‘oÃ¡n: cÆ¡n Ä‘au tháº¯t ngá»±c khÃ´ng á»•n Ä‘á»‹nh.
TXT
```

---

## 9. Cháº¡y thá»­ pipeline

Xem CLI help:

```bash
python -m src.inference.cli --help
```

Cháº¡y full ICD10 + RxNorm:

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

## 10. Cháº¡y cáº£ folder 100 file

Äáº·t input theo dáº¡ng:

```text
data/input_/1.txt
data/input_/2.txt
...
data/input_/100.txt
```

Cháº¡y:

```bash
cd /workspace/ner-linking

rm -rf output
mkdir -p output

python -m src.inference.cli \
  --input-dir data/input_ \
  --output-dir output \
  --with-rxnorm \
  --with-icd10
```

Check sá»‘ file:

```bash
find output -name "*.json" | wc -l
ls -lah output | head
```

Zip ná»™p:

```bash
cd /workspace/ner-linking
rm -f output.zip
zip -r output.zip output
ls -lh output.zip
```

Náº¿u BTC yÃªu cáº§u zip giáº£i nÃ©n ra trá»±c tiáº¿p cÃ¡c file `.json`, dÃ¹ng:

```bash
cd /workspace/ner-linking/output
zip -r ../output.zip .
cd ..
ls -lh output.zip
```

---

## 11. Lá»‡nh debug nhanh

### Kiá»ƒm tra cÃ²n path Windows khÃ´ng

```bash
cd /workspace/ner-linking

grep -R --exclude="*.pth" --exclude="*.npy" --exclude="*.index" --exclude="*.safetensors" \
  "D:\\|D:/|Viettel_AI" -n src models data | head -80
```

### Kiá»ƒm tra model_id

```bash
cat models/icd10/icd10_index_config.json | grep model_id
cat models/rxnorm/rxnorm_index_config.json | grep model_id
```

### Kiá»ƒm tra SapBERT local Ä‘á»§ file

```bash
ls -lah models/sapbert
```

Cáº§n cÃ³:

```text
config.json
model.safetensors
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

Náº¿u thiáº¿u `pytorch_model.bin` nhÆ°ng cÃ³ `model.safetensors` thÃ¬ thÆ°á»ng váº«n OK.

### Kiá»ƒm tra checkpoint NER

```bash
ls -lah models/ner
```

Cáº§n cÃ³:

```text
best_ner_assertion_span_model.pth
label_dicts.json
```

### Xem GPU RAM

```bash
watch -n 1 nvidia-smi
```

---

## 12. Sau khi cháº¡y xong

Náº¿u chá»‰ test vÃ  khÃ´ng muá»‘n tá»‘n tiá»n ná»¯a:

- `Stop` pod: thÆ°á»ng váº«n cÃ³ thá»ƒ cÃ²n tÃ­nh tiá»n storage volume.
- `Terminate` pod: dá»«ng compute.
- XÃ³a volume náº¿u khÃ´ng cáº§n giá»¯ dá»¯ liá»‡u Ä‘á»ƒ trÃ¡nh phÃ­ storage.

TrÆ°á»›c khi terminate, nhá»› táº£i `output.zip` vá».


## Bash run

```bash
python -m src.inference.cli \
  --input-dir data/input \
  --output-dir output \
  --with-llm-8b \
  --with-rxnorm \
  --with-icd10 \
  --llm-dtype bfloat16 \
  --llm-quantization none \
  --llm-cache-path output/qwen_cache.jsonl \
  --llm-audit-dir output/audit \
  2>&1 | tee run_qwen3_8b.log
```
