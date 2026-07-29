# Viettel AI NER

## Data preprocessing

Repo nay giu logic xu ly data trong `src/preprocessing/` va cung cap script CLI trong
`scripts/` de co the chay lai duoc tren local, Colab, hoac khi nop bai.

### Cau truc du lieu

```text
data/
  raw/          # du lieu goc
  synthetic/    # du lieu sinh ra, vi du train_1.jsonl
  processed/    # output sau preprocess, vi du train_bio.jsonl
src/
  preprocessing/
    json_to_bio.py
scripts/
  preprocess_json_to_bio.py
notebook/
  # notebook dung de experiment, khong nen la noi duy nhat chua logic preprocess
```

### Cai VnCoreNLP

`json_to_bio.py` dung VnCoreNLP de word-segment cho ViHealthBERT-word, nen can Java
va cac file VnCoreNLP. Tren Colab co the chay:

```bash
pip install -r requirements.txt
mkdir -p vncorenlp/models/wordsegmenter
wget -q -O vncorenlp/VnCoreNLP-1.1.1.jar https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget -q -O vncorenlp/models/wordsegmenter/vi-vocab https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget -q -O vncorenlp/models/wordsegmenter/wordsegmenter.rdr https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
```

### Chay preprocess

Chay tu root repo:

```bash
python scripts/preprocess_json_to_bio.py \
  --input data/synthetic/train_1.jsonl \
  --output data/processed/train_bio.jsonl
```

Neu dung duong dan mac dinh:

```bash
python scripts/preprocess_json_to_bio.py
```

Script se tao thu muc output neu chua co va in thong ke convert:

```text
{"total": ..., "converted": ..., "sample_failed": ..., "entities_skipped": ...}
```

## Build va truy van ICD-10 FAISS

Corpus ICD-10 duoc encode bang SapBERT, L2-normalize va dua vao FAISS
`IndexFlatIP`. Bon artifact mac dinh duoc luu trong `models/icd10/`.

```bash
python src/linking/icd10/build_icd10_faiss_index.py
```

Truy van mot mention NER va gom ket qua theo ICD code bang max score:

```bash
python src/linking/icd10/icd10_linker.py "viem tai giua tiet dich" \
  --top-k-terms 50 \
  --top-k-codes 10
```

Su dung trong Python:

```python
from src.linking.icd10 import Icd10Linker

linker = Icd10Linker("models/icd10")
results = linker.link("viem tai giua tiet dich", top_k_terms=50, top_k_codes=10)
```

## Build corpus va FAISS RxNorm

Sau khi dat 6 file RRF trong `data/raw/rxnorm/`, tao corpus co the audit lai bang:

```bash
python src/preprocessing/rxnorm/build_rxnorm_corpus.py
```

Output nam trong `data/processed/rxnorm/`: concept da enrich, quan he graph,
history, embedding terms va build report. `rxnorm_relations.jsonl` luu huong ngu
nghia cua RRF (`RXCUI2 -> RXCUI1`) va giu them `raw_rxcui1/raw_rxcui2` de audit.

Mac dinh, ca product TTY va support TTY (gom IN/PIN/MIN/BN) deu co the tra ve
candidate vi gold cua BTC co the la ingredient khi mention khong du strength/form.
Co the thay doi chinh sach ma khong sua code:

```bash
python src/preprocessing/rxnorm/build_rxnorm_corpus.py \
  --output-ttys SCD,SBD,GPCK,BPCK,IN,PIN,MIN,BN,SCDC,SCDF
```

Tao ba index rieng (product, support, historical):

```bash
python src/linking/rxnorm/build_rxnorm_faiss_indexes.py
```

Neu tai nguyen han che, co the build tung tier bang `--tiers product` hoac
`--tiers support`. Pooling/model/max length duoc ghi vao
`models/rxnorm/rxnorm_index_config.json`; metadata va vector luon cung thu tu.

## Inference BTC

Chay batch day du voi 1.7B repair/recall audit va 7B candidate selector:

```bash
python -m src.inference.cli \
  --input-dir data/public_test \
  --output-dir output \
  --with-rxnorm --with-icd10 \
  --with-llm-fixer --with-llm-selector
```

Fixer sua cac entity bi repair gate danh dau, sau do audit mot lan moi tai lieu
de bo sung mention bi sot. Moi bo sung phai la substring nguyen van, co offset
hop le, khong overlap va qua repair gate. Selector bo qua LLM khi top candidate
la exact match chac chan; cac truong hop mo ho duoc chay batch. Output gioi han
mot RxNorm code cho `THUOC` va toi da ba ICD-10 code cho `CHAN_DOAN`.

Profile sinh V5 bo sung `btc_medication_lists` va
`complete_occurrence_recall`. Du lieu sinh ra can qua QC va chuyen BIO bang
`entity_spans`; can train lai checkpoint NER de cac thay doi data co hieu luc.

### Chay tren Colab

Notebook Colab nen clone repo va goi script, thay vi copy logic preprocess vao notebook:

```python
!git clone <repo_url>
%cd viettel_ai_ner

!pip install -r requirements.txt
!mkdir -p vncorenlp/models/wordsegmenter
!wget -q -O vncorenlp/VnCoreNLP-1.1.1.jar https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget -q -O vncorenlp/models/wordsegmenter/vi-vocab https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget -q -O vncorenlp/models/wordsegmenter/wordsegmenter.rdr https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr

!python scripts/preprocess_json_to_bio.py \
  --input data/synthetic/train_1.jsonl \
  --output data/processed/train_bio.jsonl
```
