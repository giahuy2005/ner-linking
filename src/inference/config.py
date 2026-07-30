"""Cấu hình cho pipeline inference/ner.

Không đặt logic ở đây — chỉ hằng số / default path. Sửa các giá trị này
(hoặc override bằng biến môi trường / CLI arg ở tầng gọi) khi đổi model,
KHÔNG hard-code lại ở engine.py.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError:  # Lightweight validation/linking utilities can run without ML deps.
    torch = None


from ..linking.rxnorm.config import (
    DEFAULT_INDEX_DIR as DEFAULT_RXNORM_INDEX_DIR,
    DEFAULT_CLEAN_PATH as DEFAULT_RXNORM_CLEAN_PATH,
)
from ..linking.icd10.config import DEFAULT_INDEX_DIR as DEFAULT_ICD10_INDEX_DIR

# ---------------------------------------------------------------------------
# Model NER + assertion (bản CRF, khớp train_ner_colab_crf)
# ---------------------------------------------------------------------------
DEFAULT_BACKBONE = "demdecuong/vihealthbert-base-word"
DEFAULT_CHECKPOINT_PATH = Path("models/ner/best_ner_assertion_model.pth")
DEFAULT_LABEL_DICTS_PATH = Path("models/ner/label_dicts_crf.json")
DEFAULT_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


# PHẢI khớp giá trị dùng lúc train — sai giá trị này thì load_state_dict
# vẫn chạy được (không đổi shape) nhưng assertion head sẽ pool sai vùng
# context, làm assertion prediction sai lệch không báo lỗi.
CONTEXT_WINDOW = 10

# ---------------------------------------------------------------------------
# VnCoreNLP (word segmenter dùng để map offset)
# ---------------------------------------------------------------------------
DEFAULT_VNCORENLP_JAR = Path("vncorenlp/VnCoreNLP-1.1.1.jar")

# ---------------------------------------------------------------------------
# Suy luận / chunking
# ---------------------------------------------------------------------------
MAX_LEN = 256
OVERLAP_WORDS = 50
ASSERTION_THRESHOLD = 0.5
SINGLE_ASSERTION = False  # True nếu submit yêu cầu ép 1 assertion/entity

# ---------------------------------------------------------------------------
# Rule filter (repair_gate) — bật/tắt để dễ so sánh A/B khi tune
# ---------------------------------------------------------------------------
ENABLE_REPAIR_GATE = True

# Notebook V11 two-pass / grouped 7B handoff defaults.
MAXIMUM_SECOND_PASS_REGIONS = 24
ENTITY_REVIEW_SCORE_THRESHOLD = 0.82
MAXIMUM_REVIEW_REGIONS = 15
MAXIMUM_REVIEW_TARGETS_PER_REGION = 8
MAXIMUM_RECOVERY_REGIONS = 12

# ---------------------------------------------------------------------------
# Linking (RxNorm cho THUỐC, ICD-10 cho CHẨN_ĐOÁN) — dùng ở pipeline.py.
# Để None thì pipeline tự skip linking, chỉ chạy NER (test nhanh không cần
# build index/model linker).
# ---------------------------------------------------------------------------
RXNORM_INDEX_DIR: Path | None = DEFAULT_RXNORM_INDEX_DIR
RXNORM_CLEAN_PATH: Path | None = DEFAULT_RXNORM_CLEAN_PATH
ICD10_INDEX_DIR: Path | None = DEFAULT_ICD10_INDEX_DIR
LINKER_DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
LINKER_TOP_K = 10  # số candidate tối đa trả về mỗi entity
