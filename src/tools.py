#!/usr/bin/env python3
"""Tải SapBERT về thư mục local để dùng offline."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from linking.sapbert_encoder import DEFAULT_MODEL_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "sapbert"
)


def main() -> None:
    LOCAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"SapBERT model: {DEFAULT_MODEL_ID}")
    print(f"Thư mục lưu: {LOCAL_MODEL_DIR}")

    downloaded_path = snapshot_download(
        repo_id=DEFAULT_MODEL_ID,
        local_dir=str(LOCAL_MODEL_DIR),
    )

    config_path = LOCAL_MODEL_DIR / "config.json"

    if not config_path.exists():
        raise RuntimeError(
            f"Tải chưa đầy đủ, không tìm thấy: {config_path}"
        )

    print("\nTải SapBERT thành công.")
    print(f"Model local: {downloaded_path}")


if __name__ == "__main__":
    main()