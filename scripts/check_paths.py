from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BAD_TOKENS = [
    "D:\\",
    "D:/",
    "\\models\\",
    "\\data\\",
    "\\src\\",
    "/workspace/ner-linking\\",
    "models\\sapbert",
    "models\\rxnorm",
    "models\\icd10",
    "models\\ner",
]

TEXT_SUFFIXES = {
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
}


def normalize_path_string(path_str: str) -> str:
    s = str(path_str).strip().replace("\\", "/")

    old_roots = [
        "D:/Viettel_AI/viettel_ai_ner",
        "D:/Viettel_AI/ner-linking",
        "/workspace/ner-linking",
    ]

    for old_root in old_roots:
        if s.startswith(old_root):
            suffix = s[len(old_root):].lstrip("/")
            return f"{PROJECT_ROOT.as_posix()}/{suffix}" if suffix else PROJECT_ROOT.as_posix()

    for marker in ["/viettel_ai_ner/", "/ner-linking/"]:
        if marker in s:
            suffix = s.split(marker, 1)[1]
            return f"{PROJECT_ROOT.as_posix()}/{suffix}" if suffix else PROJECT_ROOT.as_posix()

    return s


def resolve_candidate(path_str: str, base_dir: Path | None = None) -> Path:
    s = normalize_path_string(path_str)

    if s.startswith("/"):
        return Path(s)

    if s.startswith(("models/", "data/", "src/", "configs/")):
        return PROJECT_ROOT / s

    if base_dir is not None and (base_dir / s).exists():
        return base_dir / s

    return PROJECT_ROOT / s


def scan_bad_text_paths() -> bool:
    print("\n[1] Scan text files for bad Windows paths...")

    found = False
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "configs",
    ]

    for root in roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
                continue

            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except UnicodeDecodeError:
                continue

            for line_no, line in enumerate(lines, 1):
                for token in BAD_TOKENS:
                    if token in line:
                        print(f"BAD_PATH: {p.relative_to(PROJECT_ROOT)}:{line_no}: contains {token!r}")
                        print(f"          {line.strip()}")
                        found = True
                        break

    if not found:
        print("OK: không thấy Windows path trong text files.")

    return found


def check_rxnorm_config() -> bool:
    print("\n[2] Check RxNorm config paths...")

    found = False
    cfg_path = PROJECT_ROOT / "models" / "rxnorm" / "rxnorm_index_config.json"

    if not cfg_path.is_file():
        print(f"MISSING: {cfg_path.relative_to(PROJECT_ROOT)}")
        return True

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    index_dir = cfg_path.parent

    model_id = cfg["model"]["model_id"]
    model_path = resolve_candidate(model_id, base_dir=PROJECT_ROOT)

    # Nếu model_id là local path thì phải tồn tại
    if str(model_id).replace("\\", "/").startswith(("models/", "data/", ".", "/", "D:/")):
        if not model_path.exists():
            print(f"MISSING_MODEL: {model_id} -> {model_path}")
            found = True
        else:
            print(f"OK model_id: {model_id} -> {model_path}")

    for tier, info in cfg["indexes"].items():
        for key in ["index_file", "metadata_file", "embedding_file"]:
            raw = info[key]
            resolved = resolve_candidate(raw, base_dir=index_dir)

            if not resolved.is_file():
                print(f"MISSING_RXNORM {tier}.{key}: {raw} -> {resolved}")
                found = True
            else:
                print(f"OK {tier}.{key}: {raw}")

    return found


def check_icd10_config() -> bool:
    print("\n[3] Check ICD10 config paths...")

    found = False
    cfg_path = PROJECT_ROOT / "models" / "icd10" / "icd10_index_config.json"

    if not cfg_path.is_file():
        print(f"MISSING: {cfg_path.relative_to(PROJECT_ROOT)}")
        return True

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    index_dir = cfg_path.parent

    model_cfg = cfg.get("model") or cfg
    model_id = model_cfg.get("model_id")

    if model_id:
        model_path = resolve_candidate(model_id, base_dir=PROJECT_ROOT)
        if str(model_id).replace("\\", "/").startswith(("models/", "data/", ".", "/", "D:/")):
            if not model_path.exists():
                print(f"MISSING_MODEL: {model_id} -> {model_path}")
                found = True
            else:
                print(f"OK model_id: {model_id} -> {model_path}")

    # Hỗ trợ nhiều schema khác nhau
    possible_files = []

    for key in ["index_file", "metadata_file", "embedding_file"]:
        if key in cfg:
            possible_files.append((key, cfg[key]))

    if "files" in cfg and isinstance(cfg["files"], dict):
        for key, value in cfg["files"].items():
            possible_files.append((f"files.{key}", value))

    if "index" in cfg and isinstance(cfg["index"], dict):
        for key, value in cfg["index"].items():
            possible_files.append((f"index.{key}", value))

    for key, raw in possible_files:
        resolved = resolve_candidate(raw, base_dir=index_dir)
        if not resolved.is_file():
            print(f"MISSING_ICD10 {key}: {raw} -> {resolved}")
            found = True
        else:
            print(f"OK {key}: {raw}")

    return found


def main() -> None:
    has_bad = False

    has_bad |= scan_bad_text_paths()
    has_bad |= check_rxnorm_config()
    has_bad |= check_icd10_config()

    print("\n==============================")
    if has_bad:
        print("STILL_HAS_PROBLEM")
        raise SystemExit(1)

    print("ALL_OK")


if __name__ == "__main__":
    main()