#!/usr/bin/env python3
"""Build auditable RxNorm concept, relation, history, and retrieval corpora.

The RRF files are streamed.  Only English RXNORM atoms are used as active
surface forms; graph edges are kept in their native RxNorm direction.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW = PROJECT_ROOT / "data" / "raw" / "rxnorm"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "rxnorm"

PRIMARY_PRODUCT_TTYS = {"SCD", "SBD"}
PACK_TTYS = {"GPCK", "BPCK"}
PRODUCT_TTYS = PRIMARY_PRODUCT_TTYS | PACK_TTYS
SUPPORT_TTYS = {
    "IN", "PIN", "MIN", "BN", "SCDC", "SCDF", "SCDFP", "SCDG", "SCDGP",
    "SBDC", "SBDF", "SBDFP", "SBDG", "SBDGP",
}
CANDIDATE_TTYS = PRODUCT_TTYS | SUPPORT_TTYS
# BTC gold is predominantly SCD, but also contains ingredient-level RXCUIs
# (for example nystatin 7597/IN).  Eligibility must therefore stay broader
# than product TTYs; candidate_priority controls granularity at reranking.
DEFAULT_OUTPUT_TTYS = set(CANDIDATE_TTYS)
PRIMARY_TTY_ORDER = [
    "SCD", "SBD", "GPCK", "BPCK", "IN", "PIN", "MIN", "BN", "SCDC",
    "SCDF", "SCDFP", "SCDG", "SCDGP", "SBDC", "SBDF", "SBDFP", "SBDG",
    "SBDGP", "DF", "DFG", "ET", "CD",
]
PRIMARY_TTY_RANK = {tty: index for index, tty in enumerate(PRIMARY_TTY_ORDER)}
ALIAS_TTYS = {"PSN", "SY", "TMSY"}
KEEP_RELA = {
    "has_ingredient", "ingredient_of", "has_precise_ingredient",
    "precise_ingredient_of", "consists_of", "constitutes", "has_dose_form",
    "dose_form_of", "has_tradename", "tradename_of", "has_ingredients",
    "ingredients_of", "contains", "contained_in", "quantified_form_of",
    "has_quantified_form", "isa", "inverse_isa",
}
KEEP_ATN = {
    "RXN_STRENGTH", "RXN_AVAILABLE_STRENGTH", "RXN_BOSS_AI", "RXN_BOSS_AM",
    "RXN_BOSS_FROM", "RXN_BOSS_STRENGTH_NUM_VALUE",
    "RXN_BOSS_STRENGTH_NUM_UNIT", "RXN_BOSS_STRENGTH_DENOM_VALUE",
    "RXN_BOSS_STRENGTH_DENOM_UNIT", "RXN_QUANTITY",
    "RXN_QUALITATIVE_DISTINCTION", "RXTERM_FORM", "RXN_HUMAN_DRUG",
    "RXN_VET_DRUG", "RXN_ACTIVATED", "RXN_OBSOLETED",
}
MULTI_ATN = {"RXN_QUALITATIVE_DISTINCTION", "RXTERM_FORM", "RXN_BOSS_FROM"}
SPACE_RE = re.compile(r"\s+")
UNIT_RE = re.compile(r"\b(mg|mcg|g|kg|ml|l|meq|mmol|unit|units|hr)\b", re.I)
PACK_QUANTITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:\([^()]*(?:ML|MG|MCG|G|L|UNT|UNIT)[^()]*\)\s*)?\(\s*$",
    re.I,
)


def fields(line: str) -> list[str]:
    values = line.rstrip("\r\n").split("|")
    if values and values[-1] == "":
        values.pop()
    return values


def clean_text(value: str) -> str:
    return SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def lexical_text(value: str) -> str:
    value = clean_text(value).casefold()
    return UNIT_RE.sub(lambda match: match.group(1).upper(), value)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    os.replace(temporary, path)
    return count


def read_concepts(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    atoms: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    stats: Counter[str] = Counter()
    with path.open(encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(handle, 1):
            row = fields(line)
            if len(row) < 18:
                raise ValueError(f"Malformed RXNCONSO row {number}: {len(row)} fields")
            rxcui, lat, isp, rxaui, sab, tty, text, suppress, cvf = (
                row[0], row[1], row[6], row[7], row[11], row[12], row[14], row[16], row[17]
            )
            if lat != "ENG" or sab != "RXNORM" or not text:
                continue
            atoms[rxcui].append({
                "text": clean_text(text), "source_tty": tty, "rxaui": rxaui,
                "ispref": isp, "suppress": suppress, "cvf": cvf,
            })
            stats["rxnorm_english_atoms"] += 1

    concepts: dict[str, dict[str, Any]] = {}
    for rxcui, group in atoms.items():
        primary = [a for a in group if a["source_tty"] not in ALIAS_TTYS]
        pool = primary or group
        canonical = min(
            pool,
            key=lambda a: (
                a["suppress"] != "N", a["ispref"] != "Y",
                PRIMARY_TTY_RANK.get(a["source_tty"], 999), len(a["text"]), a["text"].casefold(),
            ),
        )
        tty = canonical["source_tty"]
        active = any(a["suppress"] == "N" for a in group)
        seen: set[str] = set()
        names: list[dict[str, str]] = []
        ordered = sorted(
            group,
            key=lambda a: (a is not canonical, a["source_tty"] != "PSN", a["suppress"] != "N", a["text"].casefold()),
        )
        for atom in ordered:
            key = lexical_text(atom["text"])
            if not key or key in seen:
                continue
            seen.add(key)
            name_type = "canonical" if atom is canonical else (
                "prescribable" if atom["source_tty"] == "PSN" else "synonym"
            )
            names.append({
                "text": atom["text"], "normalized_text": key, "name_type": name_type,
                "source_tty": atom["source_tty"], "sab": "RXNORM", "suppress": atom["suppress"],
            })
        psn = next((a["text"] for a in ordered if a["source_tty"] == "PSN" and a["suppress"] == "N"), None)
        concepts[rxcui] = {
            "rxcui": rxcui, "tty": tty, "canonical_name": canonical["text"],
            "prescribable_name": psn, "names": names, "_atoms": group,
            "_active": active, "_prescribable": any(a["cvf"] == "4096" for a in group),
        }
        stats[f"concept_tty_{tty}"] += 1
    stats["concepts"] = len(concepts)
    return concepts, stats


def read_semantic_types(path: Path) -> dict[str, list[dict[str, str]]]:
    result: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = fields(line)
            if len(row) < 4:
                raise ValueError(f"Malformed RXNSTY row {number}")
            result[row[0]].append({"tui": row[1], "name": row[3]})
    return result


def read_attributes(path: Path) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    result: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    stats: Counter[str] = Counter()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = fields(line)
            if len(row) < 13:
                raise ValueError(f"Malformed RXNSAT row {number}: {len(row)} fields")
            rxcui, atn, sab, atv, suppress = row[0], row[8], row[9], row[10], row[11]
            if sab != "RXNORM" or atn not in KEEP_ATN or not rxcui or suppress not in {"", "N"}:
                continue
            if atn in MULTI_ATN:
                result[rxcui].setdefault(atn, [])
                if atv not in result[rxcui][atn]:
                    result[rxcui][atn].append(atv)
            else:
                result[rxcui].setdefault(atn, atv)
            stats[f"attribute_{atn}"] += 1
    stats["concepts_with_attributes"] = len(result)
    return result, stats


def read_relations(
    path: Path, concepts: dict[str, dict[str, Any]], output_path: Path,
) -> tuple[dict[str, list[tuple[str, str]]], Counter[str]]:
    adjacency: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    stats: Counter[str] = Counter()

    def records() -> Iterable[dict[str, Any]]:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                row = fields(line)
                if len(row) < 16:
                    raise ValueError(f"Malformed RXNREL row {number}: {len(row)} fields")
                # RRF defines REL/RELA as the relationship RXCUI2 has to
                # RXCUI1.  Therefore semantic source/target are the reverse of
                # their physical order in the row.
                raw_rxcui1, raw_rxcui2 = row[0], row[4]
                source, target, rela, sab, suppress = raw_rxcui2, raw_rxcui1, row[7], row[10], row[14]
                if (
                    sab != "RXNORM" or rela not in KEEP_RELA or not source or not target
                    or source not in concepts or target not in concepts
                ):
                    continue
                adjacency[source].append((rela, target))
                stats[f"relation_{rela}"] += 1
                yield {
                    "source_rxcui": source, "source_tty": concepts[source]["tty"],
                    "rela": rela, "target_rxcui": target, "target_tty": concepts[target]["tty"],
                    "sab": sab, "suppress": suppress,
                    "raw_rxcui1": raw_rxcui1, "raw_rxcui2": raw_rxcui2,
                }

    stats["relations_written"] = write_jsonl(output_path, records())
    return adjacency, stats


def concept_ref(concepts: dict[str, dict[str, Any]], rxcui: str) -> dict[str, Any]:
    concept = concepts.get(rxcui)
    return {
        "rxcui": rxcui,
        "tty": concept["tty"] if concept else None,
        "name": concept["canonical_name"] if concept else None,
    }


def strength_from(attrs: dict[str, Any]) -> dict[str, Any] | None:
    display = attrs.get("RXN_STRENGTH") or attrs.get("RXN_AVAILABLE_STRENGTH")
    numerator_value = attrs.get("RXN_BOSS_STRENGTH_NUM_VALUE")
    numerator_unit = attrs.get("RXN_BOSS_STRENGTH_NUM_UNIT")
    denominator_value = attrs.get("RXN_BOSS_STRENGTH_DENOM_VALUE")
    denominator_unit = attrs.get("RXN_BOSS_STRENGTH_DENOM_UNIT")
    if not any((display, numerator_value, numerator_unit, denominator_value, denominator_unit)):
        return None
    return {
        "display": display, "numerator_value": numerator_value,
        "numerator_unit": numerator_unit, "denominator_value": denominator_value,
        "denominator_unit": denominator_unit,
        "source": "RXNSAT_BOSS" if numerator_value or numerator_unit else "RXNSAT",
    }


def candidate_priority(tty: str, index_tier: str) -> int:
    """Lower is preferred; historical is always the final fallback."""
    if index_tier == "historical":
        return 3 if tty in CANDIDATE_TTYS else 99
    if tty in PRIMARY_PRODUCT_TTYS:
        return 0
    if tty in PACK_TTYS:
        return 1
    if tty in SUPPORT_TTYS:
        return 2
    return 99


def pack_item_quantity(pack_name: str, item_name: str) -> int | float | None:
    """Read an item count from an RxNorm canonical pack name.

    RXNREL ``contains`` identifies the product but carries no count.  RxNorm's
    canonical pack name normally encodes it as ``24 (product name)`` or
    ``1 (15 ML) (product name)``.  Return None when that structure cannot be
    matched exactly rather than guessing from unrelated numbers in the name.
    """
    start = pack_name.casefold().find(item_name.casefold())
    if start < 0:
        return None
    match = PACK_QUANTITY_RE.search(pack_name[:start])
    if not match:
        return None
    value = float(match.group(1))
    return int(value) if value.is_integer() else value


def structured_component(
    component_id: str, concepts: dict[str, dict[str, Any]],
    adjacency: dict[str, list[tuple[str, str]]], attrs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolve ingredient/strength, including branded-to-generic inheritance."""
    outgoing: defaultdict[str, list[str]] = defaultdict(list)
    for rela, target in adjacency.get(component_id, []):
        outgoing[rela].append(target)

    ingredient_ids = [
        value for value in outgoing.get("has_ingredient", [])
        if concepts.get(value, {}).get("tty") != "BN"
    ]
    precise_ids = outgoing.get("has_precise_ingredient", [])
    strength = strength_from(attrs.get(component_id, {}))

    # SBDC/SBDF point to their generic SCDC/SCDF through tradename_of.  The
    # branded node itself commonly carries only the BN relation.
    for generic_id in outgoing.get("tradename_of", []):
        generic_out: defaultdict[str, list[str]] = defaultdict(list)
        for rela, target in adjacency.get(generic_id, []):
            generic_out[rela].append(target)
        if not ingredient_ids:
            ingredient_ids = [
                value for value in generic_out.get("has_ingredient", [])
                if concepts.get(value, {}).get("tty") != "BN"
            ]
        if not precise_ids:
            precise_ids = generic_out.get("has_precise_ingredient", [])
        if strength is None:
            strength = strength_from(attrs.get(generic_id, {}))
        if ingredient_ids and (strength is not None or concepts[component_id]["tty"] in {"SCDF", "SBDF"}):
            break

    return {
        "component_rxcui": component_id,
        "component_tty": concepts[component_id]["tty"],
        "component_name": concepts[component_id]["canonical_name"],
        "ingredient": concept_ref(concepts, ingredient_ids[0]) if ingredient_ids else None,
        "precise_ingredient": concept_ref(concepts, precise_ids[0]) if precise_ids else None,
        "strength": strength,
    }


def enrich_concept(
    concept: dict[str, Any], concepts: dict[str, dict[str, Any]],
    adjacency: dict[str, list[tuple[str, str]]], attrs: dict[str, dict[str, Any]],
    semantic_types: dict[str, list[dict[str, str]]], output_ttys: set[str],
) -> dict[str, Any]:
    rxcui = concept["rxcui"]
    outgoing: defaultdict[str, list[str]] = defaultdict(list)
    for rela, target in adjacency.get(rxcui, []):
        outgoing[rela].append(target)

    components = outgoing.get("consists_of", [])
    clinical_components = [
        structured_component(component_id, concepts, adjacency, attrs)
        for component_id in components
    ]

    # Component/form support concepts do not have a ``consists_of`` child:
    # their ingredient and strength live on the concept itself.  Keep this
    # structure in rxnorm_clean for reranking, not in every embedding row.
    if not clinical_components and concept["tty"] in {
        "SCDC", "SCDF", "SCDFP", "SCDG", "SCDGP",
        "SBDC", "SBDF", "SBDFP", "SBDG", "SBDGP",
    }:
        clinical_components.append(structured_component(rxcui, concepts, adjacency, attrs))

    dose_forms = [concept_ref(concepts, value) for value in outgoing.get("has_dose_form", [])]
    branded = outgoing.get("has_tradename", [])
    generic = outgoing.get("tradename_of", [])
    brand_names = [
        value for value in outgoing.get("has_ingredient", [])
        if concepts.get(value, {}).get("tty") == "BN"
    ]
    pack_item_ids = outgoing.get("contains", [])
    pack_items = [
        {
            "product_rxcui": value,
            "quantity": pack_item_quantity(
                concept["canonical_name"], concepts[value]["canonical_name"]
            ),
        }
        for value in pack_item_ids
    ]
    own_attrs = attrs.get(rxcui, {})
    human = own_attrs.get("RXN_HUMAN_DRUG")
    vet = own_attrs.get("RXN_VET_DRUG")
    tier = "product" if concept["tty"] in PRODUCT_TTYS else "support"
    active = concept["_active"]
    index_tier = tier if active else "historical"
    return {
        "rxcui": rxcui, "tty": concept["tty"], "canonical_name": concept["canonical_name"],
        "prescribable_name": concept["prescribable_name"], "names": concept["names"],
        "clinical_components": clinical_components, "dose_forms": dose_forms,
        "dose_form_groups": [],
        "brand": concept_ref(concepts, brand_names[0]) if brand_names else None,
        "generic_product_rxcuis": generic, "branded_product_rxcuis": branded,
        "pack": {"is_pack": concept["tty"] in {"GPCK", "BPCK"}, "items": pack_items},
        "qualifiers": {
            "quantity": own_attrs.get("RXN_QUANTITY"),
            "qualitative_distinctions": own_attrs.get("RXN_QUALITATIVE_DISTINCTION", []),
        },
        "semantic_types": semantic_types.get(rxcui, []),
        "status": {
            "active": active, "historical": not active,
            "suppress_values": sorted({a["suppress"] for a in concept["_atoms"]}),
            "prescribable": concept["_prescribable"],
            "human_drug": human not in {None, "", "NO", "No", "N"},
            "veterinary_drug": vet not in {None, "", "NO", "No", "N"},
            "activated_date": own_attrs.get("RXN_ACTIVATED"),
            "obsoleted_date": own_attrs.get("RXN_OBSOLETED"),
        },
        "retrieval": {
            "index_tier": index_tier,
            "output_eligible": concept["tty"] in output_ttys,
            "candidate_priority": candidate_priority(concept["tty"], index_tier),
        },
    }


def build_history(raw_dir: Path, concepts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    history: dict[str, dict[str, Any]] = {}
    cui_path = raw_dir / "RXNCUI.RRF"
    with cui_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = fields(line)
            if len(row) < 5:
                raise ValueError(f"Malformed RXNCUI row {number}")
            old, current = row[0], row[4]
            record = history.setdefault(old, {
                "old_rxcui": old, "history_status": "retired", "current_rxcuis": [],
                "archived_names": [], "first_seen": None, "last_seen": None,
                "sources": ["RXNCUI.RRF"],
            })
            if current and current != old and current not in record["current_rxcuis"]:
                record["current_rxcuis"].append(current)
                record["history_status"] = "remapped"

    archive_path = raw_dir / "RXNATOMARCHIVE.RRF"
    with archive_path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            row = fields(line)
            if len(row) < 16:
                raise ValueError(f"Malformed RXNATOMARCHIVE row {number}")
            text, created, updated, lat, old, sab, tty, merged = (
                row[2], row[4], row[5], row[8], row[12], row[13], row[14], row[15]
            )
            if lat != "ENG" or sab != "RXNORM" or not old or not text:
                continue
            record = history.setdefault(old, {
                "old_rxcui": old, "history_status": "archived", "current_rxcuis": [],
                "archived_names": [], "first_seen": created or None, "last_seen": updated or None,
                "sources": [],
            })
            if "RXNATOMARCHIVE.RRF" not in record["sources"]:
                record["sources"].append("RXNATOMARCHIVE.RRF")
            key = lexical_text(text)
            if key not in {lexical_text(a["text"]) for a in record["archived_names"]}:
                record["archived_names"].append({"text": clean_text(text), "tty": tty, "suppress": "O"})
            if merged and merged != old and merged not in record["current_rxcuis"]:
                record["current_rxcuis"].append(merged)
                record["history_status"] = "remapped"
            if created and (record["first_seen"] is None or created < record["first_seen"]):
                record["first_seen"] = created
            if updated and (record["last_seen"] is None or updated > record["last_seen"]):
                record["last_seen"] = updated
    return sorted(history.values(), key=lambda row: int(row["old_rxcui"]))


def embedding_rows(clean_rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for concept in clean_rows:
        for index, name in enumerate(concept["names"]):
            yield {
                "term_id": f"{concept['rxcui']}|{name['name_type']}|{index}",
                "rxcui": concept["rxcui"], "text": name["text"],
                "embedding_text": name["text"], "lexical_text": name["normalized_text"],
                "term_type": name["name_type"], "source_tty": name["source_tty"],
                "concept_tty": concept["tty"],
                "index_tier": concept["retrieval"]["index_tier"],
                "output_eligible": concept["retrieval"]["output_eligible"],
                "candidate_priority": concept["retrieval"]["candidate_priority"],
                "active": concept["status"]["active"],
            }


def history_embedding_rows(
    history_rows: Iterable[dict[str, Any]], output_ttys: set[str],
) -> Iterable[dict[str, Any]]:
    """Flatten archived names without silently remapping their gold RXCUI."""
    for record in history_rows:
        for index, name in enumerate(record["archived_names"]):
            text = name["text"]
            tty = name.get("tty") or "UNKNOWN"
            yield {
                "term_id": f"{record['old_rxcui']}|historical|{index}",
                "rxcui": record["old_rxcui"], "text": text,
                "embedding_text": text, "lexical_text": lexical_text(text),
                "term_type": "historical", "source_tty": tty, "concept_tty": tty,
                "index_tier": "historical",
                "output_eligible": tty in output_ttys,
                "candidate_priority": candidate_priority(tty, "historical"),
                "active": False,
                "current_rxcuis": record["current_rxcuis"],
            }


def deduplicated_embedding_rows(
    clean_rows: Iterable[dict[str, Any]], history_rows: Iterable[dict[str, Any]],
    output_ttys: set[str], stats: Counter[str] | None = None,
) -> Iterable[dict[str, Any]]:
    """Emit unique terms in product > support > historical priority order."""
    clean_rows = list(clean_rows)
    history_rows = list(history_rows)
    seen: set[tuple[str, str]] = set()
    counters = stats if stats is not None else Counter()

    active_rows = list(embedding_rows(clean_rows))
    sources = (
        ("product", (row for row in active_rows if row["index_tier"] == "product")),
        ("support", (row for row in active_rows if row["index_tier"] == "support")),
        ("historical", (row for row in active_rows if row["index_tier"] == "historical")),
        ("historical", history_embedding_rows(history_rows, output_ttys)),
    )
    for tier, rows in sources:
        for row in rows:
            key = (row["rxcui"], row["lexical_text"])
            if key in seen:
                counters["embedding_terms_deduplicated"] += 1
                counters[f"embedding_terms_deduplicated_{tier}"] += 1
                continue
            seen.add(key)
            counters[f"embedding_terms_{row['index_tier']}"] += 1
            counters[f"embedding_terms_output_eligible_{str(row['output_eligible']).lower()}"] += 1
            counters[f"embedding_terms_candidate_priority_{row['candidate_priority']}"] += 1
            yield row


def parse_output_ttys(value: str) -> set[str]:
    result = {item.strip().upper() for item in value.split(",") if item.strip()}
    if not result:
        raise argparse.ArgumentTypeError("output TTY list cannot be empty")
    invalid = result - CANDIDATE_TTYS
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported candidate TTYs: {sorted(invalid)}"
        )
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    required = ["RXNCONSO.RRF", "RXNREL.RRF", "RXNSAT.RRF", "RXNSTY.RRF", "RXNCUI.RRF", "RXNATOMARCHIVE.RRF"]
    missing = [name for name in required if not (args.raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing RxNorm files: {', '.join(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    concepts, conso_stats = read_concepts(args.raw_dir / "RXNCONSO.RRF")
    semantic = read_semantic_types(args.raw_dir / "RXNSTY.RRF")
    attributes, sat_stats = read_attributes(args.raw_dir / "RXNSAT.RRF")
    adjacency, rel_stats = read_relations(
        args.raw_dir / "RXNREL.RRF", concepts, args.output_dir / "rxnorm_relations.jsonl"
    )
    clean_rows = [
        enrich_concept(concepts[key], concepts, adjacency, attributes, semantic, args.output_ttys)
        for key in sorted(concepts, key=int)
    ]
    history = build_history(args.raw_dir, concepts)
    history_count = write_jsonl(args.output_dir / "rxnorm_history.jsonl", history)
    clean_count = write_jsonl(args.output_dir / "rxnorm_clean.jsonl", clean_rows)
    embedding_stats: Counter[str] = Counter()
    term_count = write_jsonl(
        args.output_dir / "rxnorm_embedding_terms.jsonl",
        deduplicated_embedding_rows(clean_rows, history, args.output_ttys, embedding_stats),
    )
    report = {
        "schema_version": "1.0", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(args.raw_dir.resolve()), "output_ttys": sorted(args.output_ttys),
        "counts": {
            "clean_concepts": clean_count, "embedding_terms": term_count,
            "history_records": history_count, **conso_stats, **sat_stats, **rel_stats,
            **embedding_stats,
        },
        "relation_direction": "semantic RRF: source_rxcui=RXCUI2, target_rxcui=RXCUI1; raw columns retained",
    }
    (args.output_dir / "rxnorm_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-ttys", type=parse_output_ttys,
        default=set(DEFAULT_OUTPUT_TTYS),
        help="Comma-separated candidate TTYs; defaults to product plus support concepts.",
    )
    return parser.parse_args()


def main() -> None:
    report = build(parse_args())
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
