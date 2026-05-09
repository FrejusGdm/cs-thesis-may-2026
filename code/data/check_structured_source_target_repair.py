#!/usr/bin/env python3
"""Check the structured source-target repair outputs against the original CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD = REPO_ROOT / "experiments/data/raw/simple-dataset-enriched.csv"
DEFAULT_NEW = REPO_ROOT / "experiments/data/cleaned/simple-dataset-enriched-clean.csv"
DEFAULT_CHANGED = REPO_ROOT / "experiments/data/cleaned/source_target_repair.changed_rows.tsv"
DEFAULT_REVIEW = REPO_ROOT / "experiments/data/cleaned/source_target_repair.needs_review.tsv"

SUSPICIOUS_TAIL_RE = re.compile(
    r"\b(mi|wo|yi|e\s+wa|mi\s+wa|e\s+kuku|e\s+drodro|ŋu|ɔ|ɖɛ|axomɛ|wema|eho)\b",
    re.IGNORECASE,
)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--changed-rows", type=Path, default=DEFAULT_CHANGED)
    parser.add_argument("--needs-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--json", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    old_fields, old_rows = read_csv(args.old)
    new_fields, new_rows = read_csv(args.new)
    changed = read_tsv(args.changed_rows)
    review = read_tsv(args.needs_review)

    errors: list[str] = []
    if old_fields != new_fields:
        errors.append("CSV headers differ")
    if len(old_rows) != len(new_rows):
        errors.append(f"row count differs: old={len(old_rows)} new={len(new_rows)}")

    stable_columns = [name for name in old_fields if name not in {"french", "adja_translation"}]
    changed_by_line = {int(row["line_number"]): row for row in changed if row.get("line_number", "").isdigit()}
    metadata_mismatches = []
    changed_mismatches = []
    suspicious_new = []
    embedded_newlines = []

    for idx, (old, new) in enumerate(zip(old_rows, new_rows), start=2):
        for column in stable_columns:
            if old.get(column, "") != new.get(column, ""):
                metadata_mismatches.append((idx, column, old.get(column, ""), new.get(column, "")))
                break

        change = changed_by_line.get(idx)
        if change:
            if new.get("french", "") != change.get("new_french", ""):
                changed_mismatches.append((idx, "french", change.get("new_french", ""), new.get("french", "")))
            if new.get("adja_translation", "") != change.get("new_adja_translation", ""):
                changed_mismatches.append((
                    idx, "adja_translation", change.get("new_adja_translation", ""), new.get("adja_translation", "")
                ))

        if SUSPICIOUS_TAIL_RE.search(new.get("french", "")):
            suspicious_new.append({
                "line_number": idx,
                "sentence_id": new.get("sentence_id", ""),
                "french": new.get("french", ""),
                "adja_translation": new.get("adja_translation", ""),
            })
        if "\n" in new.get("french", "") or "\n" in new.get("adja_translation", ""):
            embedded_newlines.append({
                "line_number": idx,
                "sentence_id": new.get("sentence_id", ""),
                "french": new.get("french", ""),
                "adja_translation": new.get("adja_translation", ""),
            })

    if metadata_mismatches:
        errors.append(f"metadata mismatches: {len(metadata_mismatches)}")
    if changed_mismatches:
        errors.append(f"changed-row output mismatches: {len(changed_mismatches)}")

    report = {
        "old": str(args.old),
        "new": str(args.new),
        "row_count": len(new_rows),
        "changed_rows": len(changed),
        "needs_review_rows": len(review),
        "metadata_mismatches": len(metadata_mismatches),
        "changed_mismatches": len(changed_mismatches),
        "suspicious_new_french_rows": len(suspicious_new),
        "suspicious_new_french_examples": suspicious_new[:25],
        "embedded_newline_rows": len(embedded_newlines),
        "embedded_newline_examples": embedded_newlines[:25],
        "errors": errors,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
