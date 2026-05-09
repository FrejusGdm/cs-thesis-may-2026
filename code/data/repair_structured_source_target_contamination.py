#!/usr/bin/env python3
"""Repair structured French source fields contaminated by Adja target prefixes.

The structured enriched CSV is treated as the canonical row/order source, but
the generated grammar CSVs are used as a clean French catalog. For each row, the
script finds the longest clean French prefix for the same run/module/pronoun/verb
and moves any trailing residue from `french` to the front of `adja_translation`.
Rows that cannot be matched deterministically are written to a review file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "experiments/data/raw/simple-dataset-enriched.csv"
DEFAULT_RUN1_CLEAN = (
    REPO_ROOT / "grammatical_dataset-generation/final-output-run-1/ADJA_FRENCH_FULL_DATASET.csv"
)
DEFAULT_RUN2_CLEAN = (
    REPO_ROOT / "grammatical_dataset-generation/final-output-run-2/ADJA_FRENCH_FULL_DATASET.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "experiments/data/cleaned/simple-dataset-enriched-clean.csv"
DEFAULT_CHANGED = REPO_ROOT / "experiments/data/cleaned/source_target_repair.changed_rows.tsv"
DEFAULT_REVIEW = REPO_ROOT / "experiments/data/cleaned/source_target_repair.needs_review.tsv"
DEFAULT_SUMMARY = REPO_ROOT / "experiments/data/cleaned/source_target_repair.summary.json"
DEFAULT_MANUAL_OVERRIDES = REPO_ROOT / "experiments/data/cleaned/source_target_repair.manual_overrides.tsv"
DEFAULT_ALTERNATIVE_REFS = REPO_ROOT / "experiments/data/cleaned/source_target_repair.alternative_references.tsv"
DEFAULT_SLASH_ALTERNATIVES = REPO_ROOT / "experiments/data/cleaned/source_target_repair.slash_alternatives.tsv"
DEFAULT_SLASH_REVIEW = REPO_ROOT / "experiments/data/cleaned/source_target_repair.slash_review.tsv"

LEADING_NUMBER_RE = re.compile(r"^\s*\d+\.\s+")
SPACE_RE = re.compile(r"\s+")
EMBEDDED_RECORD_RE = re.compile(r"\n\s*\d+\.\s+.+?:\s+.+$", re.DOTALL)
SLASH_PAIR_RULES = [
    ("sɔsɔ", "xɔxɔ", "xɔxɔ", "sɔsɔ"),
    ("sɔ", "xɔ", "xɔ", "sɔ"),
    ("kuku", "xɔxɔ", "xɔxɔ", "kuku"),
    ("ku", "xɔ", "xɔ", "ku"),
    ("baci", "eba", "baci", "eba"),
    ("keke", "ehun", "keke", "ehun"),
]
SLASH_PAIR_RE = [
    (
        canonical,
        alternative,
        re.compile(
            rf"(?<!\S)({re.escape(left)}|{re.escape(right)})\s*/\s*"
            rf"({re.escape(left)}|{re.escape(right)})(?P<suffix>[?.,;:]?)(?!\S)"
        ),
        f"{left}/{right} -> canonical {canonical}",
    )
    for left, right, canonical, alternative in SLASH_PAIR_RULES
]
X_CANONICAL = {"xɔ", "xɔxɔ"}
X_ALTERNATIVE = {"sɔ", "sɔsɔ", "ku", "kuku"}
ADJA_PRONOUN_PREFIXES = {"ŋu", "ɔ", "e", "mi", "wo"}


@dataclass(frozen=True)
class CleanCandidate:
    sentence_id: str
    run: str
    module: str
    pronoun: str
    verb: str
    french: str
    french_norm: str


def collapse_ws(text: str) -> str:
    return SPACE_RE.sub(" ", str(text).strip())


def strip_leading_number(text: str) -> tuple[str, str]:
    match = LEADING_NUMBER_RE.match(str(text))
    if not match:
        return str(text), ""
    return str(text)[match.end():], match.group(0).strip()


def normalize_module(module: str) -> str:
    value = str(module).strip().lower()
    if re.fullmatch(r"m[1-5]", value):
        return value.upper()
    match = re.search(r"module\s*([1-5])", value)
    if match:
        return f"M{match.group(1)}"
    match = re.search(r"\bm([1-5])\b", value)
    if match:
        return f"M{match.group(1)}"
    return str(module).strip()


def normalize_run(run: str | None, fallback: str) -> str:
    value = str(run or "").strip().lower()
    if value in {"run1", "run2"}:
        return value
    return fallback


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header")
        return reader.fieldnames, rows


def read_clean_catalog(run1_path: Path, run2_path: Path) -> dict[tuple[str, str, str, str], list[CleanCandidate]]:
    catalog: dict[tuple[str, str, str, str], list[CleanCandidate]] = defaultdict(list)
    for path, fallback_run in [(run1_path, "run1"), (run2_path, "run2")]:
        _, rows = read_rows(path)
        for row in rows:
            french = collapse_ws(row.get("french", ""))
            if not french:
                continue
            candidate = CleanCandidate(
                sentence_id=row.get("sentence_id", ""),
                run=normalize_run(row.get("run"), fallback_run),
                module=normalize_module(row.get("module", "")),
                pronoun=collapse_ws(row.get("pronoun", "")).lower(),
                verb=collapse_ws(row.get("verb", "")).lower(),
                french=french,
                french_norm=french.lower(),
            )
            key = (candidate.run, candidate.module, candidate.pronoun, candidate.verb)
            catalog[key].append(candidate)

    for key, candidates in catalog.items():
        deduped = {candidate.french_norm: candidate for candidate in candidates}
        catalog[key] = sorted(deduped.values(), key=lambda item: len(item.french_norm), reverse=True)
    return catalog


def candidate_prefix_variants(candidate: CleanCandidate) -> list[tuple[str, int, str]]:
    variants = [(candidate.french_norm, len(candidate.french), "clean prefix")]
    if " ne " in candidate.french_norm and " pas " in candidate.french_norm:
        no_ne = candidate.french_norm.replace(" ne ", " ", 1)
        display_no_ne = candidate.french.replace(" ne ", " ", 1)
        variants.append((no_ne, len(display_no_ne), "clean negation prefix with missing ne"))
    return variants


def collect_matches(
    current: str,
    candidates: list[CleanCandidate],
) -> list[tuple[CleanCandidate, str, str]]:
    current_norm = current.lower()
    matches: list[tuple[CleanCandidate, str, str]] = []
    for candidate in candidates:
        for prefix_norm, prefix_len, match_note in candidate_prefix_variants(candidate):
            if current_norm == prefix_norm:
                matches.append((candidate, "", match_note))
            elif current_norm.startswith(prefix_norm + " "):
                residue = current[prefix_len:].strip()
                matches.append((candidate, residue, match_note))
    return matches


def choose_candidate(
    row: dict[str, str],
    catalog: dict[tuple[str, str, str, str], list[CleanCandidate]],
) -> tuple[str, CleanCandidate | None, str, str, str]:
    raw_french = row.get("french", "")
    without_number, leading_number = strip_leading_number(raw_french)
    current = collapse_ws(without_number)
    current_norm = current.lower()
    key = (
        normalize_run(row.get("run"), ""),
        normalize_module(row.get("module", "")),
        collapse_ws(row.get("pronoun", "")).lower(),
        collapse_ws(row.get("verb", "")).lower(),
    )

    matches = collect_matches(current, catalog.get(key, []))
    match_scope = "strict metadata key"

    if not matches:
        fallback_candidates = [
            candidate
            for candidate_key, candidates in catalog.items()
            if (
                candidate_key[0] == key[0]
                and candidate_key[2] == key[2]
                and candidate_key[3] == key[3]
            )
            for candidate in candidates
        ]
        matches = collect_matches(current, fallback_candidates)
        match_scope = "same run/pronoun/verb fallback"

    if not matches:
        return "needs_review", None, current, "", f"no clean French prefix for key={key}"

    max_len = max(len(candidate.french_norm) for candidate, _, _ in matches)
    best = [
        (candidate, residue, match_note)
        for candidate, residue, match_note in matches
        if len(candidate.french_norm) == max_len
    ]
    distinct = {candidate.french_norm for candidate, _, _ in best}
    if len(distinct) > 1:
        return "needs_review", None, current, "", "ambiguous longest clean French prefix"

    candidate, residue, match_note = best[0]
    if leading_number and not residue and current_norm == candidate.french_norm:
        return "fixed_number_only", candidate, current, "", f"removed leading number {leading_number!r}"
    if residue:
        return (
            "moved_residue",
            candidate,
            current,
            residue,
            f"moved trailing source residue to target prefix ({match_scope}; {match_note})",
        )
    if current != candidate.french:
        return (
            "fixed_whitespace_or_catalog_text",
            candidate,
            current,
            "",
            f"normalized source to clean catalog text ({match_scope}; {match_note})",
        )
    return "unchanged", candidate, current, "", "already matched clean catalog"


def prepend_residue(residue: str, target: str) -> str:
    residue = collapse_ws(residue)
    target = collapse_ws(target)
    if not residue:
        return target
    if not target:
        return residue
    if target.lower().startswith(residue.lower() + " ") or target.lower() == residue.lower():
        return target
    return f"{residue} {target}"


def remove_embedded_record(text: str) -> tuple[str, str]:
    value = str(text)
    match = EMBEDDED_RECORD_RE.search(value)
    if not match:
        return value, ""
    return value[:match.start()].strip(), match.group(0).strip()


def write_tsv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tsv_safe_row(row) for row in rows)


def tsv_safe_value(value: str) -> str:
    return str(value).replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def tsv_safe_row(row: dict[str, str]) -> dict[str, str]:
    return {key: tsv_safe_value(value) for key, value in row.items()}


def read_manual_overrides(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    overrides: dict[str, dict[str, str]] = {}
    for row in rows:
        sentence_id = collapse_ws(row.get("sentence_id", ""))
        if not sentence_id:
            continue
        overrides[sentence_id] = row
    return overrides


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def first_token(text: str) -> str:
    value = collapse_ws(text)
    return value.split(" ", 1)[0] if value else ""


def token_set(text: str) -> set[str]:
    return {token.strip(" ?.,;:") for token in collapse_ws(text).split()}


def prepend_missing_pronoun(left: str, right: str) -> str:
    left_first = first_token(left)
    right_first = first_token(right)
    if left_first in ADJA_PRONOUN_PREFIXES and right_first in X_CANONICAL:
        return f"{left_first} {right}"
    return right


def normalize_slash_alternatives(text: str) -> tuple[str, str, str, str]:
    """Return canonical target, alternative target, rule note, and status."""
    original = collapse_ws(text)
    if "/" not in original:
        return original, "", "", "no_slash"

    canonical = original
    alternative = original
    applied_rules: list[str] = []
    for canonical_token, alternative_token, pattern, rule_note in SLASH_PAIR_RE:
        found = False

        def canonical_repl(match: re.Match[str]) -> str:
            nonlocal found
            found = True
            return f"{canonical_token}{match.group('suffix')}"

        def alternative_repl(match: re.Match[str]) -> str:
            return f"{alternative_token}{match.group('suffix')}"

        canonical = pattern.sub(canonical_repl, canonical)
        alternative = pattern.sub(alternative_repl, alternative)
        if found:
            applied_rules.append(rule_note)

    canonical = collapse_ws(canonical)
    alternative = collapse_ws(alternative)
    if applied_rules and "/" not in canonical and "/" not in alternative and canonical != alternative:
        return canonical, alternative, "; ".join(applied_rules), "normalized"

    if original.count("/") == 1:
        left_raw, right_raw = original.split("/", 1)
        left = collapse_ws(left_raw)
        right = collapse_ws(right_raw)
        if left and right:
            left_tokens = token_set(left)
            right_tokens = token_set(right)
            if left_tokens & X_CANONICAL and right_tokens & X_ALTERNATIVE:
                canonical = left
                alternative = right
                rule = "phrase alternative with xɔ-family canonical on left"
            elif right_tokens & X_CANONICAL and left_tokens & X_ALTERNATIVE:
                canonical = prepend_missing_pronoun(left, right)
                alternative = left
                rule = "phrase alternative with xɔ-family canonical on right"
            else:
                canonical = left
                alternative = right
                rule = "whole-sentence alternative; first phrase canonical"
            if canonical and alternative and "/" not in canonical and "/" not in alternative and canonical != alternative:
                return canonical, alternative, rule, "normalized"

    return original, "", "unknown slash alternative pattern", "needs_review"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--run1-clean", type=Path, default=DEFAULT_RUN1_CLEAN)
    parser.add_argument("--run2-clean", type=Path, default=DEFAULT_RUN2_CLEAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--changed-rows", type=Path, default=DEFAULT_CHANGED)
    parser.add_argument("--needs-review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manual-overrides", type=Path, default=DEFAULT_MANUAL_OVERRIDES)
    parser.add_argument("--alternative-refs", type=Path, default=DEFAULT_ALTERNATIVE_REFS)
    parser.add_argument("--slash-alternatives", type=Path, default=DEFAULT_SLASH_ALTERNATIVES)
    parser.add_argument("--slash-review", type=Path, default=DEFAULT_SLASH_REVIEW)
    parser.add_argument("--dry-run", action="store_true", help="Only print summary; do not write outputs.")
    parser.add_argument(
        "--allow-review",
        action="store_true",
        help="Write outputs even when unmatched rows remain in needs_review.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fieldnames, rows = read_rows(args.input)
    required = {"sentence_id", "module", "french", "adja_translation", "pronoun", "verb", "run"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise SystemExit(f"Missing required columns in {args.input}: {missing}")

    catalog = read_clean_catalog(args.run1_clean, args.run2_clean)
    manual_overrides = read_manual_overrides(args.manual_overrides)
    output_rows: list[dict[str, str]] = []
    changed_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []
    slash_alternative_rows: list[dict[str, str]] = []
    slash_review_rows: list[dict[str, str]] = []
    generated_alternative_refs: list[dict[str, str]] = []
    counts: dict[str, int] = defaultdict(int)

    for line_number, row in enumerate(rows, start=2):
        status, candidate, current_french, residue, reason = choose_candidate(row, catalog)
        counts[status] += 1
        new_row = dict(row)
        old_target = row.get("adja_translation", "")
        cleaned_target, embedded_record = remove_embedded_record(old_target)
        if embedded_record:
            counts["removed_embedded_record_from_target"] += 1
            new_row["adja_translation"] = cleaned_target
            changed_rows.append({
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "status": "removed_embedded_record_from_target",
                "old_french": row.get("french", ""),
                "new_french": row.get("french", ""),
                "moved_residue": "",
                "old_adja_translation": old_target,
                "new_adja_translation": cleaned_target,
                "match_clean_sentence_id": "",
                "reason": f"removed embedded malformed CSV record from target: {collapse_ws(embedded_record)}",
            })

        override = manual_overrides.get(row.get("sentence_id", ""))
        override_applied = False
        if override:
            old_new_french = new_row.get("french", "")
            old_new_target = new_row.get("adja_translation", "")
            override_french = override.get("new_french", "").strip()
            override_target = override.get("new_adja_translation", "").strip()
            if override_french:
                new_row["french"] = override_french
            if override_target:
                new_row["adja_translation"] = override_target
            override_applied = (
                new_row.get("french", "") != old_new_french
                or new_row.get("adja_translation", "") != old_new_target
            )
            if override_applied:
                counts["manual_override"] += 1
                changed_rows.append({
                    "line_number": str(line_number),
                    "sentence_id": row.get("sentence_id", ""),
                    "run": row.get("run", ""),
                    "module": row.get("module", ""),
                    "pronoun": row.get("pronoun", ""),
                    "verb": row.get("verb", ""),
                    "status": "manual_override",
                    "old_french": row.get("french", ""),
                    "new_french": new_row["french"],
                    "moved_residue": "",
                    "old_adja_translation": old_target,
                    "new_adja_translation": new_row["adja_translation"],
                    "match_clean_sentence_id": "",
                    "reason": override.get("note", "manual override"),
                })

        if not override_applied and (status == "needs_review" or candidate is None):
            if current_french != str(row.get("french", "")):
                review_status = "fixed_review_source_whitespace"
                review_reason = "normalized whitespace in review-row source; target unchanged"
                if current_french != collapse_ws(row.get("french", "")):
                    review_status = "fixed_number_in_review_source"
                    review_reason = "removed generated numbering from review-row source; target unchanged"
                counts[review_status] += 1
                new_row["french"] = current_french
                changed_rows.append({
                    "line_number": str(line_number),
                    "sentence_id": row.get("sentence_id", ""),
                    "run": row.get("run", ""),
                    "module": row.get("module", ""),
                    "pronoun": row.get("pronoun", ""),
                    "verb": row.get("verb", ""),
                    "status": review_status,
                    "old_french": row.get("french", ""),
                    "new_french": new_row["french"],
                    "moved_residue": "",
                    "old_adja_translation": old_target,
                    "new_adja_translation": new_row["adja_translation"],
                    "match_clean_sentence_id": "",
                    "reason": review_reason,
                })
            review_rows.append({
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "old_french": row.get("french", ""),
                "new_french": new_row.get("french", ""),
                "old_adja_translation": row.get("adja_translation", ""),
                "new_adja_translation": new_row.get("adja_translation", ""),
                "reason": reason,
            })
        elif not override_applied and status != "unchanged":
            new_row["french"] = candidate.french
            new_row["adja_translation"] = prepend_residue(residue, cleaned_target)
            changed_rows.append({
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "status": status,
                "old_french": row.get("french", ""),
                "new_french": new_row["french"],
                "moved_residue": residue,
                "old_adja_translation": old_target,
                "new_adja_translation": new_row["adja_translation"],
                "match_clean_sentence_id": candidate.sentence_id,
                "reason": reason,
            })

        old_slash_target = new_row.get("adja_translation", "")
        canonical_target, alternative_target, slash_rule, slash_status = normalize_slash_alternatives(old_slash_target)
        if slash_status == "normalized":
            counts["slash_alternative_normalized"] += 1
            new_row["adja_translation"] = canonical_target
            changed_rows.append({
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "status": "slash_alternative_normalized",
                "old_french": row.get("french", ""),
                "new_french": new_row["french"],
                "moved_residue": "",
                "old_adja_translation": old_slash_target,
                "new_adja_translation": canonical_target,
                "match_clean_sentence_id": "",
                "reason": slash_rule,
            })
            slash_row = {
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "french": new_row.get("french", ""),
                "old_adja_translation": old_slash_target,
                "canonical_adja_translation": canonical_target,
                "alternative_adja_translation": alternative_target,
                "rule": slash_rule,
                "status": "normalized",
            }
            slash_alternative_rows.append(slash_row)
            generated_alternative_refs.append({
                "sentence_id": row.get("sentence_id", ""),
                "french": new_row.get("french", ""),
                "canonical_adja": canonical_target,
                "alternative_adja": alternative_target,
                "note": f"Generated from slash alternative normalization: {slash_rule}",
            })
            if review_rows and review_rows[-1].get("line_number") == str(line_number):
                review_rows[-1]["new_adja_translation"] = canonical_target
        elif slash_status == "needs_review":
            counts["slash_alternative_needs_review"] += 1
            slash_review_rows.append({
                "line_number": str(line_number),
                "sentence_id": row.get("sentence_id", ""),
                "run": row.get("run", ""),
                "module": row.get("module", ""),
                "pronoun": row.get("pronoun", ""),
                "verb": row.get("verb", ""),
                "french": new_row.get("french", ""),
                "adja_translation": old_slash_target,
                "reason": slash_rule,
            })
        output_rows.append(new_row)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "manual_overrides": str(args.manual_overrides),
        "alternative_refs": str(args.alternative_refs),
        "slash_alternatives": str(args.slash_alternatives),
        "slash_review": str(args.slash_review),
        "manual_override_count": len(manual_overrides),
        "row_count": len(rows),
        "changed_count": len(changed_rows),
        "needs_review_count": len(review_rows),
        "slash_alternative_count": len(slash_alternative_rows),
        "slash_review_count": len(slash_review_rows),
        "status_counts": dict(sorted(counts.items())),
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0 if not review_rows and not slash_review_rows else 2
    if slash_review_rows:
        write_tsv(args.slash_review, slash_review_rows, [
            "line_number", "sentence_id", "run", "module", "pronoun", "verb",
            "french", "adja_translation", "reason",
        ])
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote slash review file: {args.slash_review}", file=sys.stderr)
        print("Refusing to write cleaned CSV while slash_review rows remain.", file=sys.stderr)
        return 3
    if review_rows and not args.allow_review:
        write_tsv(args.needs_review, review_rows, list(review_rows[0].keys()))
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote review file: {args.needs_review}", file=sys.stderr)
        print("Refusing to write cleaned CSV while needs_review rows remain. Use --allow-review to override.", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    changed_fields = [
        "line_number", "sentence_id", "run", "module", "pronoun", "verb", "status",
        "old_french", "new_french", "moved_residue", "old_adja_translation",
        "new_adja_translation", "match_clean_sentence_id", "reason",
    ]
    review_fields = [
        "line_number", "sentence_id", "run", "module", "pronoun", "verb",
        "old_french", "new_french", "old_adja_translation", "new_adja_translation", "reason",
    ]
    write_tsv(args.changed_rows, changed_rows, changed_fields)
    write_tsv(args.needs_review, review_rows, review_fields)
    slash_fields = [
        "line_number", "sentence_id", "run", "module", "pronoun", "verb", "french",
        "old_adja_translation", "canonical_adja_translation", "alternative_adja_translation",
        "rule", "status",
    ]
    write_tsv(args.slash_alternatives, slash_alternative_rows, slash_fields)
    write_tsv(args.slash_review, slash_review_rows, [
        "line_number", "sentence_id", "run", "module", "pronoun", "verb",
        "french", "adja_translation", "reason",
    ])
    existing_alternative_refs = read_tsv_rows(args.alternative_refs)
    merged_refs: dict[tuple[str, str, str], dict[str, str]] = {}
    for ref in existing_alternative_refs + generated_alternative_refs:
        sentence_id = ref.get("sentence_id", "")
        canonical = ref.get("canonical_adja", "")
        alternative = ref.get("alternative_adja", "")
        if not sentence_id or not canonical or not alternative:
            continue
        merged_refs[(sentence_id, canonical, alternative)] = {
            "sentence_id": sentence_id,
            "french": ref.get("french", ""),
            "canonical_adja": canonical,
            "alternative_adja": alternative,
            "note": ref.get("note", ""),
        }
    write_tsv(args.alternative_refs, list(merged_refs.values()), [
        "sentence_id", "french", "canonical_adja", "alternative_adja", "note",
    ])
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if not review_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
