#!/usr/bin/env python3
"""Export a LaTeX longtable body for thesis qualitative decodes.

Uses **paper-era** pooled training (10K Tatoeba-seeded + 4K structured conditions)
from ``experiments/results/rebuttal_rerun/`` — not ``april2026`` runs whose
checkpoints may reflect the expanded post-paper corpus (~+9K sentences).

Default inputs: NLLB-600M ``exp1``, seed 42, three conditions matching
the ACL / thesis robustness rows: Rand-10K, Struct-4K only, R10K+S4K.

Run from repo root (default base is ``rebuttal_rerun``; repaired JSONLs set ``--base``, optional ``--caption-style thesis``, matching ``--caption-provenance`` for ``full``)::

  python experiments/tools/export_thesis_decode_table.py \\
    --caption-style thesis \\
    --latex-out PATH/main.tex \\
    --tatoeba-latex-out PATH/tatoeba.tex

See ``experiments/results/README_thesis_qualitative_decode.md`` for provenance.

Thesis builds should pass ``--caption-style thesis`` (reader-safe captions with no filesystem
paths or experiment-repo breadcrumbs). Debug or paper-appendix overlays use ``--caption-style full``
with optional ``--caption-provenance``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path


def _load_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[int(o["idx"])] = o
    return out


def _latex_escape_plain(s: str) -> str:
    # French / column headers; avoid brittle backslash handling in src
    reps = [
        ("\\", "\\textbackslash{}"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("$", "\\$"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("^", "\\textasciicircum{}"),
        ("~", "\\textasciitilde{}"),
    ]
    for a, b in reps:
        s = s.replace(a, b)
    return s


def _latex_escape_adja_inner(s: str) -> str:
    # Content inside \adja{...}; keep minimal
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("&", "\\&")
        .replace("_", "\\_")
        .replace("$", "\\$")
    )


def abbreviate_rand_prediction(
    pred: str,
    *,
    min_repeat_run: int = 3,
    max_spam_tokens_show: int = 6,
    max_tokens_before_length_clip: int = 36,
    length_clip_preview: int = 14,
) -> tuple[str, bool]:
    """Return plain-text prefix suitable for escaping; True if abbreviated."""
    tokens = pred.split()
    if not tokens:
        return "", False

    # Long identical-token run: keep preamble plus several repeats so the PDF shows
    # clear "rambling" (~second line) before [\ldots].
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        run = 1
        while i + run < len(tokens) and tokens[i + run] == tok:
            run += 1
        if run >= min_repeat_run:
            k = min(run, max_spam_tokens_show)
            prefix_tokens = tokens[: i + k]
            return " ".join(prefix_tokens), True
        i += run

    if len(tokens) > max_tokens_before_length_clip:
        return " ".join(tokens[:length_clip_preview]), True

    return pred, False


def format_adja_prediction(raw: str) -> str:
    return rf"\adja{{{_latex_escape_adja_inner(str(raw))}}}"


def format_rand10k_prediction(raw: str) -> str:
    pref, abbreviated = abbreviate_rand_prediction(raw)
    inner = _latex_escape_adja_inner(pref)
    if abbreviated:
        return rf"\adja{{{inner}}} [\ldots]\textsuperscript{{*}}"
    return rf"\adja{{{inner}}}"


def _merge_predictions(
    paths: dict[str, Path],
) -> dict[int, tuple[dict, dict[str, dict]]]:
    maps = {k: _load_jsonl(p) for k, p in paths.items()}
    names = list(paths)
    bases = maps[names[0]]
    for idx in sorted(bases):
        row0 = bases[idx]
        for n in names[1:]:
            r = maps[n].get(idx)
            if r is None:
                raise ValueError(f"Missing idx {idx} in {n}")
            if r["src"] != row0["src"] or r["ref"] != row0["ref"]:
                raise ValueError(f"src/ref mismatch at idx {idx}: {names[0]} vs {n}")
    merged: dict[int, tuple[dict, dict[str, dict]]] = {}
    for idx in sorted(bases):
        per = {n: maps[n][idx] for n in names}
        merged[idx] = (bases[idx], per)
    return merged


def tags_for(src: str) -> set[str]:
    s = src.lower().strip()
    t: set[str] = set()
    if " pas " in s or s.endswith(" pas") or " ne " in s:
        t.add("neg")
    if s.endswith("?") or s.startswith("est-ce que"):
        t.add("q")
    if " j'" in f" {s}" or s.startswith("j'") or " ai " in f" {s}" or " as " in f" {s}":
        t.add("past")
    return t


def select_indices(
    merged: dict[int, tuple[dict, dict[str, dict]]],
    *,
    rng_name: str,
    mix_name: str,
    count: int,
    min_gap_pp: float,
) -> list[int]:
    rng_key, mix_key = rng_name, mix_name

    def all_sorted_by_gap(use_min_gap: float) -> list[tuple[float, set[str], int]]:
        scored: list[tuple[float, set[str], int]] = []
        for idx, (_base, per) in merged.items():
            gap = float(per[mix_key]["sentence_chrfpp"]) - float(per[rng_key]["sentence_chrfpp"])
            if gap < use_min_gap:
                continue
            scored.append((gap, tags_for(per[rng_key]["src"]), idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    scored = all_sorted_by_gap(min_gap_pp)
    if len(scored) < count:
        scored = all_sorted_by_gap(use_min_gap=-1e9)
    # Prefer coverage of curriculum-like French patterns among high-gap lines
    want_tags: Sequence[tuple[str, Callable[[set[str]], bool]]] = [
        ("neg", lambda ts: "neg" in ts),
        ("q", lambda ts: "q" in ts),
        ("past", lambda ts: "past" in ts),
        ("general", lambda ts: True),
    ]

    chosen: list[int] = []
    picked: set[int] = set()

    def take_one(pred: Callable[[set[str]], bool]) -> None:
        nonlocal chosen, picked
        for gap, tg, idx in scored:
            if idx in picked:
                continue
            if pred(tg):
                chosen.append(idx)
                picked.add(idx)
                break

    for _tagname, pred in want_tags:
        if len(chosen) >= count:
            break
        take_one(pred)

    for _, __, idx in scored:
        if len(chosen) >= count:
            break
        if idx not in picked:
            chosen.append(idx)
            picked.add(idx)

    return chosen[:count]


def emit_rows(
    merged: dict[int, tuple[dict, dict[str, dict]]],
    indices: Sequence[int],
    *,
    colnames: Sequence[str],
) -> str:
    rand_key = colnames[0] if colnames else "rand10k"
    lines: list[str] = []
    for i, idx in enumerate(indices):
        base, per = merged[idx]
        src = _latex_escape_plain(str(base["src"]))
        ref = format_adja_prediction(str(base["ref"]))
        preds: list[str] = []
        for c in colnames:
            raw = str(per[c]["pred"])
            preds.append(format_rand10k_prediction(raw) if c == rand_key else format_adja_prediction(raw))
        row_parts = [
            str(i + 1),
            src,
            ref,
        ]
        row_parts.extend(preds)
        lines.append(" & ".join(row_parts) + r" \\")
        lines.append(r"\midrule")
    # drop trailing \midrule
    while lines and lines[-1].strip() == r"\midrule":
        lines.pop()
    return "\n".join(lines)


STRUCTURED_END_IDX_EXCLUSIVE = 455  # aligns with experiments/analysis/recompute_subset_metrics.py


def filter_merged_subset(
    merged: dict[int, tuple[dict, dict[str, dict]]],
    *,
    subset: str,
) -> dict[int, tuple[dict, dict[str, dict]]]:
    """subset: ``all`` | ``tatoeba`` | ``structured``."""
    if subset == "all":
        return dict(merged)
    if subset == "tatoeba":
        return {i: merged[i] for i in merged if i >= STRUCTURED_END_IDX_EXCLUSIVE}
    if subset == "structured":
        return {i: merged[i] for i in merged if i < STRUCTURED_END_IDX_EXCLUSIVE}
    raise ValueError(f"unknown subset {subset!r}")


def build_full_longtable_fragment(
    merged: dict[int, tuple[dict, dict[str, dict]]],
    indices: list[int],
    *,
    caption_style: str,
    caption_provenance: str,
    seed_s: str,
    latex_label: str,
    table_kind: str,
) -> str:
    """``table_kind``: ``combined`` (full held-out) | ``tatoeba`` (idx >= STRUCTURED slice)."""
    rows_tex = emit_rows(merged, indices, colnames=("rand10k", "struct4k", "r10ks4k"))
    hdr = (
        r"\textbf{\#} & French & Reference & "
        r"\textsc{Rand-10K} & \textsc{Struct-4K} & \textsc{R10K+S4K} \\"
    )
    if caption_style == "thesis":
        if table_kind == "combined":
            cap_tex = (
                rf"\caption{{Held-out FR$\to$Adja qualitative examples (\textsc{{nllb-200-600M}}, seed~{seed_s}). "
                r"Columns: models trained only on \textsc{Rand-10K} Tatoeba-seeded pairs; only on "
                r"the structured curriculum (\textsc{Struct-4K}); or on mixed supervision (\textsc{R10K+S4K}). "
                r"Rows are illustrative, chosen among instances with a large sentence-level chrF++ gain for mixed "
                r"versus random-only training (see prose). "
                r"\textsuperscript{*}\textit{Repeated sub-word runs under \textsc{Rand-10K} appear as [\ldots].}}"
            )
            hdr_comment = f"% Auto-generated qualitative decode fragment (NLLB-600M, seed {seed_s}). Thesis caption style.\n"
        else:
            cap_tex = (
                rf"\caption{{FR$\to$Adja qualitative examples on Tatoeba-sourced held-out prompts (\textsc{{nllb-200-600M}}, seed~{seed_s}). "
                r"Column training regimes match Table~\ref{tab:mt-qualitative-rebuttal}. "
                r"Rows illustrate large mixed-over-random sentence-level chrF++ within this broad-coverage slice only (see prose). "
                r"\textsuperscript{*}\textit{Repeated sub-word runs under \textsc{Rand-10K} appear as [\ldots].}}"
            )
            hdr_comment = (
                f"% Auto-generated qualitative decode fragment — Tatoeba-sourced held-out slice "
                f"(NLLB-600M, seed {seed_s}). Thesis caption style.\n"
            )
    else:
        if table_kind == "combined":
            cap_tex = (
                r"\caption{Held-out FR$\to$Adja qualitative outputs (\textsc{nllb-200-600M}, seed~"
                + seed_s
                + r") for \textsc{Rand-10K}, structured-only \textsc{Struct-4K}, and mixed \textsc{R10K+S4K}. "
                r"\textbf{Paper-era pooled data} (\texttt{"
                + caption_provenance
                + r"}; "
                r"same regime as aggregate\_subset\_metrics / ACL tables; excludes expanded post-paper training such as "
                r"\texttt{april2026} FULL checkpoints). "
                r"Rows favour large sentence chrF++ gains (mixed minus random-only); illustrative, not exhaustive. "
                r"\textsuperscript{*}\textit{Repeated sub-word runs under \textsc{Rand-10K} appear as [\ldots]; "
                r"verbatim hypotheses remain in the aligned inference files (\texttt{test\_predictions.jsonl}).}"
                r"}"
            )
        else:
            cap_tex = (
                r"\caption{Tatoeba-slice FR$\to$Adja qualitative outputs (\textsc{nllb-200-600M}, seed~"
                + seed_s
                + r") on \textbf{Tatoeba-sourced held-out idx} $\geq$ "
                + str(STRUCTURED_END_IDX_EXCLUSIVE)
                + r". "
                r"\textbf{Paper-era pooled data} (\texttt{"
                + caption_provenance
                + r"}). "
                r"Rows favour large sentence chrF++ gains (mixed minus random-only) within slice; illustrative. "
                r"\textsuperscript{*}\textit{Repeated sub-word runs under \textsc{Rand-10K} appear as [\ldots]; "
                r"verbatim hypotheses remain in the aligned inference files (\texttt{test\_predictions.jsonl}).}"
                r"}"
            )
        hdr_comment = (
            rf"% Auto-generated qualitative decode fragment (kind={table_kind}, NLLB-600M, seed {seed_s}); "
            rf"provenance \\texttt{{{caption_provenance}}}.\n"
        )
    out = hdr_comment + rf"""
\begin{{footnotesize}}
\begin{{longtable}}{{@{{}}
  >{{\centering\arraybackslash}}p{{0.038\linewidth}}
  L{{0.18\linewidth}}
  L{{0.238\linewidth}}
  L{{0.176\linewidth}}
  L{{0.176\linewidth}}
  L{{0.174\linewidth}}
@{{}}}}
{cap_tex}
\label{{{latex_label}}} \\
\toprule
{hdr}
\midrule
\endfirsthead
\multicolumn{{6}}{{l}}{{\footnotesize\textit{{\tablename~\thetable\, (continued).}}}} \\
\toprule
{hdr}
\midrule
\endhead
\midrule
\endfoot
\bottomrule
\endlastfoot
{rows_tex}
\end{{longtable}}
\end{{footnotesize}}
""".strip() + "\n"
    return out


def default_rebuttal_base() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "experiments" / "results" / "rebuttal_rerun" / "nllb-600m" / "exp1"


def main() -> None:
    p = argparse.ArgumentParser(description="LaTeX longtable rows from aligned test_predictions.jsonl")
    p.add_argument(
        "--seed",
        type=str,
        default="42",
        help="Seed subfolder under each condition directory",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Root exp1 folder (defaults to experiments/results/rebuttal_rerun/nllb-600m/exp1)",
    )
    p.add_argument("--count", type=int, default=18, help="Number of illustrative rows")
    p.add_argument(
        "--min-gap",
        type=float,
        default=12.0,
        help="Minimum sentence chrF++ improvement (mixed - random) for primary pool",
    )
    p.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Write combined held-out \\input-able fragment (historical default)",
    )
    p.add_argument(
        "--tatoeba-latex-out",
        type=Path,
        default=None,
        help='Write Tatoeba-sourced held-out slice qualitative fragment (same layout, idx >= 455)',
    )
    p.add_argument("--only-rows", action="store_true", help="Emit table body rows only (no wrapper)")
    p.add_argument(
        "--caption-provenance",
        type=str,
        default=r"rebuttal\_rerun/nllb-600m/exp1",
        help=(
            "LaTeX path for \\texttt{...} in full-style caption only (--caption-style thesis omits paths)"
        ),
    )
    p.add_argument(
        "--caption-style",
        choices=("full", "thesis"),
        default="full",
        help="thesis=self-contained caption for TCC/thesis PDF; full=paths + aggregation regime note",
    )
    args = p.parse_args()

    base = args.base or default_rebuttal_base()
    seed = args.seed

    colnames = ["rand10k", "struct4k", "r10ks4k"]
    paths = {
        "rand10k": base / "RANDOM-10K" / f"seed{seed}" / "test_predictions.jsonl",
        "struct4k": base / "STRUCTURED-4K-ONLY" / f"seed{seed}" / "test_predictions.jsonl",
        "r10ks4k": base / "RANDOM-10K_STRUCTURED-4K" / f"seed{seed}" / "test_predictions.jsonl",
    }

    missing = [k for k, v in paths.items() if not v.is_file()]
    if missing:
        raise SystemExit(f"Missing prediction files for: {missing} under {base}")

    merged = _merge_predictions(paths)
    seed_s = str(seed)

    if args.only_rows:
        indices = select_indices(
            merged,
            rng_name="rand10k",
            mix_name="r10ks4k",
            count=args.count,
            min_gap_pp=args.min_gap,
        )
        out = emit_rows(merged, indices, colnames=tuple(colnames))
        if args.latex_out:
            args.latex_out.parent.mkdir(parents=True, exist_ok=True)
            args.latex_out.write_text(out, encoding="utf-8")
            print(f"Wrote {args.latex_out}")
        elif args.tatoeba_latex_out:
            raise SystemExit("--only-rows with --tatoeba-latex-out is ambiguous; use only --latex-out.")
        else:
            print(out)
        return

    def write_longtable(path: Path, merged_slice: dict, *, kind: str, label: str) -> None:
        if not merged_slice:
            raise SystemExit(f"Merged slice empty for {kind!r}.")
        idxs = select_indices(
            merged_slice,
            rng_name="rand10k",
            mix_name="r10ks4k",
            count=args.count,
            min_gap_pp=args.min_gap,
        )
        frag = build_full_longtable_fragment(
            merged_slice,
            idxs,
            caption_style=args.caption_style,
            caption_provenance=args.caption_provenance,
            seed_s=seed_s,
            latex_label=label,
            table_kind=kind,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(frag, encoding="utf-8")
        print(f"Wrote {path}")

    planned: list[tuple[Path, dict, str, str]] = []
    if args.latex_out:
        planned.append((args.latex_out, merged, "combined", "tab:mt-qualitative-rebuttal"))
    tat_merged = filter_merged_subset(merged, subset="tatoeba")
    if args.tatoeba_latex_out:
        planned.append((args.tatoeba_latex_out, tat_merged, "tatoeba", "tab:mt-qualitative-tatoeba"))

    if planned:
        for path, mslice, kind, label in planned:
            write_longtable(path, mslice, kind=kind, label=label)
        return

    indices = select_indices(
        merged,
        rng_name="rand10k",
        mix_name="r10ks4k",
        count=args.count,
        min_gap_pp=args.min_gap,
    )
    print(
        build_full_longtable_fragment(
            merged,
            indices,
            caption_style=args.caption_style,
            caption_provenance=args.caption_provenance,
            seed_s=seed_s,
            latex_label="tab:mt-qualitative-rebuttal",
            table_kind="combined",
        )
    )


if __name__ == "__main__":
    main()
