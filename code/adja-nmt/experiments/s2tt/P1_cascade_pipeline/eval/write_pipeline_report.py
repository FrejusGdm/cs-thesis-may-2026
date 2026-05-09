#!/usr/bin/env python3
from __future__ import annotations
"""
Write thesis-ready Markdown notes for a P1 cascade pipeline run.

This intentionally writes Markdown only. The thesis chapter can later import
the tables, examples, and figure ideas from this report without touching LaTeX
during the live pipeline run.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

_HERE = Path(__file__).resolve().parent
_PIPELINE = _HERE.parent
_SHARED = _PIPELINE.parent.parent / "asr" / "shared"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from eval.stage_metrics import asr_metrics, rtt_metrics  # noqa: E402


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _short(text: Optional[str], n: int = 180) -> str:
    if not text:
        return ""
    one_line = " ".join(str(text).split())
    return one_line if len(one_line) <= n else one_line[: n - 1] + "…"


def _cache_for(run_dir: Path, sample_id: str) -> dict:
    return _load_json(run_dir / f"{sample_id}_cache.json", {})


def _collect_times(results: list[dict]) -> dict[str, dict[str, Optional[float]]]:
    buckets: dict[str, list[float]] = {}
    for result in results:
        for stage in ("asr", "mt_fwd", "response", "mt_back", "tts"):
            elapsed = result.get(stage, {}).get("elapsed_sec")
            if isinstance(elapsed, (int, float)):
                buckets.setdefault(stage, []).append(float(elapsed))
    out: dict[str, dict[str, Optional[float]]] = {}
    for stage, vals in sorted(buckets.items()):
        out[stage] = {
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "max": round(max(vals), 2),
            "n": len(vals),
        }
    return out


def _per_sample_metrics(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        sample_id = result.get("sample_id", "")
        ref = result.get("references", {}).get("transcript")
        asr_out = result.get("asr", {})
        tts_text = result.get("tts", {}).get("text") or result.get("mt_back", {}).get("text")
        rtt_out = result.get("rtt", {})

        asr = asr_metrics(ref, asr_out.get("primary", "")) if ref and asr_out.get("primary") else {}
        rtt_primary = (
            rtt_metrics(tts_text, rtt_out.get("primary", ""))
            if tts_text and rtt_out.get("primary") else {}
        )
        rtt_xlsr = (
            rtt_metrics(tts_text, rtt_out.get("xlsr", ""))
            if tts_text and rtt_out.get("xlsr") else {}
        )
        rtt_whisper = (
            rtt_metrics(tts_text, rtt_out.get("whisper", ""))
            if tts_text and rtt_out.get("whisper") else {}
        )
        rows.append({
            "sample_id": sample_id,
            "error": result.get("error"),
            "asr_cer": asr.get("cer"),
            "asr_wer": asr.get("wer"),
            "rtt_primary_wer": rtt_primary.get("wer"),
            "rtt_primary_cer": rtt_primary.get("cer"),
            "rtt_xlsr_cer": rtt_xlsr.get("cer"),
            "rtt_whisper_cer": rtt_whisper.get("cer"),
        })
    return rows


def _metric_table(rows: list[dict], limit: int = 12) -> str:
    header = (
        "| Sample | ASR CER | ASR WER | RTT WER primary | RTT CER xlsr | RTT CER whisper | Error |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
    )
    body = []
    for row in rows[:limit]:
        body.append(
            f"| `{row['sample_id']}` | {_fmt(row.get('asr_cer'), '%')} | "
            f"{_fmt(row.get('asr_wer'), '%')} | {_fmt(row.get('rtt_primary_wer'), '%')} | "
            f"{_fmt(row.get('rtt_xlsr_cer'), '%')} | {_fmt(row.get('rtt_whisper_cer'), '%')} | "
            f"{_short(row.get('error'), 80) or ''} |"
        )
    return header + "\n".join(body)


def _examples(run_dir: Path, results: list[dict], max_examples: int) -> str:
    chunks: list[str] = []
    successes = [r for r in results if not r.get("error")]
    for result in successes[:max_examples]:
        sample_id = result.get("sample_id", "")
        cache = _cache_for(run_dir, sample_id)
        refs = result.get("references", {})
        chunks.append(f"### `{sample_id}`")
        chunks.append("")
        chunks.append(f"- Audio: `{result.get('audio', '')}`")
        if refs.get("transcript"):
            chunks.append(f"- Reference Adja: `{_short(refs['transcript'], 240)}`")
        if cache.get("asr"):
            asr = cache["asr"]
            chunks.append(f"- ASR primary: `{_short(asr.get('primary'), 240)}`")
            chunks.append(f"- ASR xlsr: `{_short(asr.get('xlsr'), 240)}`")
            chunks.append(f"- ASR whisper: `{_short(asr.get('whisper'), 240)}`")
        if cache.get("mt_fwd"):
            chunks.append(f"- MT Adja->FR: `{_short(cache['mt_fwd'].get('text'), 240)}`")
        if cache.get("response"):
            chunks.append(f"- LLM French response: `{_short(cache['response'].get('text'), 240)}`")
        if cache.get("mt_back"):
            chunks.append(f"- MT FR->Adja: `{_short(cache['mt_back'].get('text'), 240)}`")
        if cache.get("tts"):
            chunks.append(f"- Generated WAV: `{cache['tts'].get('wav', '')}`")
        if cache.get("rtt"):
            rtt = cache["rtt"]
            chunks.append(f"- RTT primary transcript: `{_short(rtt.get('primary') or rtt.get('text'), 240)}`")
            if rtt.get("xlsr"):
                chunks.append(f"- RTT xlsr transcript: `{_short(rtt.get('xlsr'), 240)}`")
            if rtt.get("whisper"):
                chunks.append(f"- RTT whisper transcript: `{_short(rtt.get('whisper'), 240)}`")
        chunks.append("")
    return "\n".join(chunks).strip()


def write_report(run_dir: Path, config_path: Path, output_path: Path, max_examples: int) -> None:
    results = _load_json(run_dir / "results.json", [])
    aggregate = _load_json(run_dir / "aggregate_summary.json", {})
    metrics_json = _load_json(run_dir / "metrics.json", {})
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = _per_sample_metrics(results)
    times = _collect_times(results)

    lines: list[str] = []
    lines.append("# P1 Cascade Pipeline Notes")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Samples processed: {aggregate.get('n_total', len(results))}")
    lines.append(f"- Failed samples: {aggregate.get('n_failed', sum(1 for r in results if r.get('error')))}")
    lines.append(f"- Generated WAV files: {aggregate.get('n_generated_audio', len(metrics_json.get('generated', [])))}")
    lines.append("")

    lines.append("## System Schema")
    lines.append("")
    lines.append("```mermaid")
    lines.append("flowchart LR")
    lines.append('  A["Adja speech WAV"] --> B["ASR: XLS-R CTC + Whisper backup"]')
    lines.append('  B --> C["MT forward: Adja to French NLLB"]')
    lines.append('  C --> D["French LLM response or roundtrip pass-through"]')
    lines.append('  D --> E["MT reverse: French to Adja NLLB"]')
    lines.append('  E --> F["Spark TTS endpoint"]')
    lines.append('  F --> G["Generated Adja WAV"]')
    lines.append('  G --> H["RTT ASR scoring"]')
    lines.append("```")
    lines.append("")

    lines.append("## Models And Services")
    lines.append("")
    lines.append("| Stage | Configured model/service | Mode | Notes |")
    lines.append("|---|---|---|---|")
    lines.append(f"| ASR primary | `{config['asr']['xlsr'].get('model_id')}` | `{config['asr']['xlsr'].get('mode')}` | Feeds downstream MT when `primary=xlsr`. |")
    lines.append(f"| ASR backup | `{config['asr']['whisper'].get('model_id')}` | `{config['asr']['whisper'].get('mode')}` | Logged as second readout. |")
    lines.append(f"| MT Adja->FR | `{config['mt'].get('forward_checkpoint')}` | `{config['mt'].get('mode')}` | Directional checkpoint. |")
    lines.append(f"| MT FR->Adja | `{config['mt'].get('reverse_checkpoint')}` | `{config['mt'].get('mode')}` | Directional checkpoint. |")
    lines.append(f"| Response | `{config['response'].get('model')}` | `{config['response'].get('mode')}` | Used only in `--mode qa`. |")
    lines.append(f"| TTS | `{config['tts'].get('checkpoint')}` | `{config['tts'].get('mode')}` | Spark endpoint writes 16 kHz WAV. |")
    lines.append("")

    lines.append("## Aggregate Results")
    lines.append("")
    lines.append("| Metric | Value | N |")
    lines.append("|---|---:|---:|")
    lines.append(f"| ASR primary CER mean | {_fmt(aggregate.get('asr_primary', {}).get('cer_mean'), '%')} | {aggregate.get('asr_primary', {}).get('n', 0)} |")
    lines.append(f"| ASR primary WER mean | {_fmt(aggregate.get('asr_primary', {}).get('wer_mean'), '%')} | {aggregate.get('asr_primary', {}).get('n', 0)} |")
    lines.append(f"| RTT primary WER mean | {_fmt(aggregate.get('rtt_primary', {}).get('wer_mean'), '%')} | {aggregate.get('rtt_primary', {}).get('n', 0)} |")
    lines.append(f"| RTT primary CER mean | {_fmt(aggregate.get('rtt_primary', {}).get('cer_mean'), '%')} | {aggregate.get('rtt_primary', {}).get('n', 0)} |")
    lines.append(f"| RTT XLS-R CER mean | {_fmt(aggregate.get('rtt_xlsr', {}).get('cer_mean'), '%')} | {aggregate.get('rtt_xlsr', {}).get('n', 0)} |")
    lines.append(f"| RTT Whisper CER mean | {_fmt(aggregate.get('rtt_whisper', {}).get('cer_mean'), '%')} | {aggregate.get('rtt_whisper', {}).get('n', 0)} |")
    lines.append("")

    lines.append("## Stage Latency")
    lines.append("")
    lines.append("| Stage | Mean sec | Median sec | Max sec | N |")
    lines.append("|---|---:|---:|---:|---:|")
    for stage, stat in times.items():
        lines.append(
            f"| `{stage}` | {_fmt(stat.get('mean'))} | {_fmt(stat.get('median'))} | "
            f"{_fmt(stat.get('max'))} | {stat.get('n', 0)} |"
        )
    lines.append("")

    lines.append("## Sample Metrics")
    lines.append("")
    lines.append(_metric_table(rows))
    lines.append("")

    lines.append("## Output Schema")
    lines.append("")
    lines.append("- `results.json`: one object per input sample, with top-level `asr`, `mt_fwd`, `response`, `mt_back`, `tts`, and `rtt` blocks when those stages run.")
    lines.append("- `<sample_id>_cache.json`: resumable per-sample cache. This is the most useful file for qualitative tracing because it preserves each stage boundary.")
    lines.append("- `generated_audio/*.wav`: synthesized Adja audio from the Spark TTS stage.")
    lines.append("- `metrics.json`: reverse-WER-compatible generated-audio manifest, shaped as `{\"generated\": [{\"file\", \"text\", \"duration_sec\"}]}`.")
    lines.append("- `aggregate_summary.json`: compact aggregate metrics for plotting and thesis tables.")
    lines.append("")

    lines.append("## Qualitative Examples")
    lines.append("")
    lines.append(_examples(run_dir, results, max_examples) or "No successful examples available yet.")
    lines.append("")

    lines.append("## Figure And Thesis Notes")
    lines.append("")
    lines.append("- Architecture figure: use the Mermaid graph above as the content source for a clean Chapter 6 block diagram.")
    lines.append("- Error accumulation figure: plot ASR CER, RTT CER, and RTT WER from `aggregate_summary.json` as a compact bar chart.")
    lines.append("- Latency figure: plot mean stage elapsed seconds from `stage_elapsed_sec_mean`; endpoint cold starts should be annotated separately if they appear.")
    lines.append("- Qualitative table: use 3-5 examples from this report with columns for reference Adja, ASR primary, MT forward, MT back, and RTT transcript.")
    lines.append("- Caveat: full roundtrip has no gold French reference, so MT quality is primarily qualitative unless a reference translation set is added.")
    lines.append("- Caveat: reverse-WER inherits ASR errors; compare against C4v2's own real-speech CER floor rather than treating the number as absolute intelligibility.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote report to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--config", default=str(_PIPELINE / "configs" / "p1_config.yaml"))
    p.add_argument("--output", default="results/pipeline-comparison.md")
    p.add_argument("--max-examples", type=int, default=10)
    args = p.parse_args()
    write_report(Path(args.run_dir), Path(args.config), Path(args.output), args.max_examples)


if __name__ == "__main__":
    main()
