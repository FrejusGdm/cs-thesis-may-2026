#!/usr/bin/env python3
from __future__ import annotations
"""
Cascade error accumulation reporter.

Tracks per-stage results for a single sample and formats a
report showing where degradation occurs as it compounds.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StageResult:
    name: str
    input: Optional[str]
    output: Optional[str]
    metric_name: Optional[str]
    metric_value: Optional[float]
    extra: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "stage": self.name,
            "input_preview": (self.input or "")[:120],
            "output_preview": (self.output or "")[:120],
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }
        if self.extra:
            d.update(self.extra)
        if self.error:
            d["error"] = self.error
        return d


class AccumulationReport:
    """Accumulates per-stage results and formats a cascade error report."""

    def __init__(self, sample_id: str):
        self.sample_id = sample_id
        self.stages: list[StageResult] = []

    def add(self, stage: StageResult) -> None:
        self.stages.append(stage)

    def print_table(self) -> None:
        width = 72
        print(f"\n{'─' * width}")
        print(f"  Sample: {self.sample_id}")
        print(f"{'─' * width}")
        print(f"{'Stage':<24} {'Metric':<16} {'Output preview'}")
        print(f"{'─' * width}")
        for s in self.stages:
            if s.metric_value is not None:
                metric_str = f"{s.metric_name}={s.metric_value:.1f}"
            elif s.metric_name:
                metric_str = f"{s.metric_name}=—"
            else:
                metric_str = "—"
            preview = (s.output or "")[:30].replace("\n", " ")
            err_flag = " [ERR]" if s.error else ""
            print(f"{s.name:<24} {metric_str:<16} {preview}{err_flag}")
        print(f"{'─' * width}\n")

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "stages": [s.to_dict() for s in self.stages],
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def aggregate_reports(reports: list[AccumulationReport]) -> dict[str, dict]:
    """Mean/min/max per stage across all samples."""
    buckets: dict[str, list[float]] = {}
    for report in reports:
        for stage in report.stages:
            if stage.metric_value is not None:
                buckets.setdefault(stage.name, []).append(stage.metric_value)
    return {
        name: {
            "mean": round(sum(vals) / len(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "n": len(vals),
        }
        for name, vals in buckets.items()
    }


def print_aggregate_summary(summary: dict[str, dict]) -> None:
    print("\n=== Aggregate Error Accumulation ===")
    print(f"{'Stage':<26} {'Mean':>8}  {'Min':>8}  {'Max':>8}  {'N':>5}")
    print("─" * 60)
    for name, stats in summary.items():
        print(
            f"{name:<26} {stats['mean']:>8.2f}  {stats['min']:>8.2f}"
            f"  {stats['max']:>8.2f}  {stats['n']:>5}"
        )
    print()
