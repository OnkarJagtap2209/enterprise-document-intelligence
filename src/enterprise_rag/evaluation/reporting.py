"""Deterministic experiment comparison reports."""
from dataclasses import asdict
import json
from pathlib import Path
from .experiment import ExperimentRun

def compare_runs(runs: list[ExperimentRun]) -> dict[str, dict[str, float]]:
    return {run.spec.name: dict(sorted(run.summary.items())) for run in runs}

def write_report(path: str | Path, runs: list[ExperimentRun]) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(compare_runs(runs), indent=2, sort_keys=True) + "\n", encoding="utf-8")
