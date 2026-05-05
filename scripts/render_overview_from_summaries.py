#!/usr/bin/env python3
"""Build overview_metrics.png from four summary_single.json files (no model run)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tg_eval.metrics import plot_summary_bar


def load_summary(path: Path) -> tuple[str, dict]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    bench = obj.get("benchmark") or path.parent.name
    return bench, obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "Figure" / "overview_metrics.png",
        help="Where to write the PNG.",
    )
    parser.add_argument(
        "--exp1-root",
        type=Path,
        default=ROOT / "results" / "experiment 1" / "results",
        help="Directory containing beaver/, toxic/, wildguard/, openai/ each with nested summary.",
    )
    args = parser.parse_args()

    order = ["beaver", "toxic", "wildguard", "openai"]
    summaries: list[tuple[str, dict]] = []
    for name in order:
        candidates = [
            args.exp1_root / name / name / "summary_single.json",
            args.exp1_root / name / "summary_single.json",
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No summary_single.json for {name} under {args.exp1_root}")
        summaries.append(load_summary(path))

    plot_summary_bar(summaries, args.output)
    print("Wrote", args.output.resolve())


if __name__ == "__main__":
    main()
