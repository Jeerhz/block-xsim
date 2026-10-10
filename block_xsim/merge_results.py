"""Merge evaluation files of one configuration written by parallel jobs.

    python -m block_xsim.merge_results results/parts/ntrex_xsim_*.json --output results/ntrex_xsim.json

The files must share their configuration; a scorer split over files gets its regimes put
back together, and a regime found twice is refused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from block_xsim.models import EvaluationResult


def merge(results: list[EvaluationResult]) -> EvaluationResult:
    merged = results[0].model_copy(deep=True)
    for other in results[1:]:
        if other.config != merged.config:
            raise ValueError(f"configurations differ: {other.config} != {merged.config}")
        for name, cells in other.scorers.items():
            mine = merged.scorers.setdefault(name, cells.model_copy(update={"regimes": {}}))
            clash = mine.regimes.keys() & cells.regimes.keys()
            if clash:
                raise ValueError(f"{name}: regimes {sorted(clash)} found twice")
            mine.regimes.update(cells.regimes)
        merged.timestamp = max(merged.timestamp, other.timestamp)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    merged = merge([EvaluationResult.load(p) for p in args.inputs])
    merged.save(args.output)
    print(f"{len(merged.scorers)} scorers -> {args.output}")


if __name__ == "__main__":
    main()
