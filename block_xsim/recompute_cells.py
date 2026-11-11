"""Recompute every cell of a result file from the stored per-candidate scores, without scoring.

    python -m block_xsim.recompute_cells results/flores_*.json results/ntrex_*.json results/metricx_bleu/flores_*.json
    python -m block_xsim.recompute_cells data/rerun_mx/ntrex_xsim_*.json --output results/metricx_bleu/ntrex_xsim.json

data/scores/<corpus>_<edits>/<scorer>_<regime>.jsonl holds exactly the blocks of a scorer's
protocol, with the distractor scores in the order of `regimes.distractors`; the distractors
(their positions and categories) are rebuilt from the blocks with the configuration of the
result file. Every measure is recomputed, and the largest change of each measure the old file
already had is printed. Rewrites each file in place or, with --output, joins files of one
configuration (parallel jobs, as merge_results does) into one.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from block_xsim.files import load_blocks, load_scores
from block_xsim.metrics import cell_metrics, mean_by_k
from block_xsim.models import EvaluationConfig, EvaluationResult, RegimeCells, ScorerCells
from block_xsim.regimes import distractors

CHECKED = ("n_blocks", "accuracy", "pairwise_win", "abs_difference")


def join(paths: list[Path]) -> dict:
    """The raw files (they may predate the measures of the models), one regime at most once."""
    files = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    joined = files[0]
    for other in files[1:]:
        if other["config"] != joined["config"]:
            raise SystemExit(f"configurations differ: {other['config']} != {joined['config']}")
        for name, cells in other["scorers"].items():
            mine = joined["scorers"].setdefault(name, {"rule": cells["rule"], "regimes": {}})
            if mine["regimes"].keys() & cells["regimes"].keys():
                raise SystemExit(f"{name}: a regime is found twice")
            mine["regimes"].update(cells["regimes"])
    return joined


def recompute(paths: list[Path], output: Path) -> None:
    old, path = join(paths), " + ".join(map(str, paths))
    config = EvaluationConfig.model_validate(old["config"])
    blocks = {b.id: b for lang in config.langs for b in load_blocks(config.corpus, config.edits, lang)}
    result = EvaluationResult(timestamp=datetime.now(timezone.utc).isoformat(), config=config, scorers={})
    worst: dict[str, float] = defaultdict(float)
    for name, old_cells in old["scorers"].items():
        cells = ScorerCells(rule=old_cells["rule"], regimes={})
        for regime, old_regime in old_cells["regimes"].items():
            scores = load_scores(config.corpus, config.edits, name, regime)
            if not scores:
                raise SystemExit(f"no scores for {name} {regime} of {config.corpus}_{config.edits}")
            groups = defaultdict(list)                      # (lang, k) -> blocks
            for block_id in scores:
                b = blocks[block_id]
                groups[(b.lang_pair.split("-")[1], str(b.k))].append(b)
            by_lang = {}
            for (lang, k), bs in sorted(groups.items()):
                bs.sort(key=lambda b: b.window)
                ds = [distractors(b, regime, config.max_combos, config.seed) for b in bs]
                for b, d in zip(bs, ds):
                    if len(d) != len(scores[b.id].distractors):
                        raise SystemExit(f"{b.id} {regime}: {len(d)} distractors, {len(scores[b.id].distractors)} scores")
                cell = cell_metrics([scores[b.id] for b in bs], ds, [b.source_n_tokens for b in bs],
                                    config.n_distractors)
                by_lang.setdefault(lang, {})[k] = cell
                before = old_regime["by_lang"].get(lang, {}).get(k)
                if before is None:
                    raise SystemExit(f"{name} {regime} {lang} k={k}: no such cell in {path}")
                for m in CHECKED:
                    worst[(name, m)] = max(worst[(name, m)], abs(getattr(cell, m) - before[m]))
            if {(lang, k) for lang, c in old_regime["by_lang"].items() for k in c} != set(groups):
                raise SystemExit(f"{name} {regime}: the scores do not cover the cells of {path}")
            cells.regimes[regime] = RegimeCells(by_lang=by_lang, mean_by_k=mean_by_k(by_lang))
        result.scorers[name] = cells
    print(path)
    for (name, m), v in sorted(worst.items()):
        print(f"  {name:12s} {m:15s} largest change of a cell {v:.2g}")
    result.save(output)
    print(f"-> {output}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--output", type=Path, default=None, help="Join every input into this file.")
    args = ap.parse_args()
    if args.output:
        recompute(args.results, args.output)
    else:
        for path in args.results:
            recompute([path], path)


if __name__ == "__main__":
    main()
