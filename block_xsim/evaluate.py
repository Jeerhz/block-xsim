"""Find the reference block among D distractors, for every regime and block length.

    python -m block_xsim.evaluate --corpus flores --edits xsim
    python -m block_xsim.evaluate --corpus ntrex --edits xsim --scorers me5 --regimes every_2nd \
        --output results/parts/ntrex_xsim_me5_every_2nd.json          # parallel jobs: merge_results

Each scorer sees, for every (language, k), the blocks of its protocol (regimes.BlockGroup):
at least D distractors in every regime, every candidate read whole. A cell resting on fewer
than MIN_BLOCKS blocks is not reported. Every regime keeps a seeded sample of at most
`max_combos` distractors per block.

Writes results/<corpus>_<edits>.json (EvaluationResult) after every regime, so a stopped job
resumes where it was, and the score of every candidate to
data/scores/<corpus>_<edits>/<scorer>_<regime>.jsonl (export_scored_blocks).
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from block_xsim import RESULTS_DIR
from block_xsim.corpora import CORPORA
from block_xsim.files import embedding_cache_dir, load_blocks, scores_path
from block_xsim.metrics import cell_metrics, mean_by_k
from block_xsim.models import (BlockScores, CellMetrics, EvaluationConfig, EvaluationResult,
                               RegimeCells, ScorerCells, write_jsonl)
from block_xsim.perturber import EDIT_SETS
from block_xsim.regimes import REGIMES, BlockGroup, token_counter
from block_xsim.scorers import SCORERS, Scorer

MIN_BLOCKS = 2


def score_group(scorer: Scorer, group: BlockGroup, regime: str, kept: list[int]) -> list[BlockScores]:
    """The scores of the reference and of the distractors of the kept blocks."""
    sources, candidates, pairs = [], [], []
    for q, i in enumerate(kept):
        texts = [group.blocks[i].reference] + [d.text for d in group.distractors[regime][i]]
        pairs += [(q, len(candidates) + c) for c in range(len(texts))]
        candidates += texts
        sources.append(group.blocks[i].source)
    scores = np.asarray(scorer.score(sources, candidates, pairs), dtype=float).tolist()
    out, s = [], 0
    for i in kept:
        m = len(group.distractors[regime][i])
        out.append(BlockScores(id=group.blocks[i].id, reference=scores[s], distractors=scores[s + 1:s + 1 + m]))
        s += 1 + m
    return out


def evaluate_regime(scorer: Scorer, rule: str, groups: dict[tuple[str, int], BlockGroup], regime: str,
                    config: EvaluationConfig) -> tuple[RegimeCells, list[BlockScores]]:
    by_lang: dict[str, dict[str, CellMetrics]] = {}
    all_scores: list[BlockScores] = []
    for (lang, k), group in groups.items():
        kept = group.protocol(rule, config.n_distractors, config.max_tokens)
        if len(kept) < MIN_BLOCKS:
            continue
        scores = score_group(scorer, group, regime, kept)
        cell = cell_metrics(scores, [group.distractors[regime][i] for i in kept],
                            [group.blocks[i].source_n_tokens for i in kept], config.n_distractors)
        by_lang.setdefault(lang, {})[str(k)] = cell
        all_scores += scores
        print(f"  {regime:14s} {lang} k={k:2d}: {cell.n_blocks:4d} blocks, accuracy {cell.accuracy:.3f}, "
              f"|difference| {cell.abs_difference:.4f}")
    return RegimeCells(by_lang=by_lang, mean_by_k=mean_by_k(by_lang)), all_scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(CORPORA), required=True)
    ap.add_argument("--edits", choices=list(EDIT_SETS), required=True)
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--scorers", nargs="+", choices=list(SCORERS), default=list(SCORERS))
    ap.add_argument("--regimes", nargs="+", choices=REGIMES, default=list(REGIMES),
                    help="Regimes to score; the protocol always looks at all four.")
    ap.add_argument("--n_distractors", type=int, default=5, help="D")
    ap.add_argument("--max_combos", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512, help="XLM-R tokens a scorer reads.")
    ap.add_argument("--seed", type=int, default=0, help="Distractor samples.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--output", type=Path, default=None, help="Default: results/<corpus>_<edits>.json")
    args = ap.parse_args()

    config = EvaluationConfig(corpus=args.corpus, edits=args.edits, langs=args.langs,
                              k_list=list(range(1, CORPORA[args.corpus].window + 1)),
                              n_distractors=args.n_distractors, max_combos=args.max_combos,
                              max_tokens=args.max_tokens, seed=args.seed)
    output = args.output or RESULTS_DIR / f"{args.corpus}_{args.edits}.json"
    result = EvaluationResult(timestamp="", config=config, scorers={})
    if output.is_file():
        result = EvaluationResult.load(output)
        if result.config != config:
            raise SystemExit(f"{output} holds another configuration: give another --output")

    n_tokens = token_counter()
    groups: dict[tuple[str, int], BlockGroup] = {}
    for lang in config.langs:
        blocks = load_blocks(config.corpus, config.edits, lang)
        for k in config.k_list:
            groups[(lang, k)] = BlockGroup.build([b for b in blocks if b.k == k], config.max_combos,
                                                 config.seed, n_tokens)

    for name in args.scorers:
        spec, scorer = SCORERS[name], None
        cells = result.scorers.get(name) or ScorerCells(rule=spec.rule, regimes={})
        for regime in args.regimes:
            if regime in cells.regimes:
                print(f"{name} {regime}: already in {output}")
                continue
            print(f"{name} {regime}")
            scorer = scorer or spec.build(args.batch_size, embedding_cache_dir())
            cells.regimes[regime], scores = evaluate_regime(scorer, spec.rule, groups, regime, config)
            write_jsonl(scores_path(config.corpus, config.edits, name, regime), scores)
            result.scorers[name] = cells
            result.timestamp = datetime.now(timezone.utc).isoformat()
            result.save(output)
        del scorer
        gc.collect()
    print(f"-> {output}")


if __name__ == "__main__":
    main()
