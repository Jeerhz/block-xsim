"""Alignment accuracy, pairwise win rate and absolute score difference of a cell.

With w of a block's m distractors scored strictly below the reference (a tie counts against
it), the accuracy against D distractors drawn at random is C(w, D) / C(m, D), averaged over
every D-subset in closed form. Chance is 1 / (D + 1).
"""
from __future__ import annotations

from collections.abc import Sequence
from math import comb

import numpy as np

from block_xsim.models import CandidateScores, CellMetrics, Distractor, GroupMetrics


def accuracy(w: int, m: int, n_distractors: int) -> float:
    return comb(w, n_distractors) / comb(m, n_distractors)


def cell_metrics(scores: Sequence[CandidateScores], distractors: Sequence[Sequence[Distractor]],
                 source_tokens: Sequence[int], n_distractors: int) -> CellMetrics:
    """The cell of the given blocks; group pairs by first perturbed sentence and, for the
    distractors whose edits share one category, by category."""
    acc, win, diff = [], [], []
    by_pos: dict[str, list[tuple[bool, float]]] = {}
    by_cat: dict[str, list[tuple[bool, float]]] = {}
    for s, ds in zip(scores, distractors):
        gold, neg = s.reference, np.asarray(s.distractors)
        w = int((neg < gold).sum())
        acc.append(accuracy(w, len(neg), n_distractors))
        win.append(w / len(neg))
        diff.append(float(np.abs(gold - neg).mean()))
        for x, d in zip(neg, ds):
            pair = (bool(x < gold), float(abs(gold - x)))
            by_pos.setdefault(str(d.positions[0]), []).append(pair)
            if len(set(d.categories)) == 1:
                by_cat.setdefault(d.categories[0], []).append(pair)

    def group(pairs: list[tuple[bool, float]]) -> GroupMetrics:
        return GroupMetrics(n=len(pairs), pairwise_win=float(np.mean([w for w, _x in pairs])),
                            abs_difference=float(np.mean([x for _w, x in pairs])))

    return CellMetrics(
        n_blocks=len(acc), accuracy=float(np.mean(acc)), pairwise_win=float(np.mean(win)),
        abs_difference=float(np.mean(diff)), mean_source_tokens=float(np.mean(source_tokens)),
        by_position={k: group(v) for k, v in sorted(by_pos.items(), key=lambda x: int(x[0]))},
        by_category={k: group(v) for k, v in sorted(by_cat.items())})


def _mean(values: list[float]) -> float:
    return float(np.mean(values))                  # not sum(): compensated since Python 3.12


def _mean_groups(groups: list[dict[str, GroupMetrics]]) -> dict[str, GroupMetrics]:
    keys = list(dict.fromkeys(k for g in groups for k in g))
    out = {}
    for key in keys:
        cells = [g[key] for g in groups if key in g]
        out[key] = GroupMetrics(n=sum(c.n for c in cells),
                                pairwise_win=_mean([c.pairwise_win for c in cells]),
                                abs_difference=_mean([c.abs_difference for c in cells]))
    return out


def mean_over_langs(cells: list[CellMetrics]) -> CellMetrics:
    """Counts summed, rates averaged with every language weighted equally."""
    return CellMetrics(
        n_blocks=sum(c.n_blocks for c in cells),
        accuracy=_mean([c.accuracy for c in cells]),
        pairwise_win=_mean([c.pairwise_win for c in cells]),
        abs_difference=_mean([c.abs_difference for c in cells]),
        mean_source_tokens=_mean([c.mean_source_tokens for c in cells]),
        by_position=_mean_groups([c.by_position for c in cells]),
        by_category=_mean_groups([c.by_category for c in cells]))


def mean_by_k(by_lang: dict[str, dict[str, CellMetrics]]) -> dict[str, CellMetrics]:
    ks = sorted({k for cells in by_lang.values() for k in cells}, key=int)
    return {k: mean_over_langs([cells[k] for cells in by_lang.values() if k in cells]) for k in ks}
