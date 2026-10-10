"""Blocks, regimes, protocol, metrics and merging on small hand-made inputs."""
from itertools import combinations
from math import comb

import numpy as np
import pytest
from conftest import FakePerturber

from block_xsim.build_blocks import make_blocks
from block_xsim.corpora import windows
from block_xsim.merge_results import merge
from block_xsim.metrics import accuracy, cell_metrics, mean_over_langs
from block_xsim.models import CandidateScores, Distractor, EvaluationConfig, EvaluationResult, ParallelCorpus
from block_xsim.regimes import BlockGroup, distractors, fits

XSIM = ("causality", "entity", "number")


def corpus(n_docs: int, size: int) -> ParallelCorpus:
    doc_ids = [f"d{d}" for d in range(n_docs) for _ in range(size)]
    return ParallelCorpus(corpus="flores", split="dev", langs=["en", "de"], doc_ids=doc_ids,
                          sentence_ids=[str(i + 1) for i in range(len(doc_ids))],
                          sentences={"en": [f"e{i}" for i in range(len(doc_ids))],
                                     "de": [f"s{i}" for i in range(len(doc_ids))]})


def test_windows_do_not_overlap_or_cross_documents():
    c = corpus(2, 5)
    assert [w.rows for w in windows(c, 2)] == [[0, 1], [2, 3], [5, 6], [7, 8]]
    assert windows(c, 6) == []


def test_blocks_combine_every_edit_and_nest_over_k():
    c = corpus(1, 3)
    w = windows(c, 3)
    k1, k2, k3 = make_blocks(c, w, 0, "de", [1, 2, 3], FakePerturber({x: 1 for x in XSIM}), XSIM,
                             lambda t: len(t.split()))
    assert k3.source == "e0 e1 e2" and k3.source_n_tokens == 3 and k3.id == "de-w000-k3"
    assert [len(b.distractors) for b in (k1, k2, k3)] == [3, 9, 27] and len(set(k3.distractors)) == 27
    assert ["entity"] * 3 in k3.categories
    for short, long in ((k1, k2), (k2, k3)):
        assert {d.rsplit(" ", 1)[0] for d in long.distractors} == set(short.distractors)


def test_blocks_sample_combinations_beyond_the_cap():
    c = corpus(1, 3)
    blocks = make_blocks(c, windows(c, 3), 0, "de", [3], FakePerturber({x: 1 for x in XSIM}), XSIM, len,
                         max_combos=5)
    assert len(blocks[0].distractors) == 5


def test_a_window_with_an_unperturbable_sentence_is_dropped():
    c = corpus(1, 2)
    assert make_blocks(c, windows(c, 2), 0, "de", [1, 2], FakePerturber({}), XSIM, len) == []


def test_regimes_combine_the_same_edits(block):
    one = distractors(block, "one_sentence")
    assert [d.text for d in one] == ["A b c", "a B c", "a B2 c", "a b C"]
    assert [d.positions for d in one] == [[0], [1], [1], [2]]
    every = distractors(block, "every_sentence")
    assert [d.text for d in every] == block.distractors and every[0].positions == [0, 1, 2]
    second = distractors(block, "every_2nd")
    assert [(d.text, d.positions) for d in second] == [("A b C", [0, 2]), ("a B c", [1]), ("a B2 c", [1])]
    assert [d.text for d in distractors(block, "every_3rd")] == [d.text for d in one]
    assert len(distractors(block, "one_sentence", max_combos=2, seed=0)) == 2


def test_truncation_rule():
    assert fits("cosine", 500, 510, 512) and not fits("cosine", 500, 511, 512)
    assert fits("score", 250, 257, 512) and not fits("score", 250, 258, 512)
    assert fits("score", 10_000, 10_000, 0)


def test_protocol_keeps_blocks_with_enough_distractors_in_every_regime(block):
    group = BlockGroup.build([block], max_combos=64, seed=0, n_tokens=lambda t: len(t.split()))
    assert group.protocol("cosine", 2, 512) == [0]      # every regime gives at least 2
    assert group.protocol("cosine", 3, 512) == []       # every_sentence gives 2
    assert group.protocol("score", 2, 10) == []         # 3 + 3 > 10 - 5


def test_accuracy_is_the_mean_over_every_subset_of_distractors():
    scores = np.array([0.1, 0.9, 0.2, 0.5])             # reference 0.5: two below, one above, one tie
    w = int((scores < 0.5).sum())
    for D in (1, 2, 3):
        subsets = list(combinations(scores, D))
        assert np.isclose(accuracy(w, 4, D), sum(all(s < 0.5 for s in sub) for sub in subsets) / comb(4, D))


def test_cell_metrics_and_language_mean():
    ds = [Distractor(text="x", positions=[0], categories=["number"]),
          Distractor(text="y", positions=[1], categories=["entity"]),
          Distractor(text="z", positions=[0, 1], categories=["number", "entity"])]
    cell = cell_metrics([CandidateScores(reference=0.9, distractors=[0.5, 0.7, 0.95])], [ds], [30], 2)
    assert cell.n_blocks == 1 and cell.mean_source_tokens == 30
    assert cell.accuracy == pytest.approx(1 / 3) and cell.pairwise_win == pytest.approx(2 / 3)
    assert cell.abs_difference == pytest.approx((0.4 + 0.2 + 0.05) / 3)
    assert cell.by_position["0"].n == 2 and cell.by_position["1"].n == 1
    assert set(cell.by_category) == {"number", "entity"} and cell.by_category["number"].pairwise_win == 1.0
    other = cell.model_copy(update={"n_blocks": 3, "accuracy": 0.0, "by_category": {}})
    mean = mean_over_langs([cell, other])
    assert mean.n_blocks == 4 and mean.accuracy == pytest.approx(1 / 6)
    assert mean.by_category["number"].n == 1 and mean.by_position["0"].n == 4


def test_merge_joins_regimes_and_refuses_duplicates():
    result = EvaluationResult.load("results/flores_xsim.json")
    first, second = result.model_copy(deep=True), result.model_copy(deep=True)
    for name in result.scorers:
        first.scorers[name].regimes = {r: c for r, c in result.scorers[name].regimes.items() if r != "every_2nd"}
        second.scorers[name].regimes = {"every_2nd": result.scorers[name].regimes["every_2nd"]}
    assert merge([first, second]).scorers == result.scorers
    with pytest.raises(ValueError):
        merge([result, second])
    with pytest.raises(ValueError):
        merge([result, result.model_copy(update={"config": EvaluationConfig(
            **{**result.config.model_dump(), "seed": 1})})])


class UppercaseScorer:
    """Scores a candidate by minus its number of capitals: the lower-case reference always wins."""

    def score(self, sources, candidates, pairs):
        return np.array([-sum(ch.isupper() for ch in candidates[c]) for _q, c in pairs], dtype=float)


def test_evaluate_then_export(tmp_path, monkeypatch, block):
    from block_xsim import files
    from block_xsim.evaluate import evaluate_regime
    from block_xsim.export_scored_blocks import scored_blocks
    from block_xsim.models import RegimeCells, ScorerCells, write_jsonl
    from block_xsim.regimes import REGIMES

    monkeypatch.setattr(files, "DATA_DIR", tmp_path)
    blocks = [block, block.model_copy(update={"id": "de-w001-k3", "window": 1})]
    write_jsonl(files.blocks_path("flores", "xsim", "de"), blocks)
    config = EvaluationConfig(corpus="flores", edits="xsim", langs=["de"], k_list=[3], n_distractors=2,
                              max_combos=64, max_tokens=512, seed=0)
    groups = {("de", 3): BlockGroup.build(blocks, 64, 0, lambda t: len(t.split()))}
    regimes: dict[str, RegimeCells] = {}
    for regime in REGIMES:
        regimes[regime], scores = evaluate_regime(UppercaseScorer(), "cosine", groups, regime, config)
        write_jsonl(files.scores_path("flores", "xsim", "stub", regime), scores)
        assert regimes[regime].mean_by_k["3"].n_blocks == 2 and regimes[regime].mean_by_k["3"].accuracy == 1.0
    result = EvaluationResult(timestamp="t", config=config,
                              scorers={"stub": ScorerCells(rule="cosine", regimes=regimes)})
    rows = scored_blocks(result, "de")
    assert len(rows) == 2 * len(REGIMES)
    for row in rows:
        assert len(row.scores["stub"].distractors) == len(row.distractors)
        assert row.source is None and row.reference is None and row.sentence_ids == ["1", "2", "3"]
        assert row.scores["stub"].reference == 0 and all(s < 0 for s in row.scores["stub"].distractors)
