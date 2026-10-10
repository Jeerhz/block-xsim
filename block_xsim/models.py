"""Pydantic models: the data the scripts exchange and the result files they write.

Every model mirrors a file exactly (unknown keys are refused). Keys `k` are strings.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict

M = TypeVar("M", bound=BaseModel)


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JsonFile(Model):
    """A model stored as one JSON file."""

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=1), encoding="utf-8")


def read_jsonl(path: Path, model: type[M]) -> list[M]:
    return [model.model_validate_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(r.model_dump_json() + "\n" for r in rows), encoding="utf-8")


# ── data ──────────────────────────────────────────────────────────────────────
class ParallelCorpus(JsonFile):
    """Line-aligned sentences in several languages: line i has the official id `sentence_ids[i]`
    (FLORES+ `id`, NTREX-128 line number from 1) and belongs to document `doc_ids[i]`."""

    corpus: str
    split: str
    langs: list[str]
    sentences: dict[str, list[str]]
    sentence_ids: list[str]
    doc_ids: list[str]

    @property
    def n(self) -> int:
        return len(self.doc_ids)


class Window(Model):
    """Consecutive lines of one document."""

    rows: list[int]
    doc_id: str
    split: str


class Block(Model):
    """The first k sentences of a window in one language pair: the English source, its
    reference translation, one edit per category for each sentence, and the distractors of
    the every-sentence regime (every combination of one edit per sentence)."""

    id: str                                         # <lang>-w<window>-k<k>
    lang_pair: str                                  # en-de
    window: int                                     # shared by every k and every language
    k: int
    split: str
    doc_id: str
    rows: list[int]                                 # lines of the split
    sentence_ids: list[str]                         # official ids of these lines
    source: str
    source_n_tokens: int                            # XLM-R tokens, special tokens excluded
    reference: str
    reference_segments: list[str]
    perturbations: list[dict[str, str]]             # [j]: category -> sentence j edited
    distractors: list[str]                          # every-sentence regime
    categories: list[list[str]]                     # [d][j]: category of sentence j in distractor d


class Distractor(Model):
    text: str
    positions: list[int]                            # perturbed sentences
    categories: list[str]                           # their edits


class CandidateScores(Model):
    reference: float
    distractors: list[float]                        # in the order of `regimes.distractors`


class BlockScores(CandidateScores):
    """One line of data/scores/<corpus>_<edits>/<scorer>_<regime>.jsonl."""

    id: str


class ScoredBlock(Model):
    """One block of one regime with the scores of every model: the published dataset.
    FLORES+ text is not re-hosted: `source` and `reference` are None and are rebuilt from the
    official dataset with `split` and `sentence_ids`."""

    id: str
    corpus: str
    split: str
    lang_pair: str
    k: int
    doc_id: str
    sentence_ids: list[str]
    regime: str
    source: str | None
    reference: str | None
    distractors: list[Distractor]
    scores: dict[str, CandidateScores]              # scorer -> scores


# ── evaluation results ────────────────────────────────────────────────────────
class GroupMetrics(Model):
    """The (reference, distractor) pairs of one group: a position or a category."""

    n: int
    pairwise_win: float                             # share of pairs the reference wins
    abs_difference: float                           # mean |score(reference) - score(distractor)|


class CellMetrics(Model):
    """One (scorer, regime, language, k) cell, on the blocks of the protocol."""

    n_blocks: int
    accuracy: float                                 # P[reference above D random distractors]
    pairwise_win: float                             # mean over blocks of w / m
    abs_difference: float                           # mean over blocks of mean |ref - distractor|
    mean_source_tokens: float
    by_position: dict[str, GroupMetrics]            # first perturbed sentence
    by_category: dict[str, GroupMetrics]            # distractors whose edits share one category


class RegimeCells(Model):
    by_lang: dict[str, dict[str, CellMetrics]]      # lang -> k -> cell
    mean_by_k: dict[str, CellMetrics]               # counts summed, rates averaged over languages


class ScorerCells(Model):
    rule: Literal["score", "cosine"]
    regimes: dict[str, RegimeCells]


class EvaluationConfig(Model):
    corpus: str
    edits: str
    langs: list[str]
    k_list: list[int]
    n_distractors: int                              # D
    max_combos: int                                 # distractors sampled per block and regime
    max_tokens: int                                 # blocks a scorer would truncate are left out
    seed: int                                       # distractor samples


class EvaluationResult(JsonFile):
    """results/<corpus>_<edits>.json"""

    timestamp: str
    config: EvaluationConfig
    scorers: dict[str, ScorerCells]


# ── corpus statistics ─────────────────────────────────────────────────────────
class BlockCounts(Model):
    """The blocks of one length k, languages pooled."""

    n_windows: int                                  # every sentence admits an edit
    n_enough_distractors: int                       # at least D distractors in every regime
    n_read_by_encoders: int                         # of these: no side over the token limit
    n_read_by_cometkiwi: int                        # of these: source + candidate within it
    source_tokens: float                            # mean
    target_tokens: float                            # mean over languages of the mean reference
    target_tokens_by_lang: dict[str, float]
    n_enough_by_lang: dict[str, int]


class CorpusStatistics(Model):
    corpus: str
    edits: str
    window: int
    n_sentences: dict[str, int]                     # lang -> sentences of the corpus
    perturbable: dict[str, dict[str, int]]          # category -> lang -> sentences that admit it
    by_k: dict[str, BlockCounts]


class CorpusStatisticsConfig(Model):
    langs: list[str]
    n_distractors: int
    max_combos: int
    max_tokens: int
    seed: int
    perturber_seed: int


class CorpusStatisticsResult(JsonFile):
    """results/corpus_statistics.json"""

    timestamp: str
    config: CorpusStatisticsConfig
    corpora: dict[str, CorpusStatistics]            # <corpus>_<edits>
