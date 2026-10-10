"""Where every data file lives under data/, and how to read it back."""
from __future__ import annotations

from pathlib import Path

from block_xsim import DATA_DIR
from block_xsim.models import Block, BlockScores, ParallelCorpus, read_jsonl


def corpus_path(corpus: str, split: str) -> Path:
    return DATA_DIR / "corpora" / f"{corpus}_{split}.json"


def blocks_path(corpus: str, edits: str, lang: str) -> Path:
    return DATA_DIR / "blocks" / f"{corpus}_{edits}_{lang}.jsonl"


def scores_path(corpus: str, edits: str, scorer: str, regime: str) -> Path:
    return DATA_DIR / "scores" / f"{corpus}_{edits}" / f"{scorer}_{regime}.jsonl"


def scored_blocks_dir() -> Path:
    return DATA_DIR / "scored_blocks"


def embedding_cache_dir() -> Path:
    return DATA_DIR / "embedding_cache"


def _require(path: Path, command: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: run `python -m block_xsim.{command}` first")
    return path


def load_corpus(corpus: str, split: str) -> ParallelCorpus:
    return ParallelCorpus.load(_require(corpus_path(corpus, split), "download_corpora"))


def load_blocks(corpus: str, edits: str, lang: str) -> list[Block]:
    return read_jsonl(_require(blocks_path(corpus, edits, lang), "build_blocks"), Block)


def load_scores(corpus: str, edits: str, scorer: str, regime: str) -> dict[str, BlockScores]:
    path = scores_path(corpus, edits, scorer, regime)
    return {s.id: s for s in read_jsonl(path, BlockScores)} if path.is_file() else {}
