"""What the corpora give: blocks kept per length, their tokens, and the sentences each edit applies to.

    python -m block_xsim.corpus_statistics --corpora flores ntrex --edits xsim minor

Per corpus and edit set: for every k, the windows, those with at least D distractors in every
regime, and those of them the encoders or CometKiwi read whole (the protocol of evaluate.py);
per category, the sentences of the corpus that admit the edit (the document entity edit
draws from the whole document). CPU only. Writes results/corpus_statistics.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from block_xsim import RESULTS_DIR
from block_xsim.corpora import CORPORA, CorpusSpec
from block_xsim.files import load_blocks, load_corpus
from block_xsim.models import (BlockCounts, CorpusStatistics, CorpusStatisticsConfig,
                               CorpusStatisticsResult)
from block_xsim.perturber import EDIT_SETS, Perturber
from block_xsim.regimes import BlockGroup, token_counter


def perturbable(spec: CorpusSpec, lang: str, categories, seed: int) -> tuple[int, dict[str, int]]:
    """(sentences of the corpus, category -> sentences that admit the edit)."""
    corpora = [load_corpus(spec.name, split) for split in spec.splits]
    perturber = Perturber.for_corpus([s for c in corpora for s in c.sentences[lang]], lang, seed)
    counts = {c: 0 for c in categories}
    for corpus in corpora:
        documents: dict[str, list[str]] = {}
        for doc_id, sent in zip(corpus.doc_ids, corpus.sentences[lang]):
            documents.setdefault(doc_id, []).append(sent)
        for doc_id, sent in zip(corpus.doc_ids, corpus.sentences[lang]):
            for c in categories:
                counts[c] += perturber.edit(sent, c, documents[doc_id]) is not None
    return sum(c.n for c in corpora), counts


def block_counts(groups: dict[str, BlockGroup], config: CorpusStatisticsConfig,
                 n_tokens) -> BlockCounts:
    """The blocks of one k; `groups` maps each language to its blocks."""
    D, limit = config.n_distractors, config.max_tokens
    enough = {lang: len(g.protocol("cosine", D, 0)) for lang, g in groups.items()}
    target = {lang: float(np.mean([n_tokens(b.reference) for b in g.blocks])) for lang, g in groups.items()}
    return BlockCounts(
        n_windows=sum(len(g.blocks) for g in groups.values()),
        n_enough_distractors=sum(enough.values()),
        n_read_by_encoders=sum(len(g.protocol("cosine", D, limit)) for g in groups.values()),
        n_read_by_cometkiwi=sum(len(g.protocol("score", D, limit)) for g in groups.values()),
        source_tokens=float(np.mean([b.source_n_tokens for g in groups.values() for b in g.blocks])),
        target_tokens=float(np.mean(list(target.values()))),
        target_tokens_by_lang=target, n_enough_by_lang=enough)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", choices=list(CORPORA), default=list(CORPORA))
    ap.add_argument("--edits", nargs="+", choices=list(EDIT_SETS), default=list(EDIT_SETS))
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--n_distractors", type=int, default=5)
    ap.add_argument("--max_combos", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="Distractor samples (as in evaluate.py).")
    ap.add_argument("--perturber_seed", type=int, default=42, help="As in build_blocks.py.")
    ap.add_argument("--output", type=Path, default=RESULTS_DIR / "corpus_statistics.json")
    args = ap.parse_args()

    config = CorpusStatisticsConfig(langs=args.langs, n_distractors=args.n_distractors,
                                    max_combos=args.max_combos, max_tokens=args.max_tokens,
                                    seed=args.seed, perturber_seed=args.perturber_seed)
    result = CorpusStatisticsResult(timestamp=datetime.now(timezone.utc).isoformat(), config=config, corpora={})
    n_tokens = token_counter()
    for name in args.corpora:
        spec = CORPORA[name]
        for edits in args.edits:
            stats = CorpusStatistics(corpus=name, edits=edits, window=spec.window, n_sentences={},
                                     perturbable={}, by_k={})
            for lang in config.langs:
                n, counts = perturbable(spec, lang, EDIT_SETS[edits], config.perturber_seed)
                stats.n_sentences[lang] = n
                for c, v in counts.items():
                    stats.perturbable.setdefault(c, {})[lang] = v
            blocks = {lang: load_blocks(name, edits, lang) for lang in config.langs}
            for k in range(1, spec.window + 1):
                groups = {lang: BlockGroup.build([b for b in bs if b.k == k], config.max_combos,
                                                 config.seed, n_tokens) for lang, bs in blocks.items()}
                stats.by_k[str(k)] = block_counts(groups, config, n_tokens)
            result.corpora[f"{name}_{edits}"] = stats
            result.save(args.output)
            print(f"{name} {edits}: " + " ".join(f"k={k}:{c.n_read_by_encoders}" for k, c in stats.by_k.items()))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
