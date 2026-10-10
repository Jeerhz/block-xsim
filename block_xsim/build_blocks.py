"""Cut the corpora into windows and edit every sentence of the target side.

    python -m block_xsim.build_blocks --corpus flores --edits xsim --langs de es fr ru

Windows are non-overlapping runs of `window` sentences of one document (5 on FLORES+, 14 on
NTREX-128); the block of length k is the window's first k sentences, so every k is measured
on the same windows. Each sentence gets at most one edit per category, and a window with a
sentence that admits none is dropped. The every-sentence distractors of a block are all the
combinations of one edit per sentence, or a seeded sample of 64 of them (CorpusSpec.max_combos:
all but FLORES+ xSIM++); complete, the distractors at k+1 extend those at k.

Writes data/blocks/<corpus>_<edits>_<lang>.jsonl, one Block per (window, k).
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from itertools import product

import numpy as np

from block_xsim.corpora import CORPORA, joiner, text, windows
from block_xsim.files import blocks_path, load_corpus
from block_xsim.models import Block, ParallelCorpus, Window, write_jsonl
from block_xsim.perturber import EDIT_SETS, Perturber
from block_xsim.regimes import token_counter


def make_blocks(corpus: ParallelCorpus, wins: Sequence[Window], first_window: int, lang: str,
                k_list: Sequence[int], perturber: Perturber, categories: Sequence[str],
                n_tokens: Callable[[str], int], max_combos: int | None = None, seed: int = 42,
                pivot: str = "en") -> list[Block]:
    """The blocks of the windows of one split; window ids start at `first_window`."""
    blocks: list[Block] = []
    jn = joiner(lang)
    for w, win in enumerate(wins, start=first_window):
        segments = [corpus.sentences[lang][r] for r in win.rows]
        perts = [perturber.perturbations(s, categories, segments) for s in segments]
        if not all(perts):
            continue
        for k in k_list:
            combos = list(product(*(list(p.items()) for p in perts[:k])))
            if max_combos is not None and len(combos) > max_combos:
                rng = np.random.default_rng([seed, w, k])
                combos = [combos[i] for i in sorted(rng.choice(len(combos), max_combos, replace=False))]
            source = text(corpus, pivot, win.rows[:k])
            blocks.append(Block(
                id=f"{lang}-w{w:03d}-k{k}", lang_pair=f"{pivot}-{lang}", window=w, k=k,
                split=corpus.split, doc_id=win.doc_id, rows=win.rows[:k],
                sentence_ids=[corpus.sentence_ids[r] for r in win.rows[:k]], source=source,
                source_n_tokens=n_tokens(source), reference=jn.join(segments[:k]),
                reference_segments=segments[:k], perturbations=perts[:k],
                distractors=[jn.join(t for _c, t in combo) for combo in combos],
                categories=[[c for c, _t in combo] for combo in combos]))
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(CORPORA), required=True)
    ap.add_argument("--edits", choices=list(EDIT_SETS), required=True)
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    spec = CORPORA[args.corpus]
    corpora = [load_corpus(spec.name, split) for split in spec.splits]
    wins = [windows(c, spec.window) for c in corpora]
    k_list = range(1, spec.window + 1)
    n_tokens = token_counter()
    for lang in args.langs:
        perturber = Perturber.for_corpus([s for c in corpora for s in c.sentences[lang]], lang, args.seed)
        blocks, start = [], 0
        for corpus, w in zip(corpora, wins):
            blocks += make_blocks(corpus, w, start, lang, k_list, perturber, EDIT_SETS[args.edits],
                                  n_tokens, spec.max_combos[args.edits], args.seed)
            start += len(w)
        write_jsonl(blocks_path(spec.name, args.edits, lang), blocks)
        longest = [b for b in blocks if b.k == spec.window]
        edits = Counter(c for b in longest for p in b.perturbations for c in p)
        print(f"en-{lang}: {len(longest)} of {start} windows kept, edits {dict(edits)} "
              f"-> {blocks_path(spec.name, args.edits, lang)}")


if __name__ == "__main__":
    main()
