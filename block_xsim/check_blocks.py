"""Structural checks of the built blocks, with no model: run after build_blocks.

    python -m block_xsim.check_blocks --corpus ntrex --edits xsim --langs de fr

For every block: the source and the reference are the official lines in order, the block of
length k+1 extends the block of length k, every per-sentence edit differs from the sentence
(case folded), every distractor of every regime is the reference with its stated edits, and no
two edits or distractors of a block coincide once spaces, case and punctuation are ignored.
Prints the counts and exits with 1 on a violation.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter

from block_xsim.corpora import CORPORA, joiner, text
from block_xsim.files import load_blocks, load_corpus
from block_xsim.models import Block, ParallelCorpus
from block_xsim.perturber import EDIT_SETS
from block_xsim.regimes import REGIMES, distractors

NORMALISE = re.compile(r"[\s.,;:!?»«\"'()]+")
COUNTS = ("blocks", "edits", "distractors")


def normalise(s: str) -> str:
    return NORMALISE.sub("", s).casefold()


def check(blocks: list[Block], corpora: dict[str, ParallelCorpus], lang: str, max_combos: int,
          seed: int) -> Counter:
    """Counts of blocks, edits and distractors, and of every kind of violation found."""
    found = Counter()
    by_window: dict[int, dict[int, Block]] = {}
    jn = joiner(lang)
    for b in blocks:
        by_window.setdefault(b.window, {})[b.k] = b
        c = corpora[b.split]
        found["blocks"] += 1
        if b.source != text(c, "en", b.rows) or b.reference != text(c, lang, b.rows):
            found["text not the official lines"] += 1
        if b.sentence_ids != [c.sentence_ids[r] for r in b.rows]:
            found["sentence ids"] += 1
        for sentence, perts in zip(b.reference_segments, b.perturbations):
            found["edits"] += len(perts)
            found["edit equal to the sentence"] += sum(e.casefold() == sentence.casefold() for e in perts.values())
            found["edits coinciding after normalisation"] += len(perts) - len({normalise(e) for e in perts.values()})
        for regime in REGIMES:
            ds = distractors(b, regime, max_combos, seed)
            found["distractors"] += len(ds)
            found["distractors coinciding after normalisation"] += len(ds) - len({normalise(d.text) for d in ds})
            for d in ds:
                segments = list(b.reference_segments)
                for j, cat in zip(d.positions, d.categories):
                    segments[j] = b.perturbations[j][cat]
                if jn.join(segments) != d.text:
                    found["distractor not the stated edits"] += 1
    for byk in by_window.values():
        for k, b in byk.items():
            longer = byk.get(k + 1)
            if longer and not (longer.reference.startswith(b.reference) and longer.source.startswith(b.source)
                               and longer.rows[:k] == b.rows):
                found["block k+1 does not extend block k"] += 1
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", choices=list(CORPORA), required=True)
    ap.add_argument("--edits", choices=list(EDIT_SETS), required=True)
    ap.add_argument("--langs", nargs="+", default=["de", "es", "fr", "ru"])
    ap.add_argument("--max_combos", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpora = {split: load_corpus(args.corpus, split) for split in CORPORA[args.corpus].splits}
    failed = False
    for lang in args.langs:
        found = check(load_blocks(args.corpus, args.edits, lang), corpora, lang, args.max_combos, args.seed)
        violations = {k: v for k, v in found.items() if v and k not in COUNTS}
        failed |= bool(violations)
        print(f"{args.corpus} {args.edits} {lang}: " + ", ".join(f"{found[k]} {k}" for k in COUNTS) + "; "
              + (", ".join(f"{v} {k}" for k, v in violations.items()) or "no violation"))
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
