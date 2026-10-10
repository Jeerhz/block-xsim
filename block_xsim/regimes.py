"""The four error regimes and the blocks the protocol keeps.

All four regimes combine the same per-sentence edits of a block, so they coincide at k = 1:

    every_sentence   one edit in every sentence                       error share 1
    every_2nd        one edit in every second sentence, both offsets  error share 1/2
    every_3rd        one edit in every third sentence, three offsets  error share 1/3
    one_sentence     one edited sentence, at every position           error share 1/k

A block enters the protocol of a scorer when every regime gives it at least D distractors
and the scorer reads every candidate whole, so the regimes are compared on the same blocks.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from block_xsim.corpora import joiner
from block_xsim.models import Block, Distractor

REGIMES = ("every_sentence", "every_2nd", "every_3rd", "one_sentence")
STEP = {"every_sentence": 1, "every_2nd": 2, "every_3rd": 3}
TOKENIZER = "xlm-roberta-large"                  # the tokenizer of every COMET model


def token_counter(name: str = TOKENIZER) -> Callable[[str], int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda text: len(tok(text, add_special_tokens=False)["input_ids"])


def distractors(block: Block, regime: str, max_combos: int | None = None, seed: int = 0) -> list[Distractor]:
    """Every distractor of a regime; beyond `max_combos`, a seeded sample of them in order.
    every_sentence is read from the block, the others are built from its per-sentence edits."""
    jn = joiner(block.lang_pair.split("-")[1])

    def apply(edits: Sequence[tuple[int, str]]) -> str:
        s = list(block.reference_segments)
        for j, t in edits:
            s[j] = t
        return jn.join(s)

    if regime == "every_sentence":
        out = [Distractor(text=d, positions=list(range(block.k)), categories=c)
               for d, c in zip(block.distractors, block.categories)]
    elif regime == "one_sentence":
        out = [Distractor(text=apply([(j, t)]), positions=[j], categories=[c])
               for j, p in enumerate(block.perturbations) for c, t in p.items()]
    else:
        step, out = STEP[regime], []
        for offset in range(step):
            pos = [j for j in range(block.k) if j % step == offset]
            if not pos:                                  # k < step: no sentence at this offset
                continue
            for combo in product(*(list(block.perturbations[j].items()) for j in pos)):
                out.append(Distractor(text=apply([(j, t) for j, (_c, t) in zip(pos, combo)]),
                                      positions=pos, categories=[c for c, _t in combo]))
    if max_combos is not None and len(out) > max_combos:
        # len(regime) is part of the seed the published results were sampled with
        rng = np.random.default_rng([seed, block.window, block.k, len(regime)])
        out = [out[i] for i in sorted(rng.choice(len(out), max_combos, replace=False))]
    return out


def fits(rule: str, source_tokens: int, candidate_tokens: int, max_tokens: int) -> bool:
    """Whether a scorer reads the whole input: an encoder embeds each side alone, CometKiwi
    reads `mt </s></s> src` as one sequence (COMET keeps 5 positions for special tokens)."""
    if max_tokens <= 0:
        return True
    if rule == "score":
        return source_tokens + candidate_tokens <= max_tokens - 5
    return max(source_tokens, candidate_tokens) <= max_tokens - 2


@dataclass
class BlockGroup:
    """The blocks of one (language, k), in window order, with the distractors of every regime."""

    blocks: list[Block]
    distractors: dict[str, list[list[Distractor]]]  # regime -> block -> distractors
    longest: dict[str, list[int]]                   # regime -> block -> tokens of the longest candidate

    @classmethod
    def build(cls, blocks: Sequence[Block], max_combos: int, seed: int,
              n_tokens: Callable[[str], int]) -> BlockGroup:
        blocks = sorted(blocks, key=lambda b: b.window)
        ds = {r: [distractors(b, r, max_combos, seed) for b in blocks] for r in REGIMES}
        longest = {r: [max(n_tokens(t) for t in [b.reference] + [d.text for d in dist])
                       for b, dist in zip(blocks, ds[r])] for r in REGIMES}
        return cls(blocks, ds, longest)

    def protocol(self, rule: str, n_distractors: int, max_tokens: int) -> list[int]:
        """Indices of the blocks with at least D distractors in every regime that the scorer reads whole."""
        return [i for i, b in enumerate(self.blocks)
                if all(len(self.distractors[r][i]) >= n_distractors
                       and fits(rule, b.source_n_tokens, self.longest[r][i], max_tokens) for r in REGIMES)]
