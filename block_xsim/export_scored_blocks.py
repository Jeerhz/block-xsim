"""Join the blocks, their distractors and the scores of every model into one dataset.

    python -m block_xsim.export_scored_blocks results/flores_*.json results/ntrex_*.json
    python -m block_xsim.export_scored_blocks results/*_*.json --push <user>/block-xsim

One ScoredBlock per (block, regime) scored by at least one model, written to
data/scored_blocks/<corpus>_<edits>/en-<lang>.jsonl with a dataset card. FLORES+ text is not
re-hosted: its blocks keep only the perturbed distractors and the official sentence ids that
rebuild the source and the reference from openlanguagedata/flores_plus.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from block_xsim.files import load_blocks, load_scores, scored_blocks_dir
from block_xsim.models import CandidateScores, EvaluationResult, ScoredBlock, write_jsonl
from block_xsim.regimes import REGIMES, distractors

REHOSTED = {"ntrex"}                             # corpora whose source and reference are included

CARD = """\
---
license: cc-by-sa-4.0
language: [en, de, es, fr, ru]
task_categories:
- translation
tags:
- machine-translation-evaluation
- xsim++
configs:
{configs}
---

# Block-xSIM++ scored blocks

Blocks of k consecutive sentences of FLORES+ (dev + devtest, k = 1..5) and NTREX-128
(k = 1..14) with the distractors of four error regimes and the score each model gives the
reference and every distractor: the CometKiwi score, and the cosine of the COMET-DA encoder,
XLM-R, LaBSE and multilingual-E5 to the English block. Built with
https://github.com/Jeerhz/block-xsim.

| field | |
|---|---|
| `id` | `<lang>-w<window>-k<k>`: the block of length k of a window |
| `corpus`, `split`, `lang_pair`, `k`, `doc_id` | where the block comes from |
| `sentence_ids` | official ids of its sentences (FLORES+ `id`, NTREX-128 line number from 1) |
| `regime` | `every_sentence`, `every_2nd`, `every_3rd`, `one_sentence` (error share 1, 1/2, 1/3, 1/k) |
| `source`, `reference` | the English block and its translation; `null` for FLORES+ (see below) |
| `distractors` | the reference with edited sentences: `text`, edited `positions`, edit `categories` |
| `scores` | model -> `reference` score and `distractors` scores, in the order of `distractors`; a model is absent from a block it would truncate |

## FLORES+ source and reference

FLORES+ asks that its text not be re-hosted, so its blocks carry the sentence ids instead:

```python
from datasets import load_dataset

def flores_text(lang_code, split, ids):          # lang_code: eng_Latn, deu_Latn, ...
    rows = {{str(r["id"]): r["text"] for r in load_dataset("openlanguagedata/flores_plus", lang_code, split=split)}}
    return " ".join(rows[i] for i in ids)

source = flores_text("eng_Latn", row["split"], row["sentence_ids"])
reference = flores_text("deu_Latn", row["split"], row["sentence_ids"])    # lang_pair en-de
```

Do not train on this data if you evaluate with it. FLORES+ and NTREX-128 are CC BY-SA 4.0.
"""


def scored_blocks(result: EvaluationResult, lang: str) -> list[ScoredBlock]:
    c = result.config
    blocks = load_blocks(c.corpus, c.edits, lang)
    with_text = c.corpus in REHOSTED
    rows = []
    for regime in REGIMES:
        scores = {name: load_scores(c.corpus, c.edits, name, regime) for name in result.scorers}
        for b in blocks:
            mine = {name: CandidateScores(reference=s[b.id].reference, distractors=s[b.id].distractors)
                    for name, s in scores.items() if b.id in s}
            if not mine:
                continue
            ds = distractors(b, regime, c.max_combos, c.seed)
            if any(len(s.distractors) != len(ds) for s in mine.values()):
                raise ValueError(f"{b.id} {regime}: the scores do not match the blocks, re-run evaluate")
            rows.append(ScoredBlock(id=b.id, corpus=c.corpus, split=b.split, lang_pair=b.lang_pair, k=b.k,
                                    doc_id=b.doc_id, sentence_ids=b.sentence_ids, regime=regime,
                                    source=b.source if with_text else None,
                                    reference=b.reference if with_text else None,
                                    distractors=ds, scores=mine))
    return rows


def write_card(out: Path) -> None:
    files = sorted(out.glob("*/*.jsonl"))
    configs = "\n".join(f"- config_name: {f.parent.name}-{f.stem}\n  data_files:\n  - split: test\n"
                        f"    path: {f.parent.name}/{f.name}" for f in files)
    (out / "README.md").write_text(CARD.format(configs=configs), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--push", metavar="REPO_ID", default=None, help="Upload to this Hugging Face dataset.")
    ap.add_argument("--private", action="store_true", help="Make a new pushed repository private.")
    args = ap.parse_args()

    out = scored_blocks_dir()
    for path in args.results:
        result = EvaluationResult.load(path)
        for lang in result.config.langs:
            rows = scored_blocks(result, lang)
            dest = out / f"{result.config.corpus}_{result.config.edits}" / f"en-{lang}.jsonl"
            write_jsonl(dest, rows)
            print(f"{len(rows)} scored blocks -> {dest}")
    write_card(out)
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.push, repo_type="dataset", folder_path=str(out))
        print(f"https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
