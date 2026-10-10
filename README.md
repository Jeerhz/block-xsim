# Block-xSIM++

Code and results of *Block-xSIM++: The Degradation of CometKiwi on Longer Inputs Is Mainly the
Dilution of the Error Signal*.

xSIM++ asks a model to find the translation of a sentence among copies that carry a small
meaning error. Block-xSIM++ asks the same about **blocks of k consecutive sentences** (k = 1..5
on FLORES+, 1..14 on NTREX-128): given the English block, does the model score the reference
translation above D = 5 distractors? Four regimes place the same edits differently, to tell
the length of the input apart from the share of the text that is wrong:

| regime | edited sentences | error share |
|---|---|---|
| `every_sentence` | all | 1 |
| `every_2nd` | every second sentence | 1/2 |
| `every_3rd` | every third sentence | 1/3 |
| `one_sentence` | one | 1/k |

Two edit sets are applied to the target side with spaCy:

- **xSIM++ edits**: causality, entity, number.
- **Minor edits**: antonym, nearby number, document entity, omission, modality. These edits read fluently, and only the source reveals them.

Five models are compared:

- the CometKiwi score;
- the cosine similarity of the COMET-DA encoder, XLM-R (mean pooled), LaBSE and multilingual-E5.

## Results

`results/` holds one file per corpus and edit set (`flores_xsim.json`, `flores_minor.json`,
`ntrex_xsim.json`, `ntrex_minor.json`) and `corpus_statistics.json`; see
[`results/README.md`](results/README.md). Read them with the core install (pydantic, numpy):

```python
from block_xsim.models import EvaluationResult

result = EvaluationResult.load("results/ntrex_xsim.json")
cell = result.scorers["cometkiwi"].regimes["one_sentence"].mean_by_k["7"]
print(cell.n_blocks, cell.accuracy, cell.abs_difference)
```

The cells hold three measures:

- **Alignment accuracy** is P[the reference scores above D distractors], with chance at 1/(D+1).
- **Pairwise win rate** is the share of distractors that score below the reference.
- **Absolute score difference** is the mean of |score(reference) − score(distractor)|.

Each measure is given overall, by position of the first edited sentence and by edit category, per language and averaged over languages.

## Protocol

- **Windows.** Non-overlapping runs of 5 (FLORES+) or 14 (NTREX-128) sentences of one document.
  The block of length k is the first k sentences of its window, so every k uses the same windows.
  A window is dropped when one of its sentences admits no edit.
- **Edits.** Each sentence gets at most one edit per category, the same in every regime.
  A regime's distractors are its combinations of one edit per edited sentence. Beyond 64 combinations, a seeded sample of 64 is used.
- **Blocks kept.** A block is kept for a model when two conditions hold:
  - every regime gives it at least D = 5 distractors;
  - the model reads every candidate whole, within 512 XLM-R tokens.

  An encoder embeds each side alone, while CometKiwi reads `mt </s></s> src` as one sequence. The four regimes are therefore always compared on the same blocks. k = 1 never has 5 distractors.
- **Accuracy for D distractors.** The accuracy is computed in closed form over every D-subset: C(w, D) / C(m, D), where w of the m distractors score strictly below the reference.

## Install

```bash
pip install -e ".[experiments,dev]"
python -m spacy download de_core_news_sm      # also es_core_news_sm fr_core_news_sm ru_core_news_sm
hf auth login                                 # then accept the terms of openlanguagedata/flores_plus
```

`pip install -e .` alone is enough to read the results and run `pytest`.

## Reproduce

```bash
python -m block_xsim.download_corpora                       # data/corpora/
python -m block_xsim.build_blocks --corpus flores --edits xsim   # data/blocks/, CPU; 4 corpus x edit pairs
python -m block_xsim.evaluate --corpus flores --edits xsim       # results/flores_xsim.json, GPU
python -m block_xsim.corpus_statistics                      # results/corpus_statistics.json, CPU
python -m block_xsim.export_scored_blocks results/flores_*.json results/ntrex_*.json   # data/scored_blocks/
```

On Slurm, submit `slurm/build_blocks.sh`, `slurm/evaluate.sh` and `slurm/corpus_statistics.sh`, with `CORPUS` and `EDITS` set in the environment. Each script states its other variables. For a long run, split `evaluate` over jobs with `--scorers`, `--regimes` and `--output`, then join the files with `python -m block_xsim.merge_results`.

The edits come from spaCy small models. Other model versions may edit other words.

## Layout

| module | role |
|---|---|
| `models.py` | pydantic models of every file: corpora, blocks, scores, results |
| `corpora.py` | FLORES+ and NTREX-128 settings and fetchers, windows |
| `lexicons.py` · `perturber.py` | the edits (spaCy) |
| `regimes.py` | the distractors of each regime, the truncation rule, the blocks kept |
| `metrics.py` | accuracy, pairwise win rate, absolute difference, language means |
| `scorers.py` | the five models; add one to `SCORERS` to evaluate it |
| `files.py` | where every file lives under `data/` |
| `download_corpora.py` · `build_blocks.py` · `evaluate.py` · `merge_results.py` · `corpus_statistics.py` · `export_scored_blocks.py` | the scripts, in pipeline order (`python -m block_xsim.<script> --help`) |

## Data and licences

Code: Apache-2.0. FLORES+ and NTREX-128 are CC BY-SA 4.0.

- `data/` holds the corpora, blocks and per-candidate scores. It is not committed.
- FLORES+ asks that its text not be re-hosted. The exported dataset therefore keeps, for FLORES+ blocks:
  - the perturbed distractors and the scores;
  - the official `split` and `sentence_ids`, which rebuild the source and the reference from `openlanguagedata/flores_plus`.
- NTREX-128 blocks keep their text.
