"""The two corpora, how to fetch them, and how to cut them into windows.

FLORES+ (dev + devtest, gated on the Hugging Face Hub) has articles of 3 to 5 sentences;
NTREX-128 (Federmann et al., 2022), the English source of newstest2019 translated into 128
languages, has documents of 15 sentences on median, which lets the blocks reach several
hundred tokens.
"""
from __future__ import annotations

import subprocess
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from block_xsim.models import ParallelCorpus, Window

NO_SPACE_LANGS = {"zh", "ja", "th"}


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    splits: tuple[str, ...]
    window: int                                     # sentences per window: the longest block
    max_combos: dict[str, int | None]               # edit set -> every-sentence distractors stored per block


# The caps the published datasets were built with: only FLORES+ xSIM++ (3^5 combinations) is complete.
CORPORA = {
    "flores": CorpusSpec("flores", ("dev", "devtest"), window=5, max_combos={"xsim": None, "minor": 64}),
    "ntrex": CorpusSpec("ntrex", ("test",), window=14, max_combos={"xsim": 64, "minor": 64}),
}

FLORES_PLUS_REPO = "openlanguagedata/flores_plus"
FLORES_PLUS_CODE = {"en": "eng_Latn", "de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn",
                    "ru": "rus_Cyrl", "zh": "cmn_Hans"}
NTREX_RAW = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
NTREX_CODE = {"en": "eng", "de": "deu", "es": "spa", "fr": "fra", "ru": "rus", "zh": "zho-CN"}
NTREX_LINES = 1997


def joiner(lang: str) -> str:
    return "" if lang in NO_SPACE_LANGS else " "


def text(corpus: ParallelCorpus, lang: str, rows: Sequence[int]) -> str:
    return joiner(lang).join(corpus.sentences[lang][r] for r in rows)


def windows(corpus: ParallelCorpus, size: int) -> list[Window]:
    """Non-overlapping runs of `size` consecutive lines of one document."""
    out: list[Window] = []
    i = 0
    while i < corpus.n:
        j = i
        while j < corpus.n and j - i < size and corpus.doc_ids[j] == corpus.doc_ids[i]:
            j += 1
        if j - i == size:
            out.append(Window(rows=list(range(i, j)), doc_id=corpus.doc_ids[i], split=corpus.split))
        i = j
    return out


def fetch_flores_plus(langs: Sequence[str], split: str) -> ParallelCorpus:
    """FLORES+ aligned on the sentence id; the article URL is the document. Needs
    `hf auth login` and the terms of openlanguagedata/flores_plus accepted once."""
    from datasets import load_dataset

    rows_by_lang: dict[str, dict[str, str]] = {}
    url_of: dict[str, str] = {}
    for lang in langs:
        ds = load_dataset(FLORES_PLUS_REPO, FLORES_PLUS_CODE[lang], split=split)
        rows_by_lang[lang] = {str(r["id"]): r["text"] for r in ds}
        for r in ds:
            url_of.setdefault(str(r["id"]), r.get("url") or "")
    shared = set.intersection(*(set(r) for r in rows_by_lang.values()))
    ids = sorted(shared, key=lambda i: (0, int(i)) if i.isdigit() else (1, i))
    return ParallelCorpus(corpus="flores", split=split, langs=list(langs),
                          sentences={lang: [rows[i] for i in ids] for lang, rows in rows_by_lang.items()},
                          sentence_ids=ids, doc_ids=[url_of[i] or i for i in ids])


def fetch_ntrex(langs: Sequence[str]) -> ParallelCorpus:
    """NTREX-128 text from GitHub; the document of each line is the newstest2019 docid that
    sacrebleu echoes (same 1,997 lines in the same order)."""
    def lines(command: list[str]) -> list[str]:
        out = subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()
        if len(out) != NTREX_LINES:
            raise RuntimeError(f"{' '.join(command)}: {len(out)} lines, expected {NTREX_LINES}")
        return out

    def fetch(lang: str) -> list[str]:
        name = "newstest2019-src.eng.txt" if lang == "en" else f"newstest2019-ref.{NTREX_CODE[lang]}.txt"
        with urllib.request.urlopen(f"{NTREX_RAW}/{name}", timeout=60) as r:
            out = r.read().decode("utf-8").splitlines()
        if len(out) != NTREX_LINES:
            raise RuntimeError(f"{name}: {len(out)} lines, expected {NTREX_LINES}")
        return out

    sentences = {lang: fetch(lang) for lang in langs}
    sacrebleu = ["sacrebleu", "-t", "wmt19", "-l", "en-de", "--echo"]
    if "en" in sentences and lines(sacrebleu + ["src"])[:3] != sentences["en"][:3]:
        raise RuntimeError("NTREX English and sacrebleu newstest2019 do not line up")
    return ParallelCorpus(corpus="ntrex", split="test", langs=list(langs), sentences=sentences,
                          sentence_ids=[str(i) for i in range(1, NTREX_LINES + 1)],
                          doc_ids=lines(sacrebleu + ["docid"]))
