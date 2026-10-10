"""Fetch FLORES+ and NTREX-128 once into data/corpora/.

    python -m block_xsim.download_corpora --corpora flores ntrex --langs en de es fr ru

FLORES+ is gated: run `hf auth login` and accept the terms of openlanguagedata/flores_plus.
NTREX-128 needs `sacrebleu` on the PATH for the document ids.
"""
from __future__ import annotations

import argparse

from block_xsim.corpora import CORPORA, fetch_flores_plus, fetch_ntrex
from block_xsim.files import corpus_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpora", nargs="+", choices=list(CORPORA), default=list(CORPORA))
    ap.add_argument("--langs", nargs="+", default=["en", "de", "es", "fr", "ru"])
    args = ap.parse_args()

    for name in args.corpora:
        for split in CORPORA[name].splits:
            corpus = fetch_flores_plus(args.langs, split) if name == "flores" else fetch_ntrex(args.langs)
            corpus.save(corpus_path(name, split))
            print(f"{name} {split}: {corpus.n} sentences, {len(set(corpus.doc_ids))} documents "
                  f"-> {corpus_path(name, split)}")


if __name__ == "__main__":
    main()
