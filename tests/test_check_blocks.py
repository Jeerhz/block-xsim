"""check_blocks finds nothing on well-formed blocks and reports a case-only edit."""

from block_xsim.check_blocks import check
from block_xsim.models import Block, ParallelCorpus


def corpus_and_blocks():
    c = ParallelCorpus(corpus="ntrex", split="test", langs=["en", "de"], sentence_ids=["1", "2"], doc_ids=["d", "d"],
                       sentences={"en": ["e0", "e1"], "de": ["s0", "s1"]})
    perts = [{"number": "S0", "entity": "x0"}, {"number": "x1"}]
    blocks = [Block(id=f"de-w000-k{k}", lang_pair="en-de", window=0, k=k, split="test", doc_id="d", rows=[0, 1][:k],
                    sentence_ids=["1", "2"][:k], source=" ".join(["e0", "e1"][:k]), source_n_tokens=k,
                    reference=" ".join(["s0", "s1"][:k]), reference_segments=["s0", "s1"][:k],
                    perturbations=perts[:k],
                    distractors=[" ".join(ts) for ts in ([("S0",), ("x0",)] if k == 1 else [("S0", "x1"), ("x0", "x1")])],
                    categories=[["number"], ["entity"]] if k == 1 else [["number", "number"], ["entity", "number"]])
              for k in (1, 2)]
    return {"test": c}, blocks


def test_well_formed_blocks_pass_except_the_case_only_edit():
    corpora, blocks = corpus_and_blocks()
    found = check(blocks, corpora, "de", 64, 0)
    assert found["blocks"] == 2 and found["edits"] == 2 + 3 and found["distractors"] > 0
    assert found["edit equal to the sentence"] == 2          # "S0" differs from "s0" by case only, in both blocks
    assert not {k for k, v in found.items() if v} - {"blocks", "edits", "distractors", "edit equal to the sentence"}


def test_a_wrong_distractor_is_reported():
    corpora, blocks = corpus_and_blocks()
    blocks[1].distractors[0] = "S0 s1"                         # claims two edits, carries one
    assert check(blocks, corpora, "de", 64, 0)["distractor not the stated edits"] == 1
