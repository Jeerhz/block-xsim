import pytest

from block_xsim.models import Block


class FakePerturber:
    """Edits "<sentence>|<category><i>" for the categories listed in `admits`."""

    def __init__(self, admits: dict[str, int]):
        self.admits = admits

    def perturbations(self, sentence, categories, context=()):
        return {c: f"{sentence}|{c}" for c in categories if self.admits.get(c)}


@pytest.fixture
def block() -> Block:
    """Three sentences; the second admits two edits, the others one."""
    perts = [{"number": "A"}, {"causality": "B", "entity": "B2"}, {"number": "C"}]
    combos = [("A", "B", "C"), ("A", "B2", "C")]
    return Block(id="de-w000-k3", lang_pair="en-de", window=0, k=3, split="dev", doc_id="d", rows=[0, 1, 2],
                 sentence_ids=["1", "2", "3"],
                 source="e0 e1 e2", source_n_tokens=3, reference="a b c", reference_segments=["a", "b", "c"],
                 perturbations=perts, distractors=[" ".join(c) for c in combos],
                 categories=[["number", "causality", "number"], ["number", "entity", "number"]])
