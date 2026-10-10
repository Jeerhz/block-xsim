"""The committed results load through their models and keep the protocol's invariants."""
import json

import pytest

from block_xsim import RESULTS_DIR
from block_xsim.metrics import mean_by_k
from block_xsim.models import CorpusStatisticsResult, EvaluationResult
from block_xsim.regimes import REGIMES
from block_xsim.scorers import SCORERS

EVALUATIONS = sorted(p for p in RESULTS_DIR.glob("*_*.json") if p.stem != "corpus_statistics")


def test_every_corpus_and_edit_set_is_there():
    assert [p.stem for p in EVALUATIONS] == ["flores_minor", "flores_xsim", "ntrex_minor", "ntrex_xsim"]


@pytest.mark.parametrize("path", [*EVALUATIONS, RESULTS_DIR / "corpus_statistics.json"], ids=lambda p: p.stem)
def test_files_round_trip(path):
    model = CorpusStatisticsResult if path.stem == "corpus_statistics" else EvaluationResult
    assert model.load(path).model_dump(mode="json") == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", EVALUATIONS, ids=lambda p: p.stem)
def test_regimes_share_their_blocks_and_means_are_recomputable(path):
    result = EvaluationResult.load(path)
    assert set(result.scorers) == set(SCORERS)
    for name, cells in result.scorers.items():
        assert cells.rule == SCORERS[name].rule and set(cells.regimes) == set(REGIMES)
        counts = {r: {(lang, k): c.n_blocks for lang, byk in rc.by_lang.items() for k, c in byk.items()}
                  for r, rc in cells.regimes.items()}
        assert all(c == counts["one_sentence"] for c in counts.values()), name
        for rc in cells.regimes.values():
            assert mean_by_k(rc.by_lang) == rc.mean_by_k
