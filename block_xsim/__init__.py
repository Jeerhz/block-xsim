"""Block-xSIM++: find the reference translation of a multi-sentence block among perturbed copies."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"            # corpora, datasets, scores, embedding cache (not committed)
RESULTS_DIR = ROOT / "results"
