"""The five models and the two decision rules.

    score    the metric's own score, COMET(src = English block, mt = candidate), for a
             reference-free checkpoint (CometKiwi)
    cosine   the cosine between the embeddings of the English block and of the candidate

To evaluate another model, add an entry to SCORERS.
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ── encoders: un-normalised sentence embeddings, batch by batch ───────────────
class Encoder(ABC):
    max_length = 512

    def __init__(self, model_id: str, device: str):
        self.model_id, self.device = model_id, device

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...

    def embed(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        """L2-normalised embeddings."""
        import torch
        from tqdm import tqdm
        with torch.no_grad():
            raw = np.concatenate([self.embed_batch(list(texts[s:s + batch_size]))
                                  for s in tqdm(range(0, len(texts), batch_size), leave=False)])
        return raw / np.clip(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12, None)


class HFEncoder(Encoder):
    """A Hugging Face encoder; subclasses pool its output."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel, AutoTokenizer
        super().__init__(model_id, device)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(device)

    def encode(self, texts: list[str]):
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True,
                       max_length=self.max_length).to(self.device)
        return enc, self.model(**enc)

    @staticmethod
    def masked_mean(hidden, mask) -> np.ndarray:
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return ((hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)).cpu().float().numpy()


class MeanPoolEncoder(HFEncoder):
    """Masked mean of the last layer: the untrained XLM-R control."""

    def embed_batch(self, texts):
        enc, out = self.encode(texts)
        return self.masked_mean(out.last_hidden_state, enc["attention_mask"])


class LabseEncoder(HFEncoder):
    """LaBSE: the pooler output (dense + tanh over CLS)."""

    def embed_batch(self, texts):
        return self.encode(texts)[1].pooler_output.cpu().float().numpy()


class E5Encoder(HFEncoder):
    """multilingual-E5: masked mean, with the "query: " prefix the model expects."""

    def embed_batch(self, texts):
        enc, out = self.encode(["query: " + t for t in texts])
        return self.masked_mean(out.last_hidden_state, enc["attention_mask"])


class CometEncoder(Encoder):
    """The sentence embedding a COMET model scores with (layerwise attention, average pooling)."""

    def __init__(self, model_id: str, device: str):
        from comet import download_model, load_from_checkpoint
        super().__init__(model_id, device)
        self.model = load_from_checkpoint(download_model(model_id)).eval().to(device)
        self.model.caching = False                      # fresh batches: no embedding LRU cache

    def embed_batch(self, texts):
        enc = self.model.encoder.prepare_sample(texts)
        tt = enc.get("token_type_ids")
        emb = self.model.get_sentence_embedding(enc["input_ids"].to(self.device),
                                                enc["attention_mask"].to(self.device),
                                                tt.to(self.device) if tt is not None else None)
        return emb.cpu().float().numpy()


# ── scorers ───────────────────────────────────────────────────────────────────
class Scorer(ABC):
    @abstractmethod
    def score(self, sources: list[str], candidates: list[str],
              pairs: list[tuple[int, int]]) -> np.ndarray:
        """One score per (source index, candidate index) pair; higher is a better match."""


class QualityScorer(Scorer):
    """A reference-free COMET model; a reference-based one would be handed the answer."""

    def __init__(self, model_id: str, batch_size: int):
        import torch
        from comet import download_model, load_from_checkpoint
        self.model = load_from_checkpoint(download_model(model_id))
        segments = getattr(self.model.hparams, "input_segments", None)    # CometKiwi: [mt, src]
        reads_reference = ("ref" in segments if segments is not None
                           else "referenceless" not in type(self.model).__name__.lower())
        if reads_reference:
            raise ValueError(f"{model_id} reads a reference: the score rule needs a reference-free model")
        self.batch_size, self.gpus = batch_size, int(torch.cuda.is_available())
        self.cache: dict[tuple[str, str], float] = {}

    def score(self, sources, candidates, pairs):
        todo = sorted({(sources[q], candidates[c]) for q, c in pairs} - self.cache.keys())
        if todo:
            out = self.model.predict([{"src": s, "mt": m} for s, m in todo], batch_size=self.batch_size,
                                     gpus=self.gpus, progress_bar=True)
            self.cache.update(zip(todo, (float(x) for x in out["scores"])))
        return np.array([self.cache[(sources[q], candidates[c])] for q, c in pairs])


class CosineScorer(Scorer):
    """Cosine of the embeddings, cached on disk per (encoder, exact list of texts)."""

    def __init__(self, encoder: Encoder, batch_size: int, cache_dir: Path):
        self.encoder, self.batch_size, self.cache_dir = encoder, batch_size, cache_dir

    def embed(self, texts: list[str]) -> np.ndarray:
        digest = hashlib.md5("\n".join([self.encoder.model_id, *texts]).encode("utf-8")).hexdigest()
        path = self.cache_dir / f"{digest}.npy"
        if path.is_file():
            try:
                return np.load(path)
            except (ValueError, EOFError, OSError) as exc:      # a file cut short: recompute
                logger.warning(f"unreadable embedding cache {path.name}: {exc}")
        out = self.encoder.embed(texts, self.batch_size)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, out)
        return out

    def score(self, sources, candidates, pairs):
        q, c = self.embed(sources), self.embed(candidates)
        qi, ci = (np.array(x, dtype=int) for x in zip(*pairs))
        return np.einsum("ij,ij->i", q[qi], c[ci])


# ── registry ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScorerSpec:
    model_id: str
    encoder: type[Encoder] | None = None            # None: the score rule

    @property
    def rule(self) -> str:
        return "score" if self.encoder is None else "cosine"

    def build(self, batch_size: int, cache_dir: Path) -> Scorer:
        if self.encoder is None:
            return QualityScorer(self.model_id, batch_size)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return CosineScorer(self.encoder(self.model_id, device), batch_size, cache_dir)


SCORERS: dict[str, ScorerSpec] = {
    "cometkiwi": ScorerSpec("Unbabel/wmt22-cometkiwi-da"),
    "comet-da": ScorerSpec("Unbabel/wmt22-comet-da", CometEncoder),
    "xlm-r": ScorerSpec("xlm-roberta-large", MeanPoolEncoder),
    "labse": ScorerSpec("setu4993/LaBSE", LabseEncoder),
    "me5": ScorerSpec("intfloat/multilingual-e5-base", E5Encoder),
}
