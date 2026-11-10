"""The models and their decision rules.

    score      the metric's own score of (source block, candidate), for a reference-free
               metric: CometKiwi, MetricX-24 in QE mode
    cosine     the cosine between the embeddings of the English block and of the candidate
    reference  a reference-based metric (BLEU): the candidate is compared with the reference
               block, which then scores best against itself; only the score difference is
               informative

To evaluate another model, add an entry to SCORERS. DEFAULT_SCORERS are the five of the paper.
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
    rule = "score"

    @abstractmethod
    def score(self, sources: list[str], candidates: list[str], pairs: list[tuple[int, int]],
              references: list[str]) -> np.ndarray:
        """One score per (source index, candidate index) pair; higher is a better match.
        `references[q]` is the reference block of source q."""


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

    def score(self, sources, candidates, pairs, references):
        todo = sorted({(sources[q], candidates[c]) for q, c in pairs} - self.cache.keys())
        if todo:
            out = self.model.predict([{"src": s, "mt": m} for s, m in todo], batch_size=self.batch_size,
                                     gpus=self.gpus, progress_bar=True)
            self.cache.update(zip(todo, (float(x) for x in out["scores"])))
        return np.array([self.cache[(sources[q], candidates[c])] for q, c in pairs])


class MetricXScorer(Scorer):
    """MetricX-24 in QE mode, computed as in google-research/metricx (metricx24/predict.py and
    models.py): the input "source: <src> candidate: <mt>" in the mT5 tokenizer, cut to 1536
    tokens, final EOS removed; one decoder step from token 0; the prediction is the logit of
    <extra_id_10> (id 250089) clamped to [0, 25]. It is an error score, so the score is its negative."""

    TOKEN = 250089

    def __init__(self, model_id: str, batch_size: int, tokenizer_id: str = "google/mt5-xl",
                 max_length: int = 1536):
        import torch
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(tokenizer_id)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto").eval().to(self.device)
        self.batch_size, self.max_length = batch_size, max_length
        self.cache: dict[tuple[str, str], float] = {}
        self.n_truncated = 0

    def predict(self, texts: list[str]) -> list[float]:
        import torch
        from tqdm import tqdm
        ids = [self.tok(t, max_length=self.max_length, truncation=True)["input_ids"] for t in texts]
        self.n_truncated += sum(len(x) == self.max_length for x in ids)
        ids = [x[:-1] for x in ids]
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))          # batches of similar length
        out = [0.0] * len(ids)
        with torch.no_grad():
            for s in tqdm(range(0, len(order), self.batch_size), leave=False):
                batch = order[s:s + self.batch_size]
                width = max(len(ids[i]) for i in batch)
                x = torch.tensor([ids[i] + [self.tok.pad_token_id] * (width - len(ids[i])) for i in batch],
                                 device=self.device)
                mask = (torch.arange(width, device=self.device)[None, :]
                        < torch.tensor([len(ids[i]) for i in batch], device=self.device)[:, None]).long()
                start = torch.zeros((len(batch), 1), dtype=torch.long, device=self.device)
                logits = self.model(input_ids=x, attention_mask=mask, decoder_input_ids=start).logits
                for i, v in zip(batch, logits[:, 0, self.TOKEN].float().clamp(0, 25).tolist()):
                    out[i] = v
        return out

    def score(self, sources, candidates, pairs, references):
        todo = sorted({(sources[q], candidates[c]) for q, c in pairs} - self.cache.keys())
        if todo:
            errors = self.predict([f"source: {s} candidate: {m}" for s, m in todo])
            self.cache.update(zip(todo, (-e for e in errors)))
            if self.n_truncated:
                logger.warning(f"MetricX: {self.n_truncated} inputs cut at {self.max_length} tokens so far")
        return np.array([self.cache[(sources[q], candidates[c])] for q, c in pairs])


class BleuScorer(Scorer):
    """Sentence BLEU of the candidate against the reference block (sacrebleu, 13a tokenisation,
    effective order), on a 0-100 scale."""

    rule = "reference"

    def __init__(self):
        from sacrebleu.metrics import BLEU
        self.bleu = BLEU(effective_order=True)

    def score(self, sources, candidates, pairs, references):
        return np.array([self.bleu.sentence_score(candidates[c], [references[q]]).score for q, c in pairs])


class CosineScorer(Scorer):
    """Cosine of the embeddings, cached on disk per (encoder, exact list of texts)."""

    rule = "cosine"

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

    def score(self, sources, candidates, pairs, references):
        q, c = self.embed(sources), self.embed(candidates)
        qi, ci = (np.array(x, dtype=int) for x in zip(*pairs))
        return np.einsum("ij,ij->i", q[qi], c[ci])


# ── registry ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ScorerSpec:
    kind: type                                      # an Encoder (cosine rule) or a Scorer
    model_id: str = ""

    @property
    def rule(self) -> str:
        return "cosine" if issubclass(self.kind, Encoder) else self.kind.rule

    def build(self, batch_size: int, cache_dir: Path) -> Scorer:
        if issubclass(self.kind, Encoder):
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            return CosineScorer(self.kind(self.model_id, device), batch_size, cache_dir)
        if self.kind is BleuScorer:
            return BleuScorer()
        return self.kind(self.model_id, batch_size)


SCORERS: dict[str, ScorerSpec] = {
    "cometkiwi": ScorerSpec(QualityScorer, "Unbabel/wmt22-cometkiwi-da"),
    "comet-da": ScorerSpec(CometEncoder, "Unbabel/wmt22-comet-da"),
    "xlm-r": ScorerSpec(MeanPoolEncoder, "xlm-roberta-large"),
    "labse": ScorerSpec(LabseEncoder, "setu4993/LaBSE"),
    "me5": ScorerSpec(E5Encoder, "intfloat/multilingual-e5-base"),
    "metricx-qe": ScorerSpec(MetricXScorer, "google/metricx-24-hybrid-xl-v2p6-bfloat16"),
    "bleu": ScorerSpec(BleuScorer),
}
DEFAULT_SCORERS = ("cometkiwi", "comet-da", "xlm-r", "labse", "me5")
