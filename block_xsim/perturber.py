"""Sentence edits driven by spaCy (NER, morphology, dependency parse).

xSIM++ edits (Chen et al., 2023), visible on the target side:
  causality    an antonym, a hedged modal made assertive, or a negation added or removed
               (French "ne ... pas" as a unit, inserted on the finite verb)
  entity       a named entity swapped for one of the same type from the corpus
  number       a number or an ordinal replaced by another one

Minor edits, fluent and plausible on their own; only the source reveals them:
  antonym      a curated directional antonym, no negation
  number_near  one digit moved by +-1, or the neighbouring ordinal
  entity_doc   an entity swapped with another of the same type from the same document
  omission     a short verb-free modifier subtree dropped
  modality     a quantifier or a modal softened

    python -m spacy download de_core_news_sm     # es_core_news_sm, fr_core_news_sm, ru_core_news_sm
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from block_xsim.corpora import joiner
from block_xsim.lexicons import (ANTONYMS, MODAL_BOOST, MODAL_SOFT, ORDINALS, clean_spaces,
                                 match_case, replace_from_lexicon, seeded_rng)

EDIT_SETS: dict[str, tuple[str, ...]] = {
    "xsim": ("causality", "entity", "number"),
    "minor": ("antonym", "number_near", "entity_doc", "omission", "modality"),
}
ATTEMPTS = 6                                     # seeded draws before a sentence is declared unperturbable

SPACY_MODELS = {"en": "en_core_web_sm", "de": "de_core_news_sm", "es": "es_core_news_sm",
                "fr": "fr_core_news_sm", "ru": "ru_core_news_sm", "zh": "zh_core_web_sm"}

# Entity labels worth swapping. Dates, quantities and the like belong to `number`.
ENTITY_LABELS = {"PER", "PERSON", "LOC", "GPE", "ORG", "FAC", "NORP", "EVENT",
                 "PRODUCT", "WORK_OF_ART", "LANGUAGE", "MISC"}
# Entity types a document-internal swap may touch.
SWAP_LABELS = {"PER", "PERSON", "LOC", "GPE", "ORG"}
ELISION = ("'", "’")

# Negation particles the morphologiser may miss. CORE particles are negative on their own;
# PAIRED ones only alongside a CORE one (bare French "plus" means "more").
NEG_LEMMAS_CORE: dict[str, set[str]] = {
    "en": {"not", "n't", "never"},
    "de": {"nicht", "kein", "nie", "niemals"},
    "es": {"no", "nunca", "jamás"},
    "fr": {"ne"},
    "ru": {"не", "нет", "никогда"},
    "zh": {"不", "没", "没有", "未", "非"},
}
NEG_LEMMAS_PAIRED: dict[str, set[str]] = {
    "fr": {"pas", "plus", "jamais", "rien", "personne", "guère", "aucun", "nul"},
    "en": {"no", "any"},
}
# (particle before, particle after) the finite verb. German is V2: "wurde nicht veröffentlicht".
NEG_INSERT: dict[str, tuple[str, str]] = {
    "en": ("", "not"), "de": ("", "nicht"), "es": ("no", ""),
    "fr": ("ne", "pas"), "ru": ("не", ""), "zh": ("不", ""),
}
NEG_AUX_ONLY = {"en"}                            # no bare-verb negation in English
_VOWELS = "aeiouyáàâéèêíìîóòôúùûhAEIOUY"

# Dependency labels whose subtree can be dropped and leave a grammatical sentence:
# UD labels (en, es, fr, ru) and the TIGER labels of the German models.
OMIT_DEPS = {"prep", "obl", "advmod", "amod", "nmod", "appos", "npadvmod", "obl:tmod",
             "obl:npmod", "advmod:emph", "amod:att",
             "mo", "app", "mnr", "ag"}
OMIT_DEPS_ADJ_ONLY = {"nk"}                      # German: an adjective inside a noun phrase


def load_pipeline(lang: str):
    import spacy
    name = SPACY_MODELS[lang]
    try:
        return spacy.load(name)
    except OSError as exc:
        raise RuntimeError(f"spaCy model {name!r} is not installed: "
                           f"run `python -m spacy download {name}`") from exc


def _render(doc, replace: dict[int, str], drop: set[int],
            before: dict[int, str] | None = None, after: dict[int, str] | None = None) -> str:
    """Detokenise `doc` with edits applied; `whitespace_` keeps the spacing (none in zh).
    `before`/`after` are spliced verbatim, so the caller controls elision ("n'a")."""
    before, after = before or {}, after or {}
    parts: list[str] = []
    for t in doc:
        if t.i in before:
            parts.append(before[t.i])
        if t.i not in drop:
            parts.append(replace.get(t.i, t.text))
            if t.i in after:
                parts.append(after[t.i])
            parts.append(t.whitespace_)
    return clean_spaces("".join(parts))


@dataclass
class Perturber:
    """The edits of one target language, with entity and numeral banks harvested from its corpus."""

    lang: str
    seed: int = 42
    nlp: object = None
    docs: dict[str, object] = field(default_factory=dict)
    ent_bank: dict[str, list[str]] = field(default_factory=dict)
    card_bank: list[str] = field(default_factory=list)
    ord_bank: list[str] = field(default_factory=list)

    @classmethod
    def for_corpus(cls, sentences: Sequence[str], lang: str, seed: int = 42,
                   batch_size: int = 128) -> Perturber:
        nlp = load_pipeline(lang)
        uniq = list(dict.fromkeys(sentences))
        self = cls(lang=lang, seed=seed, nlp=nlp, docs=dict(zip(uniq, nlp.pipe(uniq, batch_size=batch_size))))
        self._harvest()
        return self

    def perturbations(self, sentence: str, categories: Sequence[str],
                      context: Sequence[str] = ()) -> dict[str, str]:
        """category -> the edit of `sentence`, for the categories it admits; two categories
        never keep the same edited sentence. `context` holds the document's sentences."""
        out: dict[str, str] = {}
        for c in categories:
            edited = self.edit(sentence, c, context)
            if edited and edited not in out.values():
                out[c] = edited
        return out

    def edit(self, sent: str, category: str, context: Sequence[str] = ()) -> str | None:
        for attempt in range(ATTEMPTS):
            rng = seeded_rng(self.seed, self.lang, category, sent, attempt)
            doc = self._doc(sent)
            match category:
                case "causality":
                    out = self._causality(doc, sent, rng)
                case "entity":
                    out = self._entity(doc, rng)
                case "number":
                    out = self._number(doc, rng)
                case "antonym":
                    out = replace_from_lexicon(sent, self.lang, ANTONYMS.get(self.lang, []), rng)
                case "number_near":
                    out = self._number_near(doc, rng)
                case "entity_doc":
                    out = self._entity_doc(doc, sent, context, rng)
                case "omission":
                    out = self._omission(doc, rng)
                case "modality":
                    out = replace_from_lexicon(sent, self.lang, MODAL_SOFT.get(self.lang, []), rng,
                                               one_way=True)
                case _:
                    raise ValueError(f"unknown category {category!r}")
            if out and out != sent:
                return out
        return None

    # ── banks ─────────────────────────────────────────────────────────────────
    def _harvest(self) -> None:
        ents: dict[str, set[str]] = {}
        cards: set[str] = set()
        ords_: set[str] = set()
        for doc in self.docs.values():
            for e in doc.ents:
                if e.label_ in ENTITY_LABELS:
                    ents.setdefault(e.label_, set()).add(e.text)
            for t in doc:
                if any(ch.isdigit() for ch in t.text):
                    continue
                if self._is_ordinal(t):
                    ords_.add(t.text)
                elif t.like_num and t.pos_ == "NUM":
                    cards.add(t.text)
        self.ent_bank = {k: sorted(v) for k, v in ents.items()}
        self.card_bank = sorted(cards)
        # not every model sets NumType=Ord (the German one does not): the curated list seeds the bank
        self.ord_bank = sorted(ords_ | set(ORDINALS.get(self.lang, [])))

    @property
    def bank(self) -> list[str]:
        return sorted({e for v in self.ent_bank.values() for e in v})

    def _is_ordinal(self, tok) -> bool:
        return ("Ord" in tok.morph.get("NumType")
                or tok.text.lower() in {w.lower() for w in ORDINALS.get(self.lang, [])})

    def _doc(self, sent: str):
        d = self.docs.get(sent)
        if d is None:
            d = self.docs[sent] = self.nlp(sent)
        return d

    # ── xSIM++ edits ──────────────────────────────────────────────────────────
    def _entity(self, doc, rng) -> str | None:
        # a span behind an elided article is left alone: "l'université" -> "l'Berlin"
        spans = [e for e in doc.ents if e.label_ in ENTITY_LABELS
                 and not (e.start and doc[e.start - 1].text.endswith(ELISION))]
        if not spans or not self.ent_bank:
            return None
        span = rng.choice(spans)
        pool = [e for e in self.ent_bank.get(span.label_, []) if e != span.text]
        if not pool:                                   # no same-type entity: any type
            pool = [e for e in self.bank if e != span.text]
        if not pool:
            return None
        return _render(doc, {span.start: rng.choice(pool)}, set(range(span.start + 1, span.end)))

    def _number(self, doc, rng) -> str | None:
        digits, words = [], []
        for t in doc:
            if any(ch.isdigit() for ch in t.text):
                digits.append(t)
            elif (t.pos_ == "NUM" and t.like_num) or self._is_ordinal(t):
                words.append(t)
        if not digits and not words:
            return None
        t = rng.choice(digits + words)
        if t in digits:
            new = _scramble_digits(t.text, rng)
        else:
            bank = (self.ord_bank if self._is_ordinal(t) else self.card_bank) or self.card_bank
            pool = [w for w in bank if w.lower() != t.text.lower()]
            new = match_case(rng.choice(pool), t.text) if pool else None
        return _render(doc, {t.i: new}, set()) if new else None

    def _causality(self, doc, sent: str, rng) -> str | None:
        ops = ["antonym", "modal", "negate"]
        rng.shuffle(ops)
        for op in ops:
            if op == "antonym":
                out = replace_from_lexicon(sent, self.lang, ANTONYMS.get(self.lang, []), rng)
            elif op == "modal":
                out = replace_from_lexicon(sent, self.lang, MODAL_BOOST.get(self.lang, []), rng,
                                           one_way=True)
            else:
                out = self._negate(doc, rng)
            if out and out != sent:
                return out
        return None

    def _negate(self, doc, rng) -> str | None:
        core = NEG_LEMMAS_CORE.get(self.lang, set())
        negs = [t for t in doc
                if "Neg" in t.morph.get("Polarity") or t.dep_ in ("neg", "ng")
                or t.lemma_.lower() in core or t.text.lower() in core]
        if negs:
            # every particle attached to one head goes, so "ne ... pas" is removed as a unit
            head = rng.choice(negs).head
            paired = NEG_LEMMAS_PAIRED.get(self.lang, set())
            drop = {t.i for t in negs if t.head == head} or {rng.choice(negs).i}
            drop |= {t.i for t in doc if t.head == head and t.lemma_.lower() in paired}
            return _render(doc, {}, drop)
        pre, post = NEG_INSERT.get(self.lang, ("", ""))
        if not (pre or post):
            return None
        verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")
                 and (not t.morph.get("VerbForm") or "Fin" in t.morph.get("VerbForm"))]
        verbs = verbs or [t for t in doc if t.pos_ in ("VERB", "AUX")]
        if self.lang in NEG_AUX_ONLY:
            verbs = [t for t in verbs if t.pos_ == "AUX"]
        if not verbs:
            return None
        v = rng.choice(verbs)
        # Romance clitics sit between the negation and the verb: "no se anunció"
        anchor = v.i
        while anchor > 0 and doc[anchor - 1].pos_ == "PRON" and doc[anchor - 1].head.i == v.i:
            anchor -= 1
        sep = joiner(self.lang)
        if self.lang == "fr" and pre == "ne" and doc[anchor].text[:1] in _VOWELS:
            pre, sep = "n'", ""
        return _render(doc, {}, set(),
                       {anchor: pre + sep} if pre else None,
                       {v.i: joiner(self.lang) + post} if post else None)

    # ── minor edits ───────────────────────────────────────────────────────────
    def _number_near(self, doc, rng) -> str | None:
        digits = [t for t in doc if any(ch.isdigit() for ch in t.text)]
        ords_ = [t for t in doc if not any(ch.isdigit() for ch in t.text) and self._is_ordinal(t)]
        if not digits and not ords_:
            return None
        t = rng.choice(digits + ords_)
        if t in digits:
            new = _nudge_digit(t.text, rng)
        else:
            ladder = [w.lower() for w in ORDINALS.get(self.lang, [])]
            if t.text.lower() not in ladder:
                return None
            i = ladder.index(t.text.lower())
            near = [j for j in (i - 1, i + 1) if 0 <= j < len(ladder)]
            new = match_case(ladder[rng.choice(near)], t.text) if near else None
        return _render(doc, {t.i: new}, set()) if new else None

    def _entity_doc(self, doc, sent: str, context: Sequence[str], rng) -> str | None:
        spans = [e for e in doc.ents if e.label_ in SWAP_LABELS
                 and not (e.start and doc[e.start - 1].text.endswith(ELISION))]
        if not spans:
            return None

        def proper(name: str) -> bool:
            return (name[:1].isupper() and len(name) >= 3 and not any(c.isdigit() for c in name)
                    and name not in sent)

        pool: dict[str, set[str]] = {}
        for other in context:
            if other == sent:
                continue
            for e in self._doc(other).ents:
                if e.label_ in SWAP_LABELS and proper(e.text):
                    pool.setdefault(e.label_, set()).add(e.text)
        cands = [(sp, sorted(pool.get(sp.label_, set()) - {sp.text})) for sp in spans]
        cands = [(sp, alts) for sp, alts in cands if alts]
        if not cands:
            return None
        span, alts = rng.choice(cands)
        return _render(doc, {span.start: rng.choice(alts)}, set(range(span.start + 1, span.end)))

    def _omission(self, doc, rng) -> str | None:
        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        cands = []
        for t in doc:
            if t.is_punct or not (t.dep_ in OMIT_DEPS
                                  or (t.dep_ in OMIT_DEPS_ADJ_ONLY and t.pos_ == "ADJ")):
                continue
            sub = list(t.subtree)
            # a short, contiguous, verb-free modifier with no punctuation inside it
            if not 1 <= len(sub) <= 5 or (root is not None and root in sub):
                continue
            if sub[-1].i - sub[0].i + 1 != len(sub):
                continue
            if any(x.is_punct or x.pos_ in ("VERB", "AUX") or x.is_quote for x in sub):
                continue
            if len(doc) - len(sub) < 4:
                continue
            cands.append(sub)
        if not cands:
            return None
        sub = rng.choice(cands)
        drop = {x.i for x in sub}
        lo, hi = min(drop), max(drop)
        # a comma that framed the dropped phrase goes with it
        if hi + 1 < len(doc) and doc[hi + 1].text == "," and lo > 0 and doc[lo - 1].text == ",":
            drop.add(hi + 1)
        elif lo > 0 and doc[lo - 1].text == "," and (hi + 1 >= len(doc) or doc[hi + 1].is_punct):
            drop.add(lo - 1)
        return _render(doc, {}, drop)


def _nudge_digit(surface: str, rng) -> str | None:
    """One digit of the first digit run moved by +-1; a leading digit never becomes 0."""
    m = re.search(r"\d+", surface)
    if not m:
        return None
    old = m.group()
    for _ in range(10):
        i = rng.randrange(len(old))
        d = int(old[i]) + rng.choice((-1, 1))
        if not 0 <= d <= 9 or (i == 0 and len(old) > 1 and d == 0):
            continue
        return surface[:m.start()] + old[:i] + str(d) + old[i + 1:] + surface[m.end():]
    return None


def _scramble_digits(surface: str, rng) -> str | None:
    """The first digit run replaced by another one of the same width."""
    m = re.search(r"\d+", surface)
    if not m:
        return None
    old = m.group()
    n = len(old)
    lo = 10 ** (n - 1) if n > 1 and old[0] != "0" else 0
    for _ in range(20):
        cand = str(rng.randint(lo, 10 ** n - 1)).zfill(n)
        if cand != old:
            return surface[:m.start()] + cand + surface[m.end():]
    return None
