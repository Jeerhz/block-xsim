"""Word lists and string helpers shared by the edits (perturber.py).

Every edit is deterministic: its RNG is seeded from (seed, lang, category, sentence, attempt).
"""
from __future__ import annotations

import random
import re
from collections.abc import Sequence

from block_xsim.corpora import NO_SPACE_LANGS

ANTONYMS: dict[str, list[tuple[str, str]]] = {
    "en": [("good", "bad"), ("high", "low"), ("large", "small"), ("big", "small"),
           ("more", "less"), ("most", "fewest"), ("increase", "decrease"),
           ("increased", "decreased"), ("rise", "fall"), ("rising", "falling"),
           ("positive", "negative"), ("possible", "impossible"), ("open", "closed"),
           ("early", "late"), ("first", "last"), ("strong", "weak"), ("new", "old"),
           ("young", "old"), ("long", "short"), ("fast", "slow"), ("hot", "cold"),
           ("safe", "dangerous"), ("healthy", "sick"), ("above", "below"),
           ("before", "after"), ("success", "failure"), ("win", "lose"),
           ("accept", "reject"), ("begin", "end"), ("best", "worst")],
    "de": [("gut", "schlecht"), ("hoch", "niedrig"), ("groß", "klein"),
           ("mehr", "weniger"), ("steigen", "fallen"), ("positiv", "negativ"),
           ("möglich", "unmöglich"), ("offen", "geschlossen"), ("früh", "spät"),
           ("erste", "letzte"), ("stark", "schwach"), ("neu", "alt"),
           ("jung", "alt"), ("lang", "kurz"), ("schnell", "langsam"),
           ("heiß", "kalt"), ("sicher", "gefährlich"), ("gesund", "krank"),
           ("über", "unter"), ("vor", "nach"), ("Erfolg", "Misserfolg")],
    "es": [("bueno", "malo"), ("buena", "mala"), ("alto", "bajo"), ("alta", "baja"),
           ("grande", "pequeño"), ("más", "menos"), ("aumentar", "disminuir"),
           ("positivo", "negativo"), ("posible", "imposible"),
           ("abierto", "cerrado"), ("temprano", "tarde"), ("primero", "último"),
           ("fuerte", "débil"), ("nuevo", "viejo"), ("joven", "viejo"),
           ("largo", "corto"), ("rápido", "lento"), ("caliente", "frío"),
           ("seguro", "peligroso"), ("sano", "enfermo"), ("antes", "después"),
           ("éxito", "fracaso")],
    "fr": [("bon", "mauvais"), ("bonne", "mauvaise"), ("haut", "bas"),
           ("grand", "petit"), ("plus", "moins"), ("augmenter", "diminuer"),
           ("positif", "négatif"), ("possible", "impossible"),
           ("ouvert", "fermé"), ("tôt", "tard"), ("premier", "dernier"),
           ("fort", "faible"), ("nouveau", "ancien"), ("jeune", "vieux"),
           ("long", "court"), ("rapide", "lent"), ("chaud", "froid"),
           ("sûr", "dangereux"), ("sain", "malade"), ("avant", "après"),
           ("succès", "échec")],
    "ru": [("хороший", "плохой"), ("высокий", "низкий"), ("большой", "маленький"),
           ("больше", "меньше"), ("увеличить", "уменьшить"),
           ("положительный", "отрицательный"), ("возможно", "невозможно"),
           ("открытый", "закрытый"), ("рано", "поздно"), ("первый", "последний"),
           ("сильный", "слабый"), ("новый", "старый"), ("молодой", "старый"),
           ("длинный", "короткий"), ("быстрый", "медленный"),
           ("горячий", "холодный"), ("безопасный", "опасный"),
           ("здоровый", "больной"), ("до", "после"), ("успех", "провал")],
    # Chinese pairs are all ≥2 characters on purpose: a single-character antonym
    # (大/小) matches *inside* compounds — 大学 "university" → 小学 "elementary
    # school" — which is a lexical swap, not a causality alternation.
    "zh": [("增加", "减少"), ("上升", "下降"), ("积极", "消极"),
           ("第一", "最后"), ("年轻", "年老"), ("安全", "危险"),
           ("健康", "生病"), ("成功", "失败"), ("支持", "反对"),
           ("同意", "反对"), ("提高", "降低"), ("接受", "拒绝")],
}

# Negation strengthening (Tan et al. 2021): hedged modal → assertive modal.
MODAL_BOOST: dict[str, list[tuple[str, str]]] = {
    "en": [("may", "will"), ("might", "will"), ("could", "will"),
           ("possibly", "certainly"), ("perhaps", "certainly"),
           ("probably", "certainly"), ("should", "must"), ("can", "must")],
    "de": [("könnte", "wird"), ("kann", "muss"), ("vielleicht", "sicherlich"),
           ("möglicherweise", "sicherlich"), ("sollte", "muss")],
    "es": [("podría", "va a"), ("puede", "debe"), ("quizás", "ciertamente"),
           ("tal vez", "ciertamente"), ("probablemente", "ciertamente")],
    "fr": [("pourrait", "va"), ("peut", "doit"), ("peut-être", "certainement"),
           ("probablement", "certainement"), ("devrait", "doit")],
    "ru": [("может", "будет"), ("возможно", "непременно"),
           ("вероятно", "непременно"), ("следует", "должен")],
    "zh": [("可能", "一定"), ("也许", "必然"), ("大概", "必然"), ("应该", "必须")],
}


ORDINALS: dict[str, list[str]] = {
    "en": ["first", "second", "third", "fourth", "fifth", "sixth", "seventh",
           "eighth", "ninth", "tenth"],
    "de": ["erste", "zweite", "dritte", "vierte", "fünfte", "sechste", "siebte",
           "achte", "neunte", "zehnte"],
    "es": ["primero", "segundo", "tercero", "cuarto", "quinto", "sexto",
           "séptimo", "octavo", "noveno", "décimo"],
    "fr": ["premier", "deuxième", "troisième", "quatrième", "cinquième",
           "sixième", "septième", "huitième", "neuvième", "dixième"],
    "ru": ["первый", "второй", "третий", "четвёртый", "пятый", "шестой",
           "седьмой", "восьмой", "девятый", "десятый"],
    "zh": ["第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八",
           "第九", "第十"],
}


# The minor `modality` edit softens a quantifier or a modal, one way only.
MODAL_SOFT: dict[str, list[tuple[str, str]]] = {
    "en": [("always", "often"), ("never", "rarely"), ("all", "most"), ("every", "most"),
           ("must", "may"), ("will", "might"), ("everyone", "many"), ("nobody", "few"),
           ("only", "mainly"), ("entirely", "largely"), ("certainly", "probably")],
    "de": [("immer", "oft"), ("nie", "selten"), ("alle", "die meisten"), ("jeder", "mancher"),
           ("muss", "kann"), ("müssen", "können"), ("wird", "könnte"), ("niemand", "kaum jemand"),
           ("nur", "vor allem"), ("völlig", "weitgehend"), ("sicherlich", "wahrscheinlich")],
    "es": [("siempre", "a menudo"), ("nunca", "rara vez"), ("todos", "la mayoría de"),
           ("todas", "la mayoría de"), ("debe", "puede"), ("deben", "pueden"), ("nadie", "casi nadie"),
           ("solo", "sobre todo"), ("totalmente", "en gran parte"), ("ciertamente", "probablemente")],
    "fr": [("toujours", "souvent"), ("jamais", "rarement"), ("tous", "la plupart"),
           ("toutes", "la plupart"), ("doit", "peut"), ("doivent", "peuvent"), ("personne", "presque personne"),
           ("seulement", "surtout"), ("entièrement", "en grande partie"), ("certainement", "probablement")],
    "ru": [("всегда", "часто"), ("никогда", "редко"), ("все", "большинство"), ("каждый", "многие"),
           ("должен", "может"), ("должны", "могут"), ("никто", "почти никто"),
           ("только", "в основном"), ("полностью", "в значительной степени"), ("несомненно", "вероятно")],
}


def match_case(new: str, old: str) -> str:
    if old[:1].isupper() and not old.isupper():
        return new[:1].upper() + new[1:]
    if old.isupper() and len(old) > 1:
        return new.upper()
    return new


def seeded_rng(seed: int, *parts) -> random.Random:
    return random.Random(f"{seed}|" + "|".join(str(p) for p in parts))


def replace_from_lexicon(sent: str, lang: str, pairs: Sequence[tuple[str, str]],
                         rng: random.Random, one_way: bool = False) -> str | None:
    """Replace one occurrence of a lexicon word by its counterpart."""
    hits: list[tuple[int, int, str, str]] = []
    for a, b in pairs:
        directions = [(a, b)] if one_way else [(a, b), (b, a)]
        for src, dst in directions:
            pattern = re.escape(src) if lang in NO_SPACE_LANGS else rf"\b{re.escape(src)}\b"
            for m in re.finditer(pattern, sent, flags=re.IGNORECASE):
                hits.append((m.start(), m.end(), m.group(), dst))
    if not hits:
        return None
    s, e, surface, dst = rng.choice(hits)
    return sent[:s] + match_case(dst, surface) + sent[e:]


def clean_spaces(text: str) -> str:
    text = re.sub(r"\s{2,}", " ", text).strip()
    return re.sub(r"\s+([.,;:!?»])", r"\1", text)


