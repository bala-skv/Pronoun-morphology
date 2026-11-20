"""Prompt construction and target-token resolution.

Key rule: a target word is tokenized *in context*
(tok(prompt + " " + word) minus tok(prompt)), never on its own. If the candidate forms share
leading tokens (e.g. Hindi " मे"+"री" vs " मे"+"रा"), the shared prefix is appended to the
prompt and the metric is scored at the first token where the forms differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

# Contrast 1 frames: "{N}" is the possessed noun; prompt ends right before the possessive.
#
# With the bare frame ("यह किताब असल में") Gemma almost never expects a
# possessive next (P(' मे') ≈ 0.005) and agreement accuracy is ~0.56. A two-sentence context that
# shows one feminine and one masculine possessive ("यह कलम मेरी है। यह घर मेरा है।") makes the
# possessive likely (P ≈ 0.55) and accuracy ≈ 0.9 for Hindi. Both orders (f-m, m-f) and two
# context-noun sets are used so that neither form is favoured by recency. Context nouns are excluded
# from the test nouns (C1_CTX_NOUN_IDS).
C1_CTX_NOUN_IDS = {2, 40, 4, 41}  # pen, house, chair, room
C1_FRAMES = {
    "hi": ["यह कलम मेरी है। यह घर मेरा है। यह {N} असल में",
           "यह घर मेरा है। यह कलम मेरी है। यह {N} असल में",
           "यह कुर्सी मेरी है। यह कमरा मेरा है। यह {N} असल में",
           "यह कमरा मेरा है। यह कुर्सी मेरी है। यह {N} असल में"],
    "pa": ["ਇਹ ਕਲਮ ਮੇਰੀ ਹੈ। ਇਹ ਘਰ ਮੇਰਾ ਹੈ। ਇਹ {N} ਅਸਲ ਵਿੱਚ",
           "ਇਹ ਘਰ ਮੇਰਾ ਹੈ। ਇਹ ਕਲਮ ਮੇਰੀ ਹੈ। ਇਹ {N} ਅਸਲ ਵਿੱਚ",
           "ਇਹ ਕੁਰਸੀ ਮੇਰੀ ਹੈ। ਇਹ ਕਮਰਾ ਮੇਰਾ ਹੈ। ਇਹ {N} ਅਸਲ ਵਿੱਚ",
           "ਇਹ ਕਮਰਾ ਮੇਰਾ ਹੈ। ਇਹ ਕੁਰਸੀ ਮੇਰੀ ਹੈ। ਇਹ {N} ਅਸਲ ਵਿੱਚ"],
    "en": ["This pen is mine. This house is mine. This {N} is actually",
           "This house is mine. This pen is mine. This {N} is actually",
           "This chair is mine. This room is mine. This {N} is actually",
           "This room is mine. This chair is mine. This {N} is actually"],
    "ta": ["இந்தப் பேனா என்னுடையது. இந்த வீடு என்னுடையது. இந்த {N} உண்மையில்",
           "இந்த வீடு என்னுடையது. இந்தப் பேனா என்னுடையது. இந்த {N} உண்மையில்",
           "இந்த நாற்காலி என்னுடையது. இந்த அறை என்னுடையது. இந்த {N} உண்மையில்",
           "இந்த அறை என்னுடையது. இந்த நாற்காலி என்னுடையது. இந்த {N} உண்மையில்"],
}
# Bare frames (no context), kept for the behavioural comparison.
C1_FRAMES_BARE = {
    "hi": ["यह {N} असल में", "वह {N} असल में", "यह {N} तो", "यह {N} सचमुच"],
    "pa": ["ਇਹ {N} ਅਸਲ ਵਿੱਚ", "ਉਹ {N} ਅਸਲ ਵਿੱਚ", "ਇਹ {N} ਤਾਂ", "ਇਹ {N} ਸੱਚਮੁੱਚ"],
    "en": ["This {N} is actually", "That {N} is actually", "This {N} is really", "This {N} is definitely"],
    "ta": ["இந்த {N} உண்மையில்", "அந்த {N} உண்மையில்", "இந்த {N} நிச்சயமாக", "இந்த {N} உண்மையாகவே"],
}
C1_FORMS = {
    "hi": {"f": "मेरी", "m": "मेरा"},
    "pa": {"f": "ਮੇਰੀ", "m": "ਮੇਰਾ"},
    "en": {"f": "mine", "m": "mine"},
    "ta": {"f": "என்னுடையது", "m": "என்னுடையது"},
}
# Contrast 2 frames: pronoun form is the cue, verb is the target.
# A neutral "हाँ, " ("Yes, ") prefix is needed: sentence-initially (no leading space) मैंने tokenizes as
# म+ैं+ने vs मैं as म+ैं, so the pair lengths differ; after a space both are single tokens ▁मैंने / ▁मैं.
C2_FRAMES = {
    "hi": {"clean": "हाँ, मैंने {O}", "corrupt": "हाँ, मैं {O}"},
    "pa": {"clean": "ਹਾਂ, ਮੈਂ {O}"},  # Punjabi 1sg has no ergative form -> behavioural control only
}


@dataclass
class Item:
    lang: str
    contrast: str
    prompt: str
    pos_forms: list[str]
    neg_forms: list[str]
    meta: dict = field(default_factory=dict)
    # filled by resolve():
    ids: list[int] | None = None
    base_len: int | None = None  # number of tokens of the prompt itself (before any shared target prefix)
    pos_ids: list[int] | None = None
    neg_ids: list[int] | None = None
    prefix_str: str = ""
    form_ntoks: dict | None = None
    ok: bool = False
    why: str = ""


def tok_ids(model, text: str) -> list[int]:
    return model.to_tokens(text, prepend_bos=True)[0].tolist()


def continuation_ids(model, prompt: str, word: str) -> list[int] | None:
    base = tok_ids(model, prompt)
    full = tok_ids(model, prompt + " " + word)
    if full[: len(base)] != base:
        return None  # tokenization merges across the boundary
    return full[len(base):]


def resolve(model, item: Item) -> Item:
    """Fill ids / pos_ids / neg_ids for an item (see module docstring)."""
    forms = list(dict.fromkeys(item.pos_forms + item.neg_forms))
    conts = {f: continuation_ids(model, item.prompt, f) for f in forms}
    if any(c is None for c in conts.values()):
        item.why = "boundary-merge"
        return item
    item.form_ntoks = {f: len(c) for f, c in conts.items()}
    base = tok_ids(model, item.prompt)
    item.base_len = len(base)
    if set(item.pos_forms) == set(item.neg_forms):  # control languages: single form, no contrast
        item.ids, item.ok = base, True
        item.pos_ids = conts[item.pos_forms[0]]
        item.neg_ids = []
        return item
    # longest common prefix over all candidate continuations
    lcp = []
    for toks in zip(*conts.values()):
        if len(set(toks)) == 1:
            lcp.append(toks[0])
        else:
            break
    if any(len(c) <= len(lcp) for c in conts.values()):
        item.why = "form-is-prefix-of-another"
        return item
    k = len(lcp)
    pos = sorted({conts[f][k] for f in item.pos_forms})
    neg = sorted({conts[f][k] for f in item.neg_forms})
    if set(pos) & set(neg):
        item.why = "pos/neg share decision token"
        return item
    item.ids = base + lcp
    item.pos_ids, item.neg_ids = pos, neg
    item.prefix_str = model.tokenizer.decode(lcp) if lcp else ""
    item.ok = True
    return item


# ----------------------------------------------------------------------------- data loading

def load_nouns() -> pd.DataFrame:
    return pd.read_csv(DATA / "c1_nouns.csv")


def load_ergative() -> pd.DataFrame:
    return pd.read_csv(DATA / "c2_ergative.csv")


def load_lexicon() -> pd.DataFrame:
    return pd.read_csv(DATA / "lexicon.csv")


def c1_items(model, lang: str, bare: bool = False) -> list[Item]:
    """One item per (frame, noun). Δ = logit(fem form) - logit(masc form)."""
    nouns = load_nouns()
    if not bare:
        nouns = nouns[~nouns.id.isin(C1_CTX_NOUN_IDS)]
    gcol = {"hi": "hi_gender", "pa": "pa_gender"}.get(lang, "hi_gender")  # en/ta: Hindi gender of translation
    forms = C1_FORMS[lang]
    items = []
    for fi, frame in enumerate((C1_FRAMES_BARE if bare else C1_FRAMES)[lang]):
        for _, r in nouns.iterrows():
            noun = r[lang]
            g = r[gcol]
            if lang in ("hi", "pa") and g not in ("f", "m"):
                continue  # e.g. "fm": both genders acceptable (native-speaker note) -> no single correct form
            if lang in ("hi", "pa"):
                pos, neg = [forms["f"]], [forms["m"]]
            else:
                pos = neg = [forms["f"]]
            it = Item(lang, "c1", frame.format(N=noun), pos, neg,
                      meta=dict(frame=fi, noun_id=int(r["id"]), noun=noun, gender=g, en=r["en"]))
            items.append(resolve(model, it))
    return items


def c2_items(model, lang: str) -> list[Item]:
    """Ergative contrast. Δ = logit(perfective agreeing w/ object) - logsumexp(habitual m, f)."""
    df = load_ergative()
    df = df[df.lang == lang]
    items = []
    for _, r in df.iterrows():
        for cond, frame in C2_FRAMES[lang].items():
            it = Item(lang, "c2", frame.format(O=r["object"]), [r["perf"]], [r["hab_m"], r["hab_f"]],
                      meta=dict(cond=cond, row_id=int(r["id"]), object=r["object"], obj_gender=r["obj_gender"],
                                gloss=r["gloss"]))
            items.append(resolve(model, it))
    return items


def c1_pairs(items: list[Item], n_max: int | None = None, seed: int = 0, both_directions: bool = True):
    """Clean/corrupt pairs for patching: same frame, fem noun vs masc noun, equal token length.

    Clean = fem noun (target मेरी), corrupt = masc noun (target मेरा) and, if both_directions,
    the reverse. Returned as (clean, corrupt, sign) where sign=+1 means Δ_clean should be > 0.
    """
    import random

    rng = random.Random(seed)
    ok = [i for i in items if i.ok]
    by_key: dict = {}
    for it in ok:
        by_key.setdefault((it.meta["frame"], len(it.ids), it.meta["gender"]), []).append(it)
    pairs = []
    for (frame, L, g), fem in by_key.items():
        if g != "f":
            continue
        masc = list(by_key.get((frame, L, "m"), []))
        fem = list(fem)
        rng.shuffle(fem)
        rng.shuffle(masc)
        for a, b in zip(fem, masc):  # 1:1 matching so nouns are not reused within a frame
            pairs.append((a, b, +1))
            if both_directions:
                pairs.append((b, a, -1))
    rng.shuffle(pairs)
    return pairs[:n_max] if n_max else pairs


def c2_pairs(items: list[Item]):
    """Hindi: clean = मैंने O, corrupt = मैं O (same object). Token lengths may differ; see patching."""
    by = {}
    for it in items:
        if it.ok:
            by.setdefault(it.meta["row_id"], {})[it.meta["cond"]] = it
    return [(d["clean"], d["corrupt"], +1) for d in by.values() if "clean" in d and "corrupt" in d]
