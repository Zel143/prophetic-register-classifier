"""
Order-sensitive ("structural") features -- the half of the register
hypothesis the original feature set never tested.

The null control (docs/null-control.md) showed that word-shuffled verses
get the same predictions as the real ones. That follows from how the
existing features are built: function-word rates, POS proportions and
ttr are all bag-of-words statistics, so word order cannot affect them.
The README defines prophetic register partly by *structure* (address
form, formula placement, clause shape), and nothing tested that. These
features can only be computed from word order. Shuffling a verse should
change them:

  struct_bg_<A>_<B>          share of adjacent UPOS pairs that are A->B,
                             for the TOP_K most frequent pairs in the seed
                             set (picked once, from the seed set only, and
                             saved to results/struct_bigrams.json so every
                             corpus uses the same columns)
  struct_clause_verb_initial share of clauses whose first word (ignoring a
                             leading conjunction) is the clause's verb or
                             its auxiliary -- "Hear, O heavens", "Go and
                             tell", "Woe unto"-type openings
  struct_clause_cc_initial   share of clauses opening with a coordinating
                             conjunction -- KJV narrative's "And he said"
  struct_clause_subj_initial share of clauses whose first word (after any
                             conjunction) belongs to the subject
  struct_vs_order            among clauses with an explicit subject, share
                             where the verb comes first -- inversion such
                             as "saith the LORD", "said he"
  struct_first_verb_relpos   position of the first verb/aux, as a fraction
                             of the chunk (1.0 when there is no verb)
  struct_mean_dep_dist       mean linear distance between each word and
                             its syntactic head (shuffling inflates it)
  struct_clauses_per_10w     clause density

Punctuation is ignored in every feature. The null control's shuffled
condition has no punctuation, so including it would let a feature detect
"shuffled" from missing commas rather than from word order.

Before parsing, archaic verb forms are modernized *for these features
only* (see modernize()). spaCy tags "saith" as a preposition, so without
this the parser misses the verb in "thus saith the LORD", the project's
central prophetic formula. Rules: a fixed map for irregulars (saith ->
says, hath -> has, spake -> spoke, ...) and -eth -> -s/-es/-ies, where the
right stem is checked against the KJV's own vocabulary (cometh -> comes
because "come" is a KJV word; teeth stays as it is because neither "te"
nor "tee" is). Capitalized words (Seth, Japheth, Nazareth) are never
touched. The same function is applied to every corpus and null condition,
so it can't favor one class. Second-person -est forms (goest, sayest) are
left alone.

Clauses come from spaCy's dependency parse: a clause head is a VERB/AUX
token attached as ROOT/conj/ccomp/advcl/relcl/xcomp/parataxis/csubj, and
its clause is its subtree minus any nested clause's subtree. en_core_web_sm
was trained on modern English and misparses some archaic KJV syntax, so
read these features as approximate (the same caveat extract_features.py
gives for the POS columns).

Writes sidecar CSVs (same rows, same order, plus a text column to check
alignment) next to each existing feature file, without touching those
files:

  results/seed_set_struct_features.csv
  results/transfer_struct_features.csv
  results/transfer_pericope_chunks_struct_features.csv

Usage:
    python src/structural_features.py
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SEED_SET_PATH = os.path.join(REPO_ROOT, "data", "seed_set.csv")
BIGRAMS_PATH = os.path.join(RESULTS_DIR, "struct_bigrams.json")

# (source CSV holding a `text` column, sidecar to write)
TARGETS = [
    (SEED_SET_PATH, os.path.join(RESULTS_DIR, "seed_set_struct_features.csv")),
    (os.path.join(RESULTS_DIR, "transfer_features.csv"),
     os.path.join(RESULTS_DIR, "transfer_struct_features.csv")),
    (os.path.join(RESULTS_DIR, "transfer_pericope_chunks_features.csv"),
     os.path.join(RESULTS_DIR, "transfer_pericope_chunks_struct_features.csv")),
]

KJV_DIR = os.path.join(REPO_ROOT, "data", "Bible-kjv")

TOP_K = 20
CLAUSE_DEPS = {"ROOT", "conj", "ccomp", "advcl", "relcl", "xcomp", "parataxis", "csubj"}
VERBISH = {"VERB", "AUX"}

_nlp = None
_kjv_vocab = None

ARCHAIC = {
    "saith": "says", "hath": "has", "doth": "does", "hast": "have", "dost": "do",
    "shalt": "shall", "wilt": "will", "spake": "spoke", "sware": "swore",
    "brake": "broke", "bare": "bore", "wast": "were", "wert": "were",
    "canst": "can", "couldest": "could", "shouldest": "should", "wouldest": "would",
    "didst": "did", "hadst": "had", "mayest": "may", "art": "are",
}
ETH_RE = re.compile(r"\b([a-z]+)eth\b")
WORD_TOKEN_RE = re.compile(r"\b[A-Za-z]+\b")


def kjv_vocab():
    global _kjv_vocab
    if _kjv_vocab is None:
        vocab = set()
        for p in glob.glob(os.path.join(KJV_DIR, "*.json")):
            with open(p, encoding="utf-8-sig") as f:
                book = json.load(f)
            for ch in book.get("chapters", []) if isinstance(book, dict) else []:
                for v in ch["verses"]:
                    vocab.update(re.findall(r"[a-z]+", v["text"].lower()))
        if not vocab:
            print(f"warning: no KJV text in {KJV_DIR}; -eth verbs won't be modernized "
                  "(run src/setup_data.py --kjv-only)")
        _kjv_vocab = vocab
    return _kjv_vocab


def _modern_eth(stem, vocab):
    if stem + "e" in vocab:                       # cometh -> comes, seeth -> sees
        return stem + "es" if stem.endswith(("s", "z")) else stem + "e" + "s"
    if stem.endswith(("s", "sh", "ch", "x", "z", "o")) and stem in vocab:
        return stem + "es"                        # passeth -> passes, goeth -> goes
    if stem.endswith("i") and len(stem) > 2:
        return stem[:-1] + "ies"                  # crieth -> cries
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[:-1] in vocab and stem not in vocab:
        return stem[:-1] + "s"                    # sitteth -> sits
    if stem in vocab:
        return stem + "s"                         # knoweth -> knows, dwelleth -> dwells
    return None


def modernize(text):
    """Archaic -> modern verb forms, for parsing only (see module docstring)."""
    vocab = kjv_vocab()

    def fix(m):
        w = m.group(0)
        if w[0].isupper() and w.lower() not in ARCHAIC:
            return w
        low = w.lower()
        if low in ARCHAIC:
            out = ARCHAIC[low]
        else:
            em = ETH_RE.fullmatch(low)
            out = _modern_eth(em.group(1), vocab) if em and vocab else None
            if out is None:
                return w
        return out.capitalize() if w[0].isupper() else out

    return WORD_TOKEN_RE.sub(fix, text)


def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _content_tokens(doc):
    return [t for t in doc if not t.is_punct and not t.is_space]


def pos_sequence(doc):
    return [t.pos_ for t in _content_tokens(doc)]


def select_bigrams(docs, k=TOP_K):
    c = Counter()
    for d in docs:
        tags = pos_sequence(d)
        c.update(zip(tags, tags[1:]))
    return [f"{a}_{b}" for (a, b), _ in c.most_common(k)]


def _clauses(doc):
    heads = [t for t in doc if t.dep_ in CLAUSE_DEPS and t.pos_ in VERBISH]
    head_set = set(heads)
    clauses = []
    for h in heads:
        nested = set()
        for t in h.subtree:
            if t is not h and t in head_set:
                nested.update(x.i for x in t.subtree)
        toks = [t for t in h.subtree
                if t.i not in nested and not t.is_punct and not t.is_space]
        if toks:
            clauses.append((h, sorted(toks, key=lambda t: t.i)))
    return clauses


def structural_features(doc, bigrams):
    content = _content_tokens(doc)
    n = len(content) or 1
    feat = {}

    tags = [t.pos_ for t in content]
    pairs = Counter(f"{a}_{b}" for a, b in zip(tags, tags[1:]))
    n_pairs = max(len(tags) - 1, 1)
    for bg in bigrams:
        feat[f"struct_bg_{bg}"] = pairs.get(bg, 0) / n_pairs

    clauses = _clauses(doc)
    verb_init = cc_init = subj_init = vs = with_subj = 0
    for head, toks in clauses:
        if toks[0].dep_ == "cc":
            cc_init += 1
        rest = [t for t in toks if t.dep_ not in ("cc", "mark")] or toks
        first = rest[0]
        if first is head or (first.head is head and first.dep_ in ("aux", "auxpass", "neg")):
            verb_init += 1
        subjects = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass", "expl")]
        if subjects:
            with_subj += 1
            subj_tokens = {x.i for x in subjects[0].subtree}
            if first.i in subj_tokens:
                subj_init += 1
            if head.i < subjects[0].i:
                vs += 1
    nc = len(clauses)
    feat["struct_clause_verb_initial"] = verb_init / nc if nc else 0.0
    feat["struct_clause_cc_initial"] = cc_init / nc if nc else 0.0
    feat["struct_clause_subj_initial"] = subj_init / nc if nc else 0.0
    feat["struct_vs_order"] = vs / with_subj if with_subj else 0.0

    pos_idx = {t.i: k for k, t in enumerate(content)}
    first_verb = next((pos_idx[t.i] for t in content if t.pos_ in VERBISH), None)
    feat["struct_first_verb_relpos"] = first_verb / n if first_verb is not None else 1.0

    dists = [abs(pos_idx[t.i] - pos_idx[t.head.i]) for t in content
             if t.head is not t and t.head.i in pos_idx]
    feat["struct_mean_dep_dist"] = float(np.mean(dists)) if dists else 0.0
    feat["struct_clauses_per_10w"] = nc / n * 10
    return feat


def load_bigrams():
    if not os.path.exists(BIGRAMS_PATH):
        raise FileNotFoundError(f"{BIGRAMS_PATH} missing -- run src/structural_features.py first")
    with open(BIGRAMS_PATH, encoding="utf-8") as f:
        return json.load(f)


def featurize_texts(texts, bigrams=None):
    """Structural features for a list of texts, as a DataFrame."""
    bigrams = bigrams or load_bigrams()
    docs = get_nlp().pipe([modernize(t) for t in texts])
    return pd.DataFrame([structural_features(d, bigrams) for d in docs])


def with_struct(df, sidecar_path):
    """Append sidecar struct_* columns to an existing feature frame, checking
    the rows line up."""
    side = pd.read_csv(sidecar_path)
    if len(side) != len(df) or not (side["text"].astype(str).values == df["text"].astype(str).values).all():
        raise ValueError(f"{sidecar_path} is out of sync with its source -- rerun src/structural_features.py")
    struct_cols = [c for c in side.columns if c.startswith("struct_")]
    return pd.concat([df.reset_index(drop=True), side[struct_cols]], axis=1)


def build():
    nlp = get_nlp()
    seed = pd.read_csv(SEED_SET_PATH)
    seed_docs = list(nlp.pipe([modernize(t) for t in seed["text"]]))
    bigrams = select_bigrams(seed_docs)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(BIGRAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(bigrams, f, indent=1)
    print(f"selected {len(bigrams)} POS bigrams from the seed set -> {BIGRAMS_PATH}")

    for src, out in TARGETS:
        if not os.path.exists(src):
            print(f"skip: {src} not found")
            continue
        df = pd.read_csv(src)
        feats = featurize_texts(df["text"].astype(str), bigrams)
        side = pd.concat([df[["text"]].reset_index(drop=True), feats], axis=1)
        side.to_csv(out, index=False)
        print(f"wrote {len(side)} rows x {feats.shape[1]} struct features to {out}")


if __name__ == "__main__":
    build()
