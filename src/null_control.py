"""
Null-control baseline: how often does the trained classifier call text
"prophetic" when that text has no register at all?

Every transfer number in this project ("Sibylline Oracles 37% prophetic",
"Bahman Yasht 7-22%") has so far been read against an implicit baseline
of 0%. That's the wrong floor. A three-class classifier fed pure noise
will still put *some* share of it in each class, and if that share is
already ~30% prophetic, a transfer corpus scoring 35% is not evidence of
anything. This script measures the real floor.

Inspiration: TempleOS (Terry A. Davis, 2005-2018) shipped an "oracle"
that drew random words -- and random Bible passages -- and presented the
result as divine speech. Readers routinely found meaning in that output,
which is the textbook case of apophenia (perceiving pattern in noise).
The question here is the machine version of the same failure mode: does
a stylometric classifier also "hear prophecy" in random text? Like the
rest of this project, this measures textual style only. It says nothing
about any text's truth, inspiration or scriptural status.

Four conditions, all drawn from the KJV with a fixed seed, with chunk
lengths sampled from the seed set's own verse-length distribution so
n_words and length-sensitive features stay in the training range:

  word_salad_uniform  random KJV word *types*, uniform -- closest to the
                      TempleOS random-word oracle; pure noise
  word_salad_freq     random KJV *tokens*, frequency-weighted -- noise
                      that still has the KJV's function-word profile
  verse_shuffle       real KJV verses (not in the seed set) with word
                      order shuffled -- right lexicon, no syntax
  random_verse        real KJV verses (not in the seed set), untouched --
                      the TempleOS random-passage oracle; this one is a
                      base rate for the whole KJV, not noise
  verse_flat          the *same* verses as verse_shuffle, lowercased and
                      stripped of punctuation exactly as verse_shuffle is,
                      but NOT shuffled. Added 2026-11-23 so verse_shuffle
                      has a matched pair: the only difference between the
                      two is word order. (Comparing verse_shuffle with
                      random_verse changes word order, casing and
                      punctuation all at once.) Built from the stored refs
                      without drawing more random numbers, so the four
                      original conditions are byte-identical to before.

For each model, the report also gives *pair agreement*: the share of
verse_shuffle chunks whose prediction matches the prediction for their
own unshuffled verse_flat twin. A model that ignores word order scores
1.0. The lower the score, the more the model's verdict depends on order.

The first three should all sit near the floor if the classifier is
reading register. random_verse is a different kind of number: the share
of an arbitrary KJV verse the model calls prophetic, which is the fair
comparison for "is X% prophetic high?".

Usage:
    python src/null_control.py [--n 1000] [--seed 143]
"""
import argparse
import glob
import json
import os
import random
import re
import sys
from collections import Counter
from math import comb

import joblib
import numpy as np
import pandas as pd
import spacy

from extract_features import general_features, prophetic_features, pos_features, WORD_RE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KJV_DIR = os.path.join(REPO_ROOT, "data", "Bible-kjv")
SEED_SET_PATH = os.path.join(REPO_ROOT, "data", "seed_set.csv")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FEATURES_OUT = os.path.join(RESULTS_DIR, "null_control_features.csv")
PRED_OUT = os.path.join(RESULTS_DIR, "null_control_predictions.csv")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "null_control_summary.txt")

CONDITIONS = ["word_salad_uniform", "word_salad_freq", "verse_shuffle", "verse_flat", "random_verse"]
NOISE_CONDITIONS = ["word_salad_uniform", "word_salad_freq", "verse_shuffle"]
MODELS = [
    ("", "full"), ("_narrow", "narrow"), ("_nostruct", "nostruct"), ("_normttr", "normttr"),
    ("_struct", "struct"), ("_structonly", "structonly"),
]

# Transfer results to read against the floor (mechanical chunking, prophetic
# share, from results/transfer_predictions*.csv -- recomputed below, not
# hardcoded, so this stays correct if those files are regenerated).
TRANSFER_PRED_FILES = {label: f"transfer_predictions{suffix}.csv" for suffix, label in MODELS}


def load_kjv_verses():
    paths = sorted(glob.glob(os.path.join(KJV_DIR, "*.json")))
    paths = [p for p in paths if not os.path.basename(p).startswith("Books")]
    if not paths:
        sys.exit(f"error: no KJV JSON in {KJV_DIR} -- run src/setup_data.py --kjv-only first")
    verses = []
    for p in paths:
        with open(p, encoding="utf-8-sig") as f:
            book = json.load(f)
        if "chapters" not in book:
            continue
        for ch in book["chapters"]:
            for v in ch["verses"]:
                verses.append((f"{book['book']} {ch['chapter']}:{v['verse']}", v["text"].strip()))
    return verses


def build_chunks(n, seed):
    rng = random.Random(seed)
    seed_df = pd.read_csv(SEED_SET_PATH)
    seed_refs = set(seed_df["ref"])
    lengths = [len(WORD_RE.findall(t.lower())) for t in seed_df["text"]]

    verses = load_kjv_verses()
    tokens = [w for _, t in verses for w in WORD_RE.findall(t.lower())]
    freq = Counter(tokens)
    types = sorted(freq)
    type_weights = [freq[w] for w in types]
    # Verses outside the seed set, so nothing the model trained on leaks in.
    pool = [(r, t) for r, t in verses if r not in seed_refs and 5 <= len(t.split()) <= 60]

    rows = []
    for i in range(n):
        k = rng.choice(lengths)
        rows.append(("word_salad_uniform", i, "", " ".join(rng.choice(types) for _ in range(k))))
    for i in range(n):
        k = rng.choice(lengths)
        rows.append(("word_salad_freq", i, "", " ".join(rng.choices(types, weights=type_weights, k=k))))
    flat = []
    for i in range(n):
        ref, text = rng.choice(pool)
        words = WORD_RE.findall(text.lower())
        flat.append((ref, " ".join(words)))
        rng.shuffle(words)
        rows.append(("verse_shuffle", i, ref, " ".join(words)))
    for i in range(n):
        ref, text = rng.choice(pool)
        rows.append(("random_verse", i, ref, text))
    # No rng calls below this line: the conditions above stay identical.
    for i, (ref, text) in enumerate(flat):
        rows.append(("verse_flat", i, ref, text))

    return pd.DataFrame(rows, columns=["condition", "chunk_id", "source_ref", "text"])


def extract(df):
    nlp = spacy.load("en_core_web_sm")
    texts = df["text"].tolist()
    feats = pd.concat(
        [
            pd.DataFrame([general_features(t) for t in texts]),
            pd.DataFrame([prophetic_features(t) for t in texts]),
            pd.DataFrame([pos_features(d) for d in nlp.pipe(texts)]),
        ],
        axis=1,
    )
    from structural_features import featurize_texts
    struct = featurize_texts(texts)
    return pd.concat([df.reset_index(drop=True), feats, struct], axis=1)


def transfer_prophetic_share(label):
    path = os.path.join(RESULTS_DIR, TRANSFER_PRED_FILES[label])
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path)
    col = next((c for c in t.columns if c.startswith("pred")), None)
    if col is None:
        return None
    return t.groupby("corpus")[col].apply(lambda s: (s == "prophetic").mean())


def binom_tail(k, n, p):
    """One-sided exact binomial P(X >= k) for X ~ Bin(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def pericope_vs_floor(label, floor):
    path = os.path.join(RESULTS_DIR, "transfer_pericopes_predictions.csv")
    if not os.path.exists(path):
        return []
    p = pd.read_csv(path)
    col = f"pred_{label}"
    lines = [f"  hand-picked pericopes vs floor (one-sided exact binomial, H0: p = {floor:.3f}):"]
    for (corpus, pid), s in p.groupby(["corpus", "pericope_id"])[col]:
        n, k = len(s), int((s == "prophetic").sum())
        pv = binom_tail(k, n, floor)
        flag = "  **" if pv < 0.05 / 12 else ("  *" if pv < 0.05 else "")
        lines.append(f"    {corpus:<18} {pid:<30} {k:>2}/{n:<3} = {k / n:.2f}  p={pv:.4f}{flag}")
    lines.append("    * p < 0.05   ** survives Bonferroni across the 12 pericopes (p < 0.0042)")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="chunks per condition")
    ap.add_argument("--seed", type=int, default=143)
    args = ap.parse_args()

    df = build_chunks(args.n, args.seed)
    print(f"built {len(df)} chunks ({args.n} per condition, seed={args.seed})")
    feats = extract(df)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    feats.to_csv(FEATURES_OUT, index=False)

    out = df.copy()
    models = [(s, l) for s, l in MODELS
              if os.path.exists(os.path.join(RESULTS_DIR, f"classifier{s}.joblib"))]
    for suffix, label in models:
        bundle = joblib.load(os.path.join(RESULTS_DIR, f"classifier{suffix}.joblib"))
        out[f"pred_{label}"] = bundle["pipeline"].predict(feats[bundle["feature_cols"]].values)
    out.to_csv(PRED_OUT, index=False)

    lines = [f"Null-control baseline (n={args.n} per condition, seed={args.seed})", ""]
    for _, label in models:
        mix = (
            out.groupby("condition")[f"pred_{label}"]
            .value_counts(normalize=True).unstack().fillna(0)
            .reindex(CONDITIONS).round(3)
        )
        lines.append(f"--- {label} model: predicted-label mix per condition ---")
        lines.append(mix.to_string())
        sh = out[out.condition == "verse_shuffle"].sort_values("chunk_id")[f"pred_{label}"].values
        fl = out[out.condition == "verse_flat"].sort_values("chunk_id")[f"pred_{label}"].values
        lines.append(f"  pair agreement (shuffled vs. same verse unshuffled): {(sh == fl).mean():.3f}")
        tr = transfer_prophetic_share(label)
        if tr is not None:
            floor = mix.loc[NOISE_CONDITIONS, "prophetic"].max()
            base = mix.loc["random_verse", "prophetic"]
            lines.append(f"  noise floor (max over noise conditions): {floor:.3f}")
            lines.append(f"  KJV base rate (random_verse):            {base:.3f}")
            for corpus, share in tr.items():
                lines.append(
                    f"  {corpus:<18} {share:.3f} prophetic  "
                    f"({share - floor:+.3f} vs floor, {share - base:+.3f} vs base rate)"
                )
            lines.extend(pericope_vs_floor(label, floor))
        lines.append("")
    report = "\n".join(lines)
    print(report)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"wrote {PRED_OUT}, {FEATURES_OUT}, {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
