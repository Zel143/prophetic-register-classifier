# Structural (order-sensitive) features

`src/structural_features.py` · models `--features struct` and `--features structonly`
· outputs in `results/*struct*`

## Why

The null control (`null-control.md`) showed that the original feature set can barely
tell a verse from its own word-shuffled copy. That follows from how it was built:
function-word rates, POS proportions and type-token ratios are bag-of-words statistics.
The README defines prophetic register partly by *structure* (address form, formula
placement, clause shape), and that half of the hypothesis had never been tested. These
features can only be computed from word order.

## Features (27)

| Feature | What it measures | Example |
|---|---|---|
| `struct_bg_<A>_<B>` (20) | share of adjacent POS pairs that are A→B, for the 20 most frequent pairs in the seed set (chosen once, saved in `results/struct_bigrams.json`) | AUX→VERB: "shall come" |
| `struct_clause_verb_initial` | share of clauses that start with their verb (ignoring a leading conjunction) | "**Hear**, O heavens" |
| `struct_clause_cc_initial` | share of clauses that start with a coordinating conjunction | "**And** he said" |
| `struct_clause_subj_initial` | share of clauses that start with their subject | "**The LORD** spake" |
| `struct_vs_order` | among clauses with a subject, share where the verb comes first | "thus **saith** the LORD" |
| `struct_first_verb_relpos` | position of the first verb, as a fraction of the chunk | |
| `struct_mean_dep_dist` | mean distance between each word and its syntactic head | |
| `struct_clauses_per_10w` | clause density | |

Punctuation is ignored throughout. The shuffled null condition has none, so including
it would let a feature spot "shuffled" from missing commas instead of from word order.

**Archaic verb forms are modernized before parsing, for these features only.** spaCy's
modern-English model tags *saith* as a preposition. Without this step, the parser misses
the verb in "thus saith the LORD", the formula the project rests on. The rules are a
fixed map for irregular forms (saith→says, hath→has, spake→spoke) and -eth→-s/-es/-ies,
with each stem checked against the KJV's own vocabulary. So *cometh* becomes *comes*,
while *teeth* is left alone. Capitalized names (Seth, Nazareth) are never changed. Every
corpus and every null condition goes through the same function.

## Results

### Seed-set cross-validation (5-fold, macro-F1)

| Feature set | Columns | Macro-F1 |
|---|---|---|
| normttr (current recommended) | 20 | 0.666 |
| **struct** (normttr + structural) | 47 | 0.663 |
| **structonly** | 27 | **0.565** |
| chance (3 balanced classes) | | ≈0.33 |

**Structure alone carries real register signal.** A model that sees no word identities
or counts, only POS order and clause shape, gets 0.565, far above chance. Its strongest
coefficients are the patterns biblical-studies readers would predict:

- **narrative**: conjunction-initial clauses (+0.60), the KJV "And he…" chain; it is
  also *less* verb-initial (−0.38)
- **prophetic**: AUX→VERB (+0.45, "shall come", "will bring"), verb-before-subject
  inversion (+0.20), verb-initial clauses (+0.20)
- **law-wisdom**: the verb comes later in the clause (+0.33), fewer "And"-initial
  clauses (−0.43)

**Combined with normttr, structure adds nothing on CV** (0.663 vs 0.666). On this
510-verse seed set, its signal overlaps with what the bag-of-words features already
capture.

### Shuffle sensitivity: the null control's paired test

The null control now includes `verse_flat`: the *same* verses as `verse_shuffle`,
processed identically (lowercased, no punctuation) but left unshuffled. Pair agreement
is the share of shuffled verses that get the same prediction as their unshuffled twin.

| Model | Pair agreement | Floor (prophetic) |
|---|---|---|
| full | 0.853 | 0.345 |
| narrow | 0.787 | 0.373 |
| nostruct | 0.790 | 0.226 |
| normttr | 0.816 | 0.302 |
| **struct** | **0.549** | **0.498** |
| **structonly** | **0.450** | 0.354 |

The new features do what they were built for: when the words are shuffled, the
structure-only model changes its answer 55% of the time.

### The catch: scrambled order looks prophetic to the struct model

The combined struct model calls **half of all shuffled verses prophetic** (0.498, up
from 0.34 for the same verses unshuffled). The likely cause is that prophetic KJV poetry
already reads as "disordered" to a modern-English parser: inverted verbs, fronted
objects, vocatives in the middle of a line. Random word order produces some of the same
surface patterns, so in the combined model, disorder itself gets read as a prophetic
cue. The structure-only model doesn't show this problem (floor 0.354, roughly the KJV
base rate).

This is a real limitation of using a modern parser on archaic poetry, not something
these features can fix. A parser trained on Early Modern English, or Hebrew-side
features from MACULA (the v2 plan), would be the way to separate "poetic inversion" from
"noise".

### Transfer, against each model's floor

| | struct | structonly |
|---|---|---|
| 1 Enoch | 0.315 (−0.18 vs floor) | 0.347 (−0.01) |
| Bahman Yasht | 0.302 (−0.20) | 0.281 (−0.07) |
| Sibylline Oracles | 0.400 (−0.10) | 0.285 (−0.07) |

No corpus clears the floor under either structural model. Under struct, the floor is
inflated for the reason above, so that comparison is stricter than it looks.

**Pericopes:** the 1 Enoch ch1_theophany (vv. 1-8) is still the most robust result.
Under **structonly** it scores 7/10 against a floor of 0.354 (p = 0.028, which does not
survive Bonferroni across 12). So it is the only pericope with even nominal evidence
*from word order alone*, on top of clearing the floor with Bonferroni under three of the
bag-of-words models. Under struct it scores 8/10 against the inflated 0.498 floor
(p = 0.053).

## What this means for the hypothesis

1. **Register has a structural component in the canon itself.** Structure alone
   separates prophetic, narrative and law-wisdom at 0.565 macro-F1, and the features
   that drive it are linguistically meaningful.
2. **That structural signal doesn't yet transfer measurably.** Part of the reason is the
   modern parser: it can't tell poetic inversion from noise, which shows up as the
   struct model's inflated floor.
3. **The theophany result now has support from two independent angles**, vocabulary and
   structure, though the structural support is only nominal.

## Reproduce

```
python src/structural_features.py                # sidecar CSVs + bigram list
python src/train_classifier.py --features struct
python src/train_classifier.py --features structonly
python src/eval_pericopes.py
python src/null_control.py
```
