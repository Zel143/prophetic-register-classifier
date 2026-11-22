# Null-control baseline

`src/null_control.py` · outputs in `results/null_control_*.{csv,txt}`

## Why

Every transfer number in this project has been read against an implicit floor of
0% prophetic. That floor is wrong. A three-class classifier fed text with *no*
register will still put some share of it in each class. If that share is already
~30%, then "Sibylline Oracles is 37% prophetic" means much less than it sounds.
This pass measures the real floor.

The idea comes from TempleOS (Terry A. Davis). Its built-in oracle drew random words
and random Bible passages and presented them as divine speech, and readers routinely
found meaning in the output. That is the classic case of apophenia, which means
seeing a pattern in noise. The question here is the machine version: does a
stylometric classifier also "hear prophecy" in random text? As with the rest of the
project, this measures style only. It makes no claim about any text's truth,
inspiration or scriptural status.

## Conditions

Each condition has 1000 chunks drawn from the KJV with seed 143. Noise-chunk lengths
are sampled from the seed set's own verse-length distribution, which keeps `n_words`
and the length-sensitive features inside the training range. The verse-based
conditions only draw from verses that are **not** in the seed set.

| Condition | What it is | Role |
|---|---|---|
| `word_salad_uniform` | random KJV word types, drawn uniformly | pure noise (closest to the TempleOS random-word oracle) |
| `word_salad_freq` | random KJV tokens, weighted by frequency | noise that keeps the KJV's function-word profile |
| `verse_shuffle` | real KJV verses with word order shuffled | right vocabulary, no syntax |
| `random_verse` | real KJV verses, unchanged | KJV base rate (the TempleOS random-passage oracle) |

The **noise floor** is the highest prophetic share among the three noise conditions.

## Results (prophetic share)

| Model | uniform salad | freq salad | shuffle | random verse | **floor** |
|---|---|---|---|---|---|
| full | 0.067 | 0.319 | 0.345 | 0.395 | **0.345** |
| narrow | 0.373 | 0.317 | 0.325 | 0.320 | **0.373** |
| nostruct | 0.025 | 0.222 | 0.226 | 0.233 | **0.226** |
| normttr | 0.274 | 0.300 | 0.302 | 0.288 | **0.302** |

### Finding 1: syntax contributes almost nothing at the chunk level

`verse_shuffle` scores almost exactly like `random_verse` under every model, and so
does frequency-weighted word salad. Scrambling word order, or replacing the verse
with random KJV tokens altogether, barely changes the verdict. At the sentence-chunk
level the classifier is reading a bag-of-words and frequency profile, not anything
that depends on how the words are arranged. The README frames "register" as verb
mood, address form and structure. The structural part of that is not what the model
is using.

### Finding 2: most mechanical-chunking transfer results sit at the noise floor

| Corpus | normttr | vs floor | z | nostruct | vs floor | z |
|---|---|---|---|---|---|---|
| 1 Enoch | 0.262 | −0.040 | −3.2 | 0.209 | −0.017 | −1.6 |
| Bahman Yasht | 0.150 | −0.152 | −8.8 | 0.074 | −0.152 | −9.7 |
| Sibylline Oracles | 0.373 | **+0.071** | **7.2** | 0.282 | **+0.056** | **6.3** |

(The z values are one-sample against the floor and ignore the floor's own sampling
error, about ±1.5pp at n=1000. The Sibylline margin is large enough to survive that.)

Under mechanical chunking, 1 Enoch as a whole is **indistinguishable from KJV word
salad**. Bahman Yasht falls clearly *below* noise: it is actively read as
non-prophetic, which supports the "doesn't transfer" reading in `classifier.md`.
Sibylline Oracles is the only corpus that clears the floor, under both reduced
feature sets.

### Finding 3: the 1 Enoch opening theophany survives the floor

Hand-picked pericopes, tested with a one-sided exact binomial test against each
model's floor, with Bonferroni correction across the 12 pericopes (p < 0.0042):

| Pericope | full | narrow | nostruct | normttr |
|---|---|---|---|---|
| 1 Enoch ch1_theophany (vv. 1-8) | 7/10, p=.024 | **9/10, p=.0009** | **8/10, p=.0002** | **9/10, p=.0002** |
| Sibylline book1_prophecy_of_christ | n.s. | n.s. | **10/17, p=.0014** | 9/17, p=.042 |
| 1 Enoch ch10_judgment | 14/25, p=.022 | n.s. | n.s. | n.s. |

Everything else is at or below the floor. The 1 Enoch opening theophany passes
correction under three of the four models, so the project's headline transfer
result holds up against a real null. That result already excludes 1 Enoch 1:9 (see
`decisions.md`). Most other pericopes do not hold up.

## What this changes

- Report transfer results **relative to the floor**, not relative to 0%. A
  prophetic share near 0.30 is the noise level, not weak evidence of transfer.
- The real signal is narrow: one 1 Enoch pericope robustly, and Sibylline Oracles
  modestly. The "transfer is plausible" reading should be scoped to those cases.
- Finding 1 points to a next step. Features that *cannot* survive shuffling, such
  as clause-initial verb patterns, POS bigrams and sentence-position features,
  would test the structural half of the register hypothesis. Right now nothing in
  the feature set tests it.
- The uniform-salad column is unstable across models (0.025 to 0.373). Uniform word
  types are mostly rare words, so the text is far outside the training distribution
  and each model extrapolates differently. The floor is therefore taken as the
  *max* across the noise conditions, which makes it the conservative choice.

## Reproduce

```
python src/setup_data.py --kjv-only   # if data/Bible-kjv/ is missing
python src/null_control.py            # --n 1000 --seed 143 by default
```

The saved models were trained with scikit-learn 1.8.0. Pin that version, because
newer versions warn when they unpickle the models.
