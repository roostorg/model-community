# What's new in Sentinel 2.0

Sentinel 1.0 gave you an index and a score. Getting a *good* score still meant
hand-building the index, guessing at hyperparameters, and re-encoding your corpus every
time you wanted to try something different.

Version 2.0 is about closing that loop: build an index in one call, resize it without
re-encoding, measure which settings actually work, and see which text drove a score.

The main changes:

| | |
|---|---|
| [One-line index building](#build-an-index-in-one-call) | `from_texts()` replaces an eight-step recipe with two silent failure modes |
| [Resizing without re-encoding](#resize-an-index-without-re-encoding) | `subsample()` makes index size an affordable thing to explore |
| [Six summarize metrics, not two](#more-ways-to-summarize-and-ways-to-measure-which-one-works) | Plus explainability, and the evaluation metrics to choose between them |
| [A tuning harness](#tune-with-the-simulation-harness) | Measure configurations on your own labelled data, numpy-only |
| [Index sweep axes](#sweep-index-size-and-ratio) | Grid search over index size and balance, encoding the observations once |
| [Explanations that survive a reload](#explanations-survive-a-reload) | The corpus is persisted, so explanations name text rather than row numbers |
| [Reproducible loading](#reproducible-loading) | A saved index behaves identically on every load |
| [Run it in a container](#run-it-in-a-container) | Dockerfiles for GPU and CPU-only use |

Migration notes are at the end: [Migrating from 1.0](#migrating-from-10).

## Build an index in one call

`SentinelLocalIndex.from_texts()` replaces an eight-step recipe, two steps of which fail
*silently* when you skip them: omit `normalize_embeddings=True` and the similarity maths
quietly returns wrong numbers, and omit the corpus and you lose explanations. Neither
raises. Neither warns.

```python
from sentinel import SentinelLocalIndex

index = SentinelLocalIndex.from_texts(
    positive_texts=["...examples of the rare class..."],
    negative_texts=["...examples of ordinary content..."],
    neg_to_pos_ratio=5.0,
    seed=42,
)
```

Surplus negatives are dropped *before* encoding rather than after. Encoding is per-text
and is the only expensive step, so at a 1:1 ratio against 1,000 positives, passing
100,000 negatives no longer means paying to encode 99,000 rows that never reach the index.

## Resize an index without re-encoding

Encoding a sentence produces the same numbers regardless of which index it ends up in, so
a small index is just a large one with rows removed. `subsample()` copies the rows you
want instead of re-running the model:

```python
smaller = index.subsample(n_positive=1000, neg_to_pos_ratio=5.0, seed=42)
```

It returns a new index and never mutates the original, which matters when you loop over
sizes: if each call shrank the receiver, run two would start from run one's leftovers and
every later result would be quietly wrong.

Each side draws from its own seeded generator. Sharing one would couple them, so a
configuration that kept every positive (and therefore drew nothing) would select
different negatives from one that subsampled - two cells meant to differ along one axis
would differ along two.

## Tune with the simulation harness

`sentinel.simulation` answers "which settings work best on *my* data?". It is numpy-only,
with no Ray, S3 or experiment trackers, and it separates two things that are easy to
confuse:

- The **summarize metric** (the aggregator) is how a group's many per-observation scores
  become one number. This is what you sweep.
- The **evaluation metric** is how good the resulting separation is. You do not choose
  one; every row reports three families, so you can tune for whatever matters.

Why the summarize metric matters so much: picture a hateful podcast episode with 97 dull
segments and 3 nasty ones, against a normal episode of 100 unremarkable segments.

| Episode | Segment scores | Average | Maximum |
|---|---|---|---|
| Hateful | 97 x 0.05, 3 x 0.90 | 0.0755 | 0.90 |
| Normal | 100 x 0.06 | 0.06 | 0.08 |

Averaging drowns the signal; the maximum finds it. But the maximum is fragile - one odd
segment flags an innocent episode - which is why the default is `skewness`, which looks
for a *pattern* of spikes and does not care how long the episode is.

```python
from sentinel.simulation import LabeledGroup, score_groups, compare_aggregators
import pandas as pd

groups = [
    LabeledGroup(name="source_a", label=1, observations=[...]),
    LabeledGroup(name="source_b", label=0, observations=[...]),
]

scored = score_groups(index, groups, top_k=5)     # expensive, once
pd.DataFrame(compare_aggregators(scored))          # cheap, all six aggregators
```

## More ways to summarize, and ways to measure which one works

1.0 gave you two summarize metrics: `mean_of_positives` and `skewness`. That is not much of
a choice, and no way at all to know whether either suited your data.

2.0 brings the count to six and adds the means to decide between them.

**Four new summarize metrics** in `sentinel.score_formulae`, alongside the original two:

| Metric | Shape it looks for |
|---|---|
| `skewness` (default) | A few spikes in an otherwise flat set of scores, regardless of count |
| `mean_of_positives` | Overall level among scores that cleared the floor |
| `top_k_mean` | The strongest handful of signals |
| `percentile_score` | A high percentile, robust to single outliers |
| `softmax_weighted_mean` | Smoothly emphasises higher scores |
| `max_score` | The single strongest signal |

**Explainability on every result.** `RareClassAffinityResult` now carries
`aggregation_name`, `aggregation_stats` and per-text `explanations` - which neighbours were
closest, their similarities, and the contrastive terms behind the score. You can see *why* a
source scored as it did, not just that it did.

**Evaluation metrics.** Three families, reported on every result row, because "good" depends
on what you are building:

| Family | Metrics | Answers |
|---|---|---|
| Ranking | `roc_auc`, `recall_at_n`, `precision_at_n`, `rank_ratio` | Do the known positives come out on top? Right for a review queue. |
| Threshold | `precision`, `recall`, `f1`, `false_positive_rate` | Of the sources I flagged, how many were right? Right for automatic action. |
| Separation | `mean_separation`, `cohens_d`, `ks_statistic` | How far apart are the two populations, independent of any cutoff? Right for monitoring drift. |

You do not pick a family. All three are computed, so you read whichever matches your use
case - and you can see when they disagree, which they do: an aggregator can rank well while
having a mediocre best F1.

**A way to compare them side by side.** `DEFAULT_AGGREGATORS` is a name-to-function map of
all six, so `compare_aggregators` can evaluate every one of them on a single set of scores,
and `evaluate_groups` can measure any one of them on its own:

```python
from sentinel.simulation import DEFAULT_AGGREGATORS, compare_aggregators, evaluate_groups

# All six, ranked however you like.
pd.DataFrame(compare_aggregators(scored)).sort_values("roc_auc", ascending=False)

# Or one at a time, with all three families reported.
evaluate_groups(scored, DEFAULT_AGGREGATORS["skewness"], aggregator_name="skewness")
```

This is what turns six options into a decision you can defend with evidence rather than a
default you inherited.

## Sweep index size and ratio

`run_grid_search` can now sweep the index itself, not just `top_k` and the threshold:

```python
pd.DataFrame(run_grid_search(
    index, groups,
    n_positive_values=[1000, 5000, 10000],
    neg_to_pos_ratios=[0.2, 1.0, 5.0],
    top_k_values=[3, 5, 10],
    index_seed=42,
))
```

The arguments are ordered to match the loop nesting, which is itself ordered by cost:

```
for n_positive in n_positive_values:      # cheap: subsample()
  for ratio in neg_to_pos_ratios:         # cheap: subsample()
    for top_k in top_k_values:            # re-scores
      for min_score in min_score_values:  # cheap
        for aggregator in aggregators:    # cheap
```

**Observations are encoded once for the whole sweep.** An observation's embedding depends
on the encoder, never on the index it is scored against, and a subsampled index shares its
parent's model - so re-encoding per pass was recomputing identical numbers. On a 2x2x3
sweep over 320 observations this took a real run from **2.99s to 0.52s, a 5.7x saving**,
producing byte-identical rows. The saving grows with the grid, because the one encoding
pass is amortised over more scoring passes.

Set `cache_observation_embeddings=False` if memory is tight; the cache holds one embedding
per observation.

Rows report both what you asked for and what you got, because a request is clipped when
the index is smaller than requested:

```
index_n_positive=10   -> index_n_positive_actual=10, index_n_negative_actual=20
index_n_positive=999  -> index_n_positive_actual=15, index_n_negative_actual=30
```

Without the actual counts, those two rows could look like a genuine size sweep when they
describe nearly the same index.

## Explanations survive a reload

In 1.0, `save()` wrote only embeddings and config. After any reload the corpus was gone,
so explanations reported the row number of a match instead of the matched sentence.

2.0 writes a `corpus.json` beside the embeddings. It is written **unconditionally**, with
nulls when there is no corpus, so it can never describe rows from an earlier save to the
same path. Saving without a corpus therefore clears any corpus already there - which is
deliberate, because keeping it is only correct if the new embeddings are the same rows in
the same order, and nothing enforces that.

Alignment is defended at three points: a mismatched corpus is refused at save time,
discarded with a warning at load time, and carried through row-for-row whenever rows are
dropped. Degrading to row numbers is recoverable; naming the wrong sentence is not.

Indices saved before this file existed simply lack it and continue to load normally.

## Reproducible loading

`load()` downsamples negatives to the requested ratio, and that choice is random. In 1.0
it was unseeded, so a saved index behaved like a slightly different model on every load -
on the shipped example index, the same text scored 0.028636 on one load and 0.011523 on
the next.

```python
index = SentinelLocalIndex.load(path="...", seed=42)
```

Seeding uses a private generator, so your own randomness is untouched.

## Run it in a container

2.0 ships Dockerfiles, so you can try Sentinel without resolving a Python and PyTorch
environment first. There are two: the default is GPU-capable, and `Dockerfile.cpu` is a
smaller CPU-only build for when you are only scoring.

```bash
docker build -t sentinel .                       # GPU-capable
docker build -f Dockerfile.cpu -t sentinel:cpu . # smaller, CPU only
```

See the README for running the demo and mounting your own scripts.

## Migrating from 1.0

Most code needs no change. The breaking changes are deliberate and narrow.

**Python 3.10 is now the minimum.** 1.0 declared support for 3.9, but never tested it - CI has
run 3.10 to 3.12 only. That claim had also become impossible to honour: current `torch`
requires 3.10 or newer, so a 3.9 install had to silently resolve to an older, untested torch.
Raising the floor is a breaking change, which is why it belongs in a major release. Dropping
3.9 also lets three backport packages go - `importlib-metadata`, `importlib-resources` and
`zipp` - whose functionality is in the standard library from 3.10.

**Arguments after the first are now keyword-only** on `calculate_rare_class_affinity`,
`from_texts` and `load`. This is what lets a new argument sit in a logical place instead
of being appended to an eleven-parameter list forever.

```python
# 1.0 - still works, but only because the first argument is positional
index.calculate_rare_class_affinity(texts)

# 1.0 positional style - no longer valid
index.calculate_rare_class_affinity(texts, 10)

# 2.0
index.calculate_rare_class_affinity(texts, top_k=10)
```

Every call site in this repository already used keywords, and the documentation always
taught that style, so in practice this is unlikely to affect you.

**Grid-search index columns are prefixed.** `evaluate_groups` returns an `n_positive`
meaning the number of positive *groups* in your evaluation set. The index size briefly
shared that name and silently overwrote it. If you read grid-search output, rename:

| Old | New |
|---|---|
| `n_positive` (index size) | `index_n_positive` |
| `neg_to_pos_ratio` | `index_neg_to_pos_ratio` |
| `n_positive_actual` | `index_n_positive_actual` |
| `n_negative_actual` | `index_n_negative_actual` |

`n_positive` now unambiguously means the positive-group count, as `evaluate_groups`
always documented. Writing a column twice raises instead of overwriting.

**Rows are still plain dicts**, so `pd.DataFrame(rows)` keeps working.

## Known limitations

- `top_k` still forces a re-score. Caching the neighbour search would make it as cheap as
  the index axes, but that is a deeper change.
- `RareClassAffinityResult.observation_scores` is keyed by observation text, so duplicate
  observations within one group collapse into a single entry.
