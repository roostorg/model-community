# Sentinel

**Version 2.0** — see [V2_FEATURES.md](V2_FEATURES.md) for what changed and how to migrate.

> **About this copy.** This folder is a snapshot of
> [Roblox/Sentinel](https://github.com/Roblox/Sentinel) at commit
> [`66c6985`](https://github.com/Roblox/Sentinel/commit/66c69854e806bd9f5c866379b6c75f3fcb4d2e6c).
> The upstream repository stays the canonical source for issues and pull requests.
>
> Two upstream directories are omitted here, because they are stored with
> [Git LFS](https://git-lfs.com) in Sentinel and this repository does not use LFS:
> `examples/data/` (the SPLC and neutral example corpora) and
> `examples/hate_speech_model/` (a 25 MB prebuilt index). Both are generated, so you
> can rebuild them: run the two scripts in [`examples/scripts/`](examples/scripts) to
> fetch the corpora, then run
> [`examples/sentinel_against_hate.ipynb`](examples/sentinel_against_hate.ipynb), which
> writes the index to `examples/hate_speech_model/`. The [`tests/`](tests) suite needs
> neither directory.

## Overview

Roblox Sentinel, part of the Roblox Safety Toolkit, is a Python library designed specifically for realtime detection of extremely rare classes of text by using contrastive learning principles. While traditional classifiers struggle with highly imbalanced datasets, Sentinel excels by:

1. Collecting recent observations from a single source (e.g., recent messages from a user)
2. Calculating individual observation scores using embedding similarity
3. Aggregating these scores using statistical measures like skewness to detect patterns

By prioritizing recall over precision, Sentinel serves as a high-recall candidate generator for more thorough investigation. This approach is particularly effective for applications where rare patterns are critical to identify. Rather than treating each message in isolation, Sentinel analyzes patterns across messages to identify concerning behavior.

## What's new in 2.0

Version 1.0 gave you an index and a score. Getting a *good* score still meant
hand-building the index, guessing at hyperparameters, and re-encoding your corpus every
time you wanted to try something different. 2.0 closes that loop:

| | |
|---|---|
| [`from_texts()`](#creating-a-new-index) | Build an index in one call, instead of an eight-step recipe where two steps fail silently if skipped |
| [`subsample()`](#resizing-an-index) | Resize an index without re-encoding — a 3x3 grid in 9 ms rather than 12.5 s |
| [`sentinel.simulation`](#simulation-based-tuning) | Measure which aggregator and hyperparameters suit your data |
| [Index sweeps](#simulation-based-tuning) | Grid search over index size and negative ratio, encoding the observations only once |
| [Persisted corpus](#storage-options) | Explanations name the matched sentence after a reload, not a row number |
| [Seeded loading](#quick-start) | A saved index behaves identically on every load |

Full detail and migration notes: [V2_FEATURES.md](V2_FEATURES.md).

### Aggregation options and explainability

Sentinel includes multiple aggregation strategies and built‑in explainability to help you tune for your use case and understand why a score was assigned.

- Aggregators (in `sentinel.score_formulae`):
    - `skewness(scores, min_size_of_scores=10)`: default, pattern‑oriented and robust to message count
    - `top_k_mean(scores, k=3)`: focuses on the strongest signals
    - `percentile_score(scores, q=90.0)`: robust to outliers via a percentile over positives
    - `softmax_weighted_mean(scores, temperature=1.0)`: smoothly emphasizes higher scores
    - `max_score(scores)`: simplest, picks the highest positive score

- Explainability (in results):
    - Each call to `calculate_rare_class_affinity` returns a `RareClassAffinityResult` with:
        - `aggregation_name`, `aggregation_stats`: which aggregator was used and key params
        - `explanations`: per‑text details including top‑K positive/negative similarities, contrastive components, and neighbor snippets (when available)

## Previous Release (v1)

- [Roblox Technical Announcement](https://about.roblox.com/newsroom/2025/08/open-sourcing-roblox-sentinel-preemptive-risk-detection)

## Terminology

In Sentinel's codebase:
- **Positive examples**: Examples of text that belong to the rare class of interest (e.g., harmful, unsafe, or critical content)
- **Negative examples**: Examples of text that belong to the common class (e.g., safe, neutral, or typical content)

## Installation

```bash
pip install .
```

By default `sentinel` doesn't pull in all transitive dependencies, specifically avoiding pulling in sentence transformers and its dependencies (torch).
To pull them in as well, use:

```bash
pip install '.[sbert]'
```

## Run with Docker

If you'd rather not set up Python and Poetry locally, you can run Sentinel in a
container. A container is just an isolated, pre-configured "box" that already
has the right Python version and all dependencies installed, so it runs the
same way on any machine that has [Docker](https://docs.docker.com/get-docker/).

### 1. Build the image

There are two Dockerfiles so you can pick the one that fits your needs:

| File | torch build | Approx. image size | Use when |
|------|-------------|--------------------|----------|
| `Dockerfile` (default) | GPU (CUDA) on x86_64 Linux; CPU on arm64 | ~6–8 GB on x86_64 Linux; ~1 GB on arm64 | You want to run on a GPU, or want the fully pinned (`poetry.lock`) install |
| `Dockerfile.cpu` | CPU-only (all architectures) | ~1.5–2.5 GB | You just need CPU inference (the common case) — much smaller and faster to pull |

> **Note on image size and architecture.** The sizes above assume you build on
> an **x86_64 (amd64) Linux** host. The default `Dockerfile` only pulls in the
> full NVIDIA CUDA libraries (several GB) there, because `poetry.lock` marks
> those packages as x86_64-Linux-only. If you build on **Apple Silicon (arm64)**
> — e.g. an M-series Mac — Docker produces an arm64 image, the CUDA packages are
> skipped, and the default image is CPU-only and much smaller (~1 GB). In that
> case both Dockerfiles end up CPU-only and similar in size. `Dockerfile.cpu`
> installs the CPU build of torch explicitly, so it is guaranteed CPU-only on
> **every** architecture (including x86_64 Linux).

Sentinel's own code runs on CPU, so for most people `Dockerfile.cpu` is the
better choice. On x86_64 Linux the default `Dockerfile` pulls in the full NVIDIA
CUDA libraries (several GB) that are only useful if you actually have a GPU.

From the repository root, build whichever you want:

```bash
# Default (GPU-capable) image, tagged "sentinel"
docker build -t sentinel .

# Smaller CPU-only image, tagged "sentinel:cpu"
docker build -f Dockerfile.cpu -t sentinel:cpu .
```

Both install the `sentinel` library together with the `sbert` extra
(sentence-transformers + torch). The first build downloads a lot of
dependencies, so it can take several minutes.

> The examples below use the `sentinel` tag. If you built the CPU image, just
> swap in `sentinel:cpu` (e.g. `docker run --rm sentinel:cpu`).

### 2. Run the demo

```bash
docker run --rm sentinel
```

This runs `examples/beginner_demo.py`, which builds a tiny in-memory index and
scores a batch of example messages. On the first run it downloads the
`all-MiniLM-L6-v2` embedding model, so give it a moment.

To avoid re-downloading that model on every run, mount a local folder as the
model cache:

```bash
docker run --rm -v "$(pwd)/.hf-cache:/home/sentinel/.cache/huggingface" sentinel
```

### 3. Run your own script

The default command is just a starting point. Override it to run any other
script in the image — for example the threshold tuning script:

```bash
docker run --rm sentinel python examples/Example_Threshold_Script.py
```

Or drop into an interactive shell to explore:

```bash
docker run --rm -it sentinel bash
```

To run a script from your own machine (not baked into the image), mount your
current directory into the container's `/app` folder:

```bash
docker run --rm -v "$(pwd):/app" sentinel python your_script.py
```

## Quick Start

```python
from sentinel.sentinel_local_index import SentinelLocalIndex

# Load a previously saved index from a local path
index = SentinelLocalIndex.load(path="path/to/local/index")

# Loading downsamples the negatives to the requested ratio, and that choice is
# random. Pass a seed to make it reproducible, so the same saved index always
# behaves identically - worth doing whenever you are tuning or debugging.
index = SentinelLocalIndex.load(path="path/to/local/index", seed=42)

# Or load from S3
index = SentinelLocalIndex.load(
    path="s3://my-bucket/path/to/index",
    aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
    aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"  # Optional if using environment credentials
)

# Collect recent observations from a single source (e.g., recent messages from a user)
user_recent_messages = [
    "Hey how are you?",
    "What are you doing today?",
    "Do you have any pictures you can share?",
    "Where do you live?",
    "Are your parents home right now?"
]

# Calculate rare class affinity across all observations
result = index.calculate_rare_class_affinity(user_recent_messages)

# Get the overall score (uses skewness by default)
overall_score = result.rare_class_affinity_score
print(f"Overall rare class affinity score: {overall_score:.4f}")

# Examine individual observation scores
for message, score in result.observation_scores.items():
    risk_level = "High" if score > 0.5 else "Medium" if score > 0.1 else "Low"
    print(f"'{message}' - Score: {score:.4f} - Risk: {risk_level}")

# Inspect explainability
print("Aggregator:", result.aggregation_name)
print("Aggregation stats:", result.aggregation_stats)
for message, ex in result.explanations.items():
    print("--", message)
    print("   topk_positive:", ex["topk_positive"])  # scaled similarities
    print("   topk_negative:", ex["topk_negative"])  # scaled similarities
    print("   contrastive:", ex["contrastive"])      # positive_term, negative_term, log_ratio_unclipped
    print("   neighbors (sample):", ex["neighbors"][:2] if ex["neighbors"] else None)
```

## Creating a New Index

```python
from sentinel.sentinel_local_index import SentinelLocalIndex

index = SentinelLocalIndex.from_texts(
    positive_texts=["rare class example", "critical content example"],
    negative_texts=["common class example", "typical content"],
    model_name="all-MiniLM-L6-v2",
)

index.save(path="path/to/local/index", encoder_model_name_or_path="all-MiniLM-L6-v2")
```

`from_texts` encodes both sides, keeps the corpus so explanations can name real
sentences, and applies the correct encoding options. It optionally downsamples the
negatives for you:

```python
index = SentinelLocalIndex.from_texts(
    positive_texts=positive_texts,
    negative_texts=negative_texts,
    neg_to_pos_ratio=5.0,   # keep 5 negatives per positive
    seed=42,                # ...reproducibly
)
```

### Advanced: building an index manually

Use this if you need custom encoding — a different embedding backend, precomputed
vectors, or non-standard encoding arguments. Two steps here fail *silently* if you
skip them: without `normalize_embeddings=True` the similarity maths returns wrong
numbers, and without the corpus you lose explanations. Neither raises an error.

```python
import torch
from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.embeddings.sbert import get_sentence_transformer_and_scaling_fn

# Initialize sentence model and get scaling function
model_name = "all-MiniLM-L6-v2"
model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

# Prepare examples
positive_examples = ["positive message 1", "rare class example", "critical content example"]
negative_examples = ["neutral message 1", "common class example", "typical content"]

# Encode examples
positive_embeddings = model.encode(positive_examples, normalize_embeddings=True)
negative_embeddings = model.encode(negative_examples, normalize_embeddings=True)

# Create the index
index = SentinelLocalIndex(
    sentence_model=model,
    positive_embeddings=positive_embeddings,
    negative_embeddings=negative_embeddings,
    scale_fn=scale_fn,
    positive_corpus=positive_examples,
    negative_corpus=negative_examples,
)

# Save locally - provide the model name when saving
# You must provide the encoder model name as it can't be reliably extracted from a SentenceTransformer instance
# The save method returns the SavedIndexConfig for informational purposes, but it's already saved at the specified location
saved_config = index.save(path="path/to/local/index", encoder_model_name_or_path=model_name)
print(f"Saved index with encoder model: {saved_config.encoder_model_name_or_path}")

# Or save to S3
saved_config = index.save(
    path="s3://my-bucket/path/to/index",
    encoder_model_name_or_path=model_name,
    aws_access_key_id="YOUR_ACCESS_KEY_ID",  # Optional if using environment credentials
    aws_secret_access_key="YOUR_SECRET_ACCESS_KEY"  # Optional if using environment credentials
)
```

## Resizing an index

Encoding a sentence produces the same numbers regardless of which index it ends up in,
so a small index is just a large one with rows removed. `subsample()` copies the rows you
want rather than re-running the model, which is what makes sweeping index size
affordable — on the shipped example index, building a 3x3 grid of configurations takes
about 9 ms against roughly 12.5 s to re-encode the same 16,100 rows.

```python
smaller = index.subsample(n_positive=1000, neg_to_pos_ratio=5.0, seed=42)
```

It returns a new index and never modifies the original, which matters when you loop over
sizes: if each call shrank the receiver, the second run would start from the first run's
leftovers and every later result would be quietly wrong. Corpus texts follow their own
embedding rows, so explanations stay truthful.

## Testing for optimal Thresholds and data ratio's

Usage of the 'examples\Example_Threshold_Script.py' script will allow for quick threshold checks for a variety of ratios, by default these are 10:1, 5:1 and 1:1 ratios. This has predefined example chat logs, and should, show optimal settings for the dataset being used based on an average score and average detection count.
You will be able to get detailed information output in the ratios with the -r or --review flags.

## Choosing an aggregation strategy

Different deployments optimize for different trade‑offs. You can swap in any aggregator using the `aggregation_function` argument:

```python
from sentinel.score_formulae import top_k_mean, percentile_score, softmax_weighted_mean, max_score

texts = ["msg a", "msg b", "msg c"]

# Focus on the strongest few signals
res1 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: top_k_mean(arr, k=3))

# Robust to outliers
res2 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: percentile_score(arr, q=90))

# Smoothly emphasize higher scores
res3 = index.calculate_rare_class_affinity(texts, aggregation_function=lambda arr: softmax_weighted_mean(arr, temperature=0.5))

# Simplest, picks the maximum
res4 = index.calculate_rare_class_affinity(texts, aggregation_function=max_score)
```

Notes:
- All aggregators operate over per‑observation scores where non‑confident observations are already clipped to 0.
- The default `skewness` remains a good choice when user activity volume varies widely.

## Simulation-based tuning

Which aggregator and hyperparameters work best depends on your data, so it helps to measure them on labeled examples. The `sentinel.simulation` module is a small, numpy‑only harness for exactly that — no Ray, S3, or experiment trackers required.

Give it groups of observations with known labels (`1` = rare/positive source, `0` = common/negative source), score them once, then compare every aggregator (and hyperparameter) cheaply:

```python
import pandas as pd
from sentinel.simulation import LabeledGroup, score_groups, compare_aggregators, run_grid_search

groups = [
    LabeledGroup(name="source_a", label=1, observations=["...", "..."]),  # known rare-class source
    LabeledGroup(name="source_b", label=0, observations=["...", "..."]),  # known common-class source
    # ...
]

# Expensive step (runs the model) — done once.
scored = score_groups(index, groups, top_k=5)

# Cheap step — compare all six aggregators on the same scores.
pd.DataFrame(compare_aggregators(scored))

# Or sweep hyperparameters as well.
pd.DataFrame(run_grid_search(index, groups, top_k_values=[3, 5, 10], min_score_values=[0.0, 0.1, 0.25]))
```

Two different things are called "metrics" here, and keeping them apart helps:

- The **summarize metric** (the aggregator) is *how* a group's many per-observation scores become one number. This is what you sweep.
- The **evaluation metric** is *how good* that separation turned out to be. You do not pick one — every row reports all three families below, so you can tune for whatever matters to your use case.

- Ranking: `roc_auc`, `recall_at_n`, `precision_at_n`, `rank_ratio` (do known positives rank at the top?)
- Threshold / classification: `precision`, `recall`, `f1`, `false_positive_rate` at a chosen (or automatically best‑F1) cutoff
- Separation / distribution: `mean_separation`, `cohens_d`, `ks_statistic` (threshold‑free)

### Sweeping the index itself

The grid search can also vary the index, not just `top_k` and the threshold. Index size is
usually the highest-leverage knob, and `subsample()` makes it cheap to explore:

```python
pd.DataFrame(run_grid_search(
    index, groups,
    n_positive_values=[1000, 5000, 10000],   # index size
    neg_to_pos_ratios=[0.2, 1.0, 5.0],       # index balance
    top_k_values=[3, 5, 10],
    index_seed=42,                            # reproducible index configurations
))
```

The arguments are ordered to match the loop nesting, which is itself ordered by cost:
resizing the index is cheap, re-scoring is not, and thresholds and aggregators are free on
the cached scores.

**Observations are encoded once for the whole sweep.** An observation's embedding depends
on the encoder, never on the index it is scored against, so re-encoding on every pass was
recomputing identical numbers. On a 2x2x3 sweep over 320 observations this took a measured
run from **2.99 s to 0.52 s, a 5.7x saving**, with byte-identical results. Pass
`cache_observation_embeddings=False` if memory is tight.

Rows report both the requested and actual index sizes, because a request is clipped when
the index holds fewer examples than you asked for — without `index_n_positive_actual` two
rows can look like a genuine size sweep while describing nearly the same index.

See [examples/sentinel_against_hate.ipynb](examples/sentinel_against_hate.ipynb) for a worked example comparing aggregators on hate‑speech data.

## How It Works

Sentinel uses a two-step process to detect rare classes of text, focusing on high recall for realtime applications:

1. **Individual Observation Scoring**:
   - Each text observation (e.g., message, post) is compared against both rare class examples and common class examples
   - Using embedding similarity, we calculate how close the observation is to each class
   - The observation score is the ratio between rare class similarity and common class similarity
   - Scores > 0.1 indicate closer similarity to rare class examples

2. **Pattern Recognition via Skewness**:
   - Recent individual observation scores from the same source are collected
   - Skewness measures the asymmetry in the distribution of these scores
   - A positive skewness indicates a pattern where most content is common, but with enough rare-class similarities to create a right-skewed distribution
   - This method is resistant to variations in the number of observations, making it ideal for sources with different activity levels
   - By focusing on patterns rather than individual messages, it achieves higher recall for rare phenomena
   - The aggregated score reveals patterns that would be missed when analyzing messages individually

As a high-recall candidate generator, Sentinel identifies potential cases for further investigation, prioritizing not missing true positives even at the cost of some false positives.

## Motivating Use Case

Sentinel was developed to detect extremely rare classes of harmful content where traditional classification approaches fail due to the scarcity of examples. A prominent application was detecting child grooming attempts on Roblox:

1. **The Challenge**: Child grooming patterns are extremely rare in overall communications but devastating when they occur. Traditional classifiers struggle with such imbalanced classes.

2. **The Approach**:
   - Collect recent communications from a single source (e.g., a user's recent chat messages)
   - Score each message individually using contrastive learning to determine similarity to known harmful patterns
   - Aggregate these scores using skewness to detect overall patterns, regardless of message volume
   - Generate candidates for thorough investigation, prioritizing recall over precision

3. **Real-world Impact**: This approach led to over 1,000 NCMEC (National Center for Missing & Exploited Children) reports in just the first few months of deployment at Roblox, significantly improving platform safety.

The same methodology can be applied to any rare text classification problem where:
- Examples of the target class are extremely scarce
- Traditional classifiers would struggle with recall
- Realtime detection is required
- Context across multiple observations from the same source is meaningful
- High recall is prioritized over precision for initial screening

## Storage Options

Sentinel supports both local file storage and S3 storage:

- For local storage, use paths starting with `/` or a relative path
- For S3 storage, use URI format: `s3://bucket-name/path/to/index`

The storage is abstracted using `smart_open`, making it seamless to switch between storage backends.

A saved index is a directory of three files:

| File | Contents |
|------|----------|
| `sentinel_local_index_config.json` | Encoder model name, encoding kwargs, model card |
| `embeddings.safetensors` | The positive and negative embedding tensors |
| `corpus.json` | The original texts behind those embeddings — contents **optional** |

`corpus.json` is what lets explanations name the matched sentence after a reload.
Without those texts, `explanations` falls back to reporting the row number of the
match instead of its text.

The file itself is always written, holding nulls when the index has no corpus, so
that it always describes the embeddings saved beside it. Saving an index without a
corpus therefore clears any corpus already at that path, rather than leaving
behind texts that describe rows which no longer exist. Indices saved before this
file existed simply lack it and continue to load normally.

## Examples
To run the notebook examples
```bash
# Install with examples dependencies
poetry install --with examples
poetry install --extras=sbert
poetry run jupyter notebook
```

## License

Apache License 2.0
