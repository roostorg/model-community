# Reasoning Trace Clustering Experiment

Does adding an LLM's reasoning traces to content embeddings improve clustering for toxicity detection?

## The Experiment

This experiment compares how toxic/safe content clusters when embedded two ways:

1. **Comments Only**: Raw comment text → embedding
2. **Comments + Reasoning**: Comment + gpt-oss-safeguard's chain-of-thought reasoning → embedding

We use KMeans clustering and measure cluster purity (how well clusters separate toxic from safe content).

## Quick Start

```bash
# 1. Install dependencies
uv sync
uv add datasets  # For downloading source data

# 2. Set API keys
export GROQ_API_KEY="your-groq-key"      # For gpt-oss-safeguard
export GEMINI_API_KEY="your-gemini-key"  # For embeddings

# 3. Prepare source data (downloads from HuggingFace)
uv run python prepare_dataset.py --samples 1000

# 4. Run experiment
uv run python clustering_analysis.py --samples 100
```

## What It Does

```
prepare_dataset.py
        │
        ▼
civil_comments_balanced_toxic_0.9.json  ← Balanced toxic/safe samples
        │
        │  clustering_analysis.py --samples N
        │
        ├──► Step 1: Classify with gpt-oss-safeguard
        │    Extract reasoning traces from model's CoT channel
        │
        ├──► Step 2: Generate embeddings (Gemini)
        │    - Comments only
        │    - Comments + reasoning traces
        │
        ├──► Step 3: KMeans clustering
        │    Compare purity, precision, recall, F1
        │
        ▼
    RESULTS
```

## Key Findings

At ~1000 samples, we found:

| k | Metric | Comments Only | + Reasoning | Notes |
|---|--------|---------------|-------------|-------|
| 3 | Precision | **0.96** | 0.75 | Comments more precise |
| 3 | Recall | 0.89 | **0.96** | Reasoning catches more toxic |
| 7 | All metrics | ~0.90 | ~0.90 | Converge at higher k |

**Insight**: Reasoning traces create embeddings that cluster by *model decision* rather than *content similarity*. This shifts behavior toward high recall (catches more toxic) at the cost of precision.

## CLI Options

```bash
uv run python clustering_analysis.py --help

Options:
  --samples, -n    Number of samples (default: 100)
  --clusters, -k   Cluster sizes to test (default: 3,5,7)
  --force, -f      Force regenerate cached data
  --output, -o     Save results to JSON file
```

## Project Structure

```
reasoning-clustering-experiment/
├── clustering_analysis.py    # Main entry point
├── classify_toxicity.py      # gpt-oss-safeguard classification
├── embedding_experiment.py   # Embedding generation
├── prepare_dataset.py        # Dataset download/prep
├── src/
│   ├── config.py             # API keys, model config
│   └── embeddings.py         # Gemini embedding functions
├── toxicity_policy.md        # Classification policy
├── pyproject.toml
└── README.md
```

## Requirements

- Python 3.11+
- API Keys:
  - `GROQ_API_KEY` - For [gpt-oss-safeguard](https://console.groq.com/) via Groq
  - `GEMINI_API_KEY` - For [Gemini embeddings](https://ai.google.dev/)

## How It Works

### Reasoning Trace Extraction

gpt-oss-safeguard uses the [Harmony format](https://github.com/openai/harmony) which separates reasoning from output. We extract the model's chain-of-thought via Groq's `include_reasoning: true` parameter:

```python
response = await client.post(
    "https://api.groq.com/openai/v1/chat/completions",
    json={
        "model": "openai/gpt-oss-safeguard-20b",
        "messages": [...],
        "include_reasoning": True  # Key parameter
    }
)

# Response structure:
# message.content = "1"  (just the label)
# message.reasoning = "We need to classify... this is harassment... so label 1"
```

### Embedding Format

Comments + reasoning are combined before embedding:

```
Original comment text here...

Reasoning: We need to classify content. The content is a harassing statement...
This is a form of harassment. According to policy... So classification: unsafe (1).
```

## License

MIT
