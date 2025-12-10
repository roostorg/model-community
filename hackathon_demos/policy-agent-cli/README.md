# Policy Agent CLI - Agentic Policy Optimization for GPT-OSS-Safeguard

An agentic tool for iteratively improving content moderation policies through automated evaluation and LLM-powered refinement. Built for the Open Safeguard Hackathon hosted by ROOST and OpenAI.

## Overview

Policy Agent CLI provides an agentic, data-driven approach to optimize content moderation policies for the [GPT-OSS-Safeguard](https://huggingface.co/openai/gpt-oss-safeguard-20b) model. Instead of manually tweaking policy definitions, this tool automates the process:

1. **Evaluate** - Run the current policy against a labeled dataset
2. **Analyze** - Calculate performance metrics (accuracy, precision, recall, F1)
3. **Improve** - Use an LLM to suggest policy refinements based on failures
4. **Iterate** - Track policy versions and continuously improve

## Problem Statement

Content moderation policies are typically written manually and improved through trial and error. This process is:
- **Time-consuming**: Manual review of failures and policy updates
- **Subjective**: Hard to know which changes will improve performance
- **Untracked**: Difficult to compare policy versions systematically

Policy Agent CLI solves this by automating the evaluation → analysis → improvement cycle.

## How It Works

### 1. Interactive CLI
```bash
poetry install
poetry run policy chat
```

A conversational interface for working with policies, featuring:
- Custom tools for running and modifying policies
- Natural language interaction with the policy optimization agent

## Setup

### Prerequisites
- [Poetry](https://python-poetry.org/) for dependency management
- API keys for LLM providers (GROQ_API_KEY)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/roostorg/model-community.git
cd hackathon_demos/policy-agent-cli
```

2. Install dependencies:
```bash
poetry install
```

3. Set up environment variables:
```bash
cp .env.sample .env
# Edit .env and add your API keys
```

4. Download the evaluation dataset from Hugging Face "mmathys/openai-moderation-api-evaluation":
```bash
python a02_download_hf_dataset.py
```

## Manual Usage (instead of CLI)

### Basic Workflow

### 1. Policy Evaluation
```bash
python a03_run_policy.py
```

Evaluates the latest policy version against the GPT-OSS-Safeguard dataset across 8 moderation categories:
- **S** - Sexual content
- **H** - Hate speech
- **V** - Violence
- **HR** - Harassment
- **SH** - Self-harm
- **S3** - Sexual content involving minors
- **H2** - Hate speech with threats
- **V2** - Graphic violence

Outputs:
- `policy_pak/runs/run_policy_v###_YYYYMMDD_HHMMSS_results.json` - Detailed predictions
- `policy_pak/runs/run_policy_v###_YYYYMMDD_HHMMSS_metrics.json` - Performance metrics

### 2. Policy Modification
```bash
python a04_modify_policy.py
```

Analyzes evaluation results and uses an LLM (GPT-OSS-120B) to:
- Identify patterns in false positives and false negatives
- Suggest specific improvements to policy definitions
- Generate a new policy version with refined criteria

The new policy is saved as `policy_pak/policy/policy_v###.md` with an incremented version number.


### 3. **Repeat**: The new policy version is now ready for evaluation

### Configuration

- `NUM_ROWS_TO_EVAL` in `.env`: Limit dataset size for faster iteration (default: 30)
- Policy files: `policy_pak/policy/policy_v###.md`
- Evaluation results: `policy_pak/runs/`

## Project Structure

```
openai-gptoss-hackathon/
├── policy_cli/              # CLI application code
│   ├── core/               # Core chat/LLM clients
│   ├── tools/              # Custom tools (run_policy, modify_policy)
│   ├── agents/             # Agent execution framework
│   └── main.py            # CLI entry point
├── policy_pak/
│   ├── policy/            # Policy versions (policy_v001.md, policy_v002.md, ...)
│   ├── data/              # HuggingFace dataset (downloaded)
│   └── runs/              # Evaluation results and metrics
├── a01_groq.py            # Test Groq API connection
├── a02_download_hf_dataset.py  # Download evaluation dataset
├── a03_run_policy.py      # Evaluate current policy
└── a04_modify_policy.py   # Generate improved policy
```

## Results

Each evaluation run produces detailed metrics including:

```json
{
  "S": {
    "accuracy": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "tp": 4, "fp": 0, "tn": 10, "fn": 0
  },
  "V": {
    "accuracy": 0.93,
    "precision": 0.5,
    "recall": 0.5,
    "f1": 0.5,
    "tp": 1, "fp": 1, "tn": 25, "fn": 1
  }
  // ... other categories
}
```

The agentic system tracks improvements across policy versions, enabling data-driven optimization.

## Technical Details

- **Models Used**:
  - GPT-OSS-Safeguard-20B (via Groq) for policy evaluation
  - GPT-OSS-120B (via Groq) for policy modification
  - GPT-5-mini (via OpenAI) for CLI interface

- **Dataset**: [OpenAI Moderation API evaluation dataset](https://huggingface.co/datasets/mmathys/openai-moderation-api-evaluation) from HuggingFace

- **Policy Format**: Markdown files with structured definitions and criteria for each moderation category
