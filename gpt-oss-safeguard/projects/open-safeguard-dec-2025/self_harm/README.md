# Self-Harm Risk Classification and Assessment 

Build and evaluate a lightweight suicide-ideation classifier or labeler using OpenAI's gpt-oss chat models hosted on Groq. The goal is to prototype an inference flow that could slot into real moderation or triage pipelines (e.g., queue prioritization, human-in-the-loop review) while documenting performance and edge-case behavior. This repo contains a single runnable script plus a small red-team set to probe safety alignment.

## Team Members 
Ratnakar Pawar, Yang Liu

## Problem Statement
We want to quickly flag content that expresses suicidal ideation, self-harm, or suicidal intent so human moderators or clinicians can intervene. The system should minimize false negatives (missed high-risk messages) while keeping false positives low enough that reviewers are not overwhelmed. This project offers a transparent, reproducible baseline that teams can extend with prompt tuning, escalation rules, and monitoring for production use.

## Dataset
- Source: Kaggle “Suicide Watch” dataset (https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch?resource=download).
- Composition: Posts from r/SuicideWatch (labeled `suicide`) and r/depression (labeled `suicide` in this binary framing), plus non-suicide posts sampled from r/teenagers as negative examples.
- Collection window: r/SuicideWatch from Dec 16, 2008 (creation) to Jan 2, 2021; r/depression from Jan 1, 2009 to Jan 2, 2021.
- Motivation: Public suicide-ideation datasets are scarce; this set provides a starting point for safety classifiers while acknowledging ethical handling and human-in-the-loop review are essential.

## Prototype / Method
- Dataset: `Suicide_Detection.csv` (columns: `text`, `class`).
- Prompt: Harmony-style safety policy as the system message; user text is evaluated with a strict 0/1 response.
- Models tested: `openai/gpt-oss-20b` and `openai/gpt-oss-safeguard-20b`.
- Script: `oai-hackathon-self-harm.py` loads the dataset, calls the Groq endpoint with configurable concurrency, and reports accuracy, classification report, and confusion matrix.

## How to Run
1) Export your key: `export GROQ_API_KEY=sk-...`
2) From this folder:
   - Base model: `python oai-hackathon-self-harm.py --examples 1000 --concurrency 4`
   - Safeguard model: `python oai-hackathon-self-harm.py --examples 1000 --concurrency 4 --model openai/gpt-oss-safeguard-20b`
3) Flags: `--debug-first-n` prints raw outputs; `--dataset-path` points to a custom CSV.

## Results (1,000-sample eval)
- `openai/gpt-oss-20b`: accuracy ≈ 0.904; f1 (weighted) ≈ 0.90.
- `openai/gpt-oss-safeguard-20b`: accuracy ≈ 0.888; f1 (weighted) ≈ 0.89. The base model was marginally better in this test.
- Both models showed strong precision/recall balance; the base model’s slight edge suggests it can be used in real triage workflows, provided human review remains in the loop for edge cases.

## Red Teaming
- Contains crafted risky/benign phrases; run through the same prompt and mapping
- Model: `openai/gpt-oss-20b`.
- Eval: 40 examples (30 suicide, 10 non-suicide).
- Outcome:
  - Accuracy: 0.975
  - Suicide: precision 1.00, recall 0.97, f1 0.98
  - Non-suicide: precision 0.91, recall 1.00, f1 0.95
- The red-team set stresses borderline phrasing (humor, past self-harm in remission, passive ideation). High recall on suicide and perfect recall on non-suicide indicate strong policy adherence; remaining misses were subtle passive-ideation wording. These results support cautious real-world use with monitoring and periodic prompt/model checks.

## What We Learned / Takeaways
- Base model strength: `openai/gpt-oss-20b` delivered slightly better F1/accuracy than the safeguard variant while still following the safety prompt closely.
- Policy-following: Both models adhered well to the binary 0/1 instruction; remaining errors clustered around passive ideation or nuanced wording.
- Red-team value: Curated stress tests (humor, remission, passive ideation) surfaced edge failures that standard metrics miss; keep expanding this set for deployment.
- Human-in-the-loop: Metrics are strong enough for triage/queueing, but moderator oversight and escalation rules are essential for real-world use.
- Data dependence: Results are specific to the Kaggle “Suicide Watch” mix; monitor drift and retrain/retest if the content distribution shifts.
