# Shieldstral

Resources related to Mistral AI's Shieldstral, a policy-adaptive, multimodal safety classifier.

- [Download Shieldstral 1.0 3B on HuggingFace](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
- [Mistral Announcement](https://mistral.ai/news/shieldstral/)
- [ROOST Announcement](https://roost.tools/blog/mistral-shieldstral-joins-the-RMC/)
- [Model Card](https://huggingface.co/mistralai/Shieldstral-1.0-3B)
- [Technical report](https://arxiv.org/abs/2607.25857)

Shieldstral is a 3B-parameter open-weight (Apache 2.0) classifier that flags harmful inputs and outputs across text and images. Moderation is cast as a single yes/no question: you supply a natural-language policy at inference time and the model returns a safety score, so the same weights serve any operator-defined policy without retraining.

## How it works

Each call composes a structured prompt with three tagged fields:

- `<Instruct>` — the evaluation context and how strict to be. Usually constant across a product surface.
- `<Query>` — one yes/no question, e.g. `Does this content promote physical violence?`
- `<Document>` — the content under review: a prompt, a response, a prompt+response pair, or an image with optional text.

The model unembeds the yes/no token logits; the safety score is the softmax-normalized probability of "yes," thresholded at τ = 0.5 for a binary decision. One query enforces one policy; for several policies, issue one query per policy rather than combining them.

See [`quickstart.ipynb`](./quickstart.ipynb) for a runnable example.

## Contents

- [`quickstart.ipynb`](./quickstart.ipynb) — minimal load-and-run example (text and image).
- [`evaluations/`](./evaluations) — community evaluation outcomes. See the note on evaluating steerable models below.

## A note on evaluating policy-adaptive models

Shieldstral is policy-adaptive: it takes an operator-defined policy at inference rather than enforcing a fixed taxonomy. Evaluating this kind of model well takes two distinct measurements:

1. **Safety performance** — does it correctly flag violative content and leave benign content alone?
2. **Steerability** — does it follow the *specific* policy you supply, rather than falling back on the priors it learned in training?

These can diverge, and community evaluations to date report different results depending on which is being measured and how policies are phrased. Contributions of evaluation outcomes on both dimensions are especially welcome; add them under [`evaluations/`](./evaluations).
