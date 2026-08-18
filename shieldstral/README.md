# Shieldstral

Resources and projects related to Mistral AI's Shieldstral, a policy-adaptive, multimodal safety classifier.

Shieldstral is a 3B-parameter open-weight (Apache 2.0) classifier that flags harmful inputs and outputs across text and images. Moderation is cast as a single yes/no question: you supply a natural-language policy at inference time and the model returns a safety score, so the same weights serve any operator-defined policy without retraining.

## At a glance

| | |
| --- | --- |
| **Partner** | Mistral AI |
| **Model** | [Shieldstral-1.0](https://huggingface.co/mistralai/Shieldstral-1.0) |
| **License** | Apache 2.0 |
| **Parameters** | 3B |
| **Modality** | Text and image |
| **Output** | Safety score (probability of "yes"), thresholded for a binary decision |
| **Policy interface** | Natural-language query at inference (`<Instruct>`, `<Query>`, `<Document>`) |
| **Model card** | https://docs.mistral.ai/models/model-cards/shieldstral-1-0 |
| **Paper** | https://arxiv.org/abs/2607.25857 |

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

## Links

- Announcement blog: TODO (roost.tools link once published)
- Mistral model card: https://docs.mistral.ai/models/model-cards/shieldstral-1-0
- Technical report: https://arxiv.org/abs/2607.25857
- Weights: https://huggingface.co/mistralai/Shieldstral-1.0
