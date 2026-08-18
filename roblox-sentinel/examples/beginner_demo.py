"""
Beginner demo for Sentinel.

Goal: see the whole Sentinel flow end-to-end in ~1 screen of code, with no
dataset downloads and no S3. We build a tiny index in memory, then score a
batch of messages.

Run it with the project's venv active:
    cd ~/workspace/Sentinel
    source .venv/bin/activate
    python examples/beginner_demo.py
"""

import torch

from sentinel.sentinel_local_index import SentinelLocalIndex
from sentinel.embeddings.sbert import get_sentence_transformer_and_scaling_fn


# 1) Load the "text -> numbers" model (downloads once, then cached locally).
model_name = "all-MiniLM-L6-v2"
model, scale_fn = get_sentence_transformer_and_scaling_fn(model_name)

# 2) Define our two piles of examples.
#    Positive = the rare class we want to catch (here: aggressive/threatening).
#    Negative = normal, everyday chat.
positive_examples = [
    "I'm going to hurt you if you don't listen",
    "You should be scared of what I'll do",
    "I know where you live and I'm coming for you",
    "Give me your password or you'll regret it",
]
negative_examples = [
    "Want to play a game later?",
    "That movie was really fun, we should watch it again",
    "Happy birthday! Hope you have a great day",
    "Can you help me with my math homework?",
    "The weather is nice today, let's go outside",
]

# 3) Turn the example text into embeddings (lists of numbers capturing meaning).
positive_embeddings = torch.tensor(
    model.encode(positive_examples, normalize_embeddings=True)
)
negative_embeddings = torch.tensor(
    model.encode(negative_examples, normalize_embeddings=True)
)

# 4) Build the index (this is the object that does the scoring).
index = SentinelLocalIndex(
    sentence_model=model,
    positive_embeddings=positive_embeddings,
    negative_embeddings=negative_embeddings,
    scale_fn=scale_fn,
    positive_corpus=positive_examples,  # keep the text so explanations are readable
    negative_corpus=negative_examples,
)

# 5) Score a batch of NEW messages (imagine: one user's recent chat).
new_messages = [
    "Hey how are you?",
    "Do you want to grab lunch?",
    "I'm going to make you pay for this",
    "Cool, see you at 3pm",
    "You better watch your back",
]

result = index.calculate_rare_class_affinity(new_messages)

# 6) Read the results.
print(f"\nOverall rare-class affinity score: {result.rare_class_affinity_score:.4f}")
print(f"Aggregator used: {result.aggregation_name}\n")

print("Per-message scores (higher = more suspicious):")
for message, score in result.observation_scores.items():
    risk = "HIGH" if score > 0.5 else "MED" if score > 0.1 else "low"
    print(f"  [{risk:>4}] {score:.4f}  {message}")
