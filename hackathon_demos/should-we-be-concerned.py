import os
import json
from collections import defaultdict
from itertools import islice

import gradio as gr
from groq import Groq
from datasets import load_dataset

# -----------------------------
# 1. Groq client + model config
# -----------------------------

MODEL_ID = "openai/gpt-oss-safeguard-20b"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# -----------------------------
# 2. Load a SMALL sample of Exorde
# -----------------------------

SAMPLE_SIZE = 20000  # adjust up/down if needed

print(f"Loading a sample of {SAMPLE_SIZE} rows from Exorde...")
streaming_ds = load_dataset(
    "Exorde/exorde-social-media-one-month-2024",
    split="train",
    streaming=True,
)

sample_rows = list(islice(streaming_ds, SAMPLE_SIZE))

posts_by_author: dict[str, list[str]] = defaultdict(list)
for row in sample_rows:
    author = row.get("author_hash")
    text = row.get("original_text")
    if author and text:
        posts_by_author[author].append(text)

# Pick some example users (those with the most posts in our sample)
sorted_authors = sorted(
    posts_by_author.keys(),
    key=lambda a: len(posts_by_author[a]),
    reverse=True,
)
EXAMPLE_USERS = sorted_authors[:20]

print(f"Indexed {len(posts_by_author)} unique authors from the sample.")
print("Example author_hashes you can try:", EXAMPLE_USERS)

# -----------------------------
# 3. Default safety policy (editable in UI)
# -----------------------------

DEFAULT_POLICY = """
You are a safety classifier for social media users based on their public posts.

Inputs:
- You receive multiple posts written by a single (anonymized) author from a social-media dataset.
- Posts may include different topics, languages, and sentiments.

Goal:
- Decide whether this author's overall posting behavior in the provided sample is mostly SAFE ("good")
  or PROBLEMATIC/TOXIC ("bad") according to the toxicity criteria below.
- Also provide a short explanation.

Toxic / "bad" behavior includes (non-exhaustive):
- Insults and personal attacks
- Harassment, bullying, or demeaning language toward others
- Threats or celebration of violence
- Highly aggressive, combative, or dehumanizing language
- Very frequent toxic messages even if some are neutral

Non-toxic / "good" behavior:
- Mostly neutral, informative, or positive posts
- Occasional mild negativity that is not targeted or abusive
- Legitimate criticism or strong opinions expressed without personal attacks

Output format:
Return ONLY valid JSON with this exact shape:
{
  "label": "good" | "bad",
  "reasoning": "<short explanation of why, based ONLY on the posts you saw>"
}

Do not include any extra text outside the JSON.
If the evidence is mixed, choose the label that best matches the *overall* tendency.
"""

# -----------------------------
# 4. Core classifier (text + policy -> label + reasoning)
# -----------------------------

def classify_text(text: str, policy: str):
    """
    Call Groq with the safeguard model, asking for JSON:
    { "label": "good"|"bad", "reasoning": "..." }
    """
    # Fallback to default policy if somehow empty
    policy_to_use = policy.strip() or DEFAULT_POLICY

    completion = client.chat.completions.create(
        model=MODEL_ID,
        temperature=0,
        max_tokens=512,
        messages=[
            {"role": "system", "content": policy_to_use},
            {"role": "user", "content": text},
        ],
    )

    raw_content = completion.choices[0].message.content

    # Try to parse the JSON we asked for
    try:
        result = json.loads(raw_content)
        label = str(result.get("label", "unknown")).lower()
        reasoning = str(result.get("reasoning", ""))
    except Exception:
        # Fallback if the model didn't obey perfectly
        reasoning = raw_content
        text_lower = raw_content.lower()
        if "bad" in text_lower and "good" not in text_lower:
            label = "bad"
        elif "good" in text_lower and "bad" not in text_lower:
            label = "good"
        else:
            label = "unknown"

    # Map to scores for gr.Label
    scores = {"good": 0.0, "bad": 0.0, "unknown": 0.0}
    if label in scores:
        scores[label] = 1.0

    return scores, reasoning

# -----------------------------
# 5. Wrapper: author_hash -> posts -> classification
# -----------------------------

def classify_author(author_hash_manual: str, author_hash_dropdown: str, policy_text: str):
    """
    - If the user typed an author_hash, use that.
    - Otherwise, fall back to the dropdown selection.
    - We look up all posts for that author in our sampled subset,
      concatenate them, and send them to the classifier.
    - Also return the posts so the UI can display them.
    """
    author_hash = (author_hash_manual or "").strip() or (author_hash_dropdown or "").strip()

    if not author_hash:
        msg = "Please paste an author_hash or select one from the examples."
        return (
            {"unknown": 1.0, "good": 0.0, "bad": 0.0},
            msg,
            "",
        )

    posts = posts_by_author.get(author_hash)
    if not posts:
        msg = (
            f"No posts found for author_hash `{author_hash}` in our sample of {SAMPLE_SIZE} rows.\n\n"
            "Make sure you:\n"
            "- Copy the `author_hash` exactly from the dataset (40-char hex string), and\n"
            "- Understand this demo only uses a small sample, not all 269M rows."
        )
        return (
            {"unknown": 1.0, "good": 0.0, "bad": 0.0},
            msg,
            "",
        )

    # Join up to N posts from this author into one big text block for the model
    N_POSTS = 30
    selected_posts = posts[:N_POSTS]
    joined_text = "\n\n".join(selected_posts)

    scores, reasoning = classify_text(joined_text, policy_text)

    # Build a human-readable preview of the posts used
    posts_preview_lines = []
    for i, p in enumerate(selected_posts, start=1):
        posts_preview_lines.append(f"Post {i}:\n{p}")
    posts_preview = "\n\n" + ("\n\n".join(posts_preview_lines))

    return scores, reasoning, posts_preview

# -----------------------------
# 6. Gradio UI
# -----------------------------

with gr.Blocks(title="Should We Be Concerned?") as gradio_app:
    gr.Markdown(
        """
        # Should We Be Concerned? (Demo on Exorde Dataset)

        Paste an `author_hash` from the **Exorde/exorde-social-media-one-month-2024** dataset
        or pick one of the example users below.  
        We'll fetch that author's posts from a small sampled subset and run a safety classification
        (mostly **safe** vs mostly **toxic**).

        > Note: `author_hash` is an anonymized ID — this is about content behavior, not real identities.
        """
    )

    with gr.Row():
        author_manual = gr.Textbox(
            label="Author hash from dataset (author_hash column)",
            placeholder="e.g. 705d7472cc9708026676a0e13116bb9fd4c1b0c3",
        )
        author_dropdown = gr.Dropdown(
            choices=EXAMPLE_USERS,
            label="Or pick an example author_hash from our sample",
            interactive=True,
        )

    # Editable policy text
    policy_input = gr.Textbox(
        label="Classification policy (system prompt)",
        value=DEFAULT_POLICY,
        lines=20,
    )

    classify_button = gr.Button("Classify this user")

    label_out = gr.Label(label="Classification (good / bad)", num_top_classes=2)
    reasoning_out = gr.Textbox(label="Model reasoning", lines=8)
    posts_out = gr.Textbox(
        label="Posts used for classification (sample from this author)",
        lines=15,
    )

    classify_button.click(
        fn=classify_author,
        inputs=[author_manual, author_dropdown, policy_input],
        outputs=[label_out, reasoning_out, posts_out],
    )

if __name__ == "__main__":
    gradio_app.launch()
