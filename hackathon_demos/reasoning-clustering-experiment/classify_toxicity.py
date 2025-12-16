"""Classify civil comments using gpt-oss-safeguard via Groq API.

Extracts the model's reasoning trace (chain-of-thought) separately from the classification.
Uses async concurrency for faster processing of large datasets.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path
import httpx

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import GROQ_API_KEY, SAFEGUARD_MODEL

# Paths
PROJECT_ROOT = Path(__file__).parent
POLICY_PATH = PROJECT_ROOT / "toxicity_policy.md"
DATA_PATH = PROJECT_ROOT / "data" / "civil_comments_balanced_toxic_0.9.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "civil_comments_classified_1000.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "civil_comments_classified_1000.json"

# Concurrency settings
MAX_CONCURRENT_REQUESTS = 5  # Reduced for Groq rate limits
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Response format instruction to append to policy
# Simplified format since reasoning is extracted from the separate reasoning channel
RESPONSE_FORMAT = """
## Response Format

Given a content item, respond with ONLY a single digit:
- 1 for a policy violation
- 0 for no violation

Do not include any other text, explanation, or formatting in your response.
"""


def load_policy() -> str:
    """Load toxicity policy from markdown file."""
    with open(POLICY_PATH) as f:
        return f.read()


def load_dataset() -> dict:
    """Load civil comments dataset."""
    with open(DATA_PATH) as f:
        return json.load(f)


def build_system_prompt(policy: str) -> str:
    """Build system prompt with policy and response format."""
    return f"""{policy}

{RESPONSE_FORMAT}
"""


async def classify_comment_async(
    client: httpx.AsyncClient,
    text: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore
) -> dict:
    """
    Classify a single comment using gpt-oss-safeguard (async).

    Extracts the model's reasoning trace from the separate reasoning channel.
    Includes retry logic with exponential backoff for rate limits.

    Returns:
        dict with 'label', 'reasoning_trace', 'content', and metadata
    """
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    user_message = f"""Content: {text}

Classify as 1 (violation) or 0 (safe):"""

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": SAFEGUARD_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1000,
                        "include_reasoning": True  # Extract reasoning trace
                    },
                    timeout=REQUEST_TIMEOUT
                )

                # Handle rate limiting with retry
                if response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    return {"error": "Rate limited after retries", "label": None, "reasoning_trace": ""}

                if response.status_code != 200:
                    return {"error": f"API error: {response.status_code}", "body": response.text}

                data = response.json()
                message = data["choices"][0]["message"]

                # Extract reasoning trace (the model's chain-of-thought)
                reasoning_trace = message.get("reasoning", "")
                content = message.get("content", "")

                # Parse label from content
                label = None
                content_stripped = content.strip()
                if content_stripped in ("0", "1"):
                    label = int(content_stripped)
                elif content_stripped.startswith("0"):
                    label = 0
                elif content_stripped.startswith("1"):
                    label = 1

                # Extract token usage for reasoning
                usage = data.get("usage", {})
                reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)

                return {
                    "label": label,
                    "reasoning_trace": reasoning_trace,
                    "content": content,
                    "reasoning_tokens": reasoning_tokens,
                    "error": None
                }

            except httpx.TimeoutException:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY)
                    continue
                return {"error": "Request timeout", "label": None, "reasoning_trace": ""}
            except Exception as e:
                return {"error": str(e), "label": None, "reasoning_trace": ""}

        return {"error": "Max retries exceeded", "label": None, "reasoning_trace": ""}


def extract_reasoning_only(categories: list[dict]) -> str:
    """Extract just the reasoning text (no category or policy source)."""
    if not categories:
        return ""
    parts = []
    for cat in categories:
        reasoning = cat.get("reasoning", "")
        if reasoning:
            parts.append(reasoning)
    return " ".join(parts)


def extract_category(categories: list[dict]) -> str:
    """Extract the primary category."""
    if not categories:
        return ""
    return categories[0].get("category", "")


async def classify_dataset_async(
    dataset: dict,
    system_prompt: str,
    max_samples: int | None = None,
    max_concurrent: int = MAX_CONCURRENT_REQUESTS
) -> list[dict]:
    """
    Classify all comments in the dataset using async concurrency.

    Args:
        dataset: dict with 'toxic' and 'safe' lists
        system_prompt: system prompt with policy
        max_samples: limit per category (None for all)
        max_concurrent: maximum concurrent API requests

    Returns:
        list of classification results with reasoning traces
    """
    # Combine toxic and safe examples
    all_examples = []
    for ex in dataset["toxic"][:max_samples]:
        all_examples.append({**ex, "ground_truth": "toxic"})
    for ex in dataset["safe"][:max_samples]:
        all_examples.append({**ex, "ground_truth": "safe"})

    total = len(all_examples)
    print(f"Classifying {total} examples with {max_concurrent} concurrent requests...")

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    # Progress tracking
    completed = 0
    results = [None] * total

    async def classify_with_progress(idx: int, example: dict, client: httpx.AsyncClient):
        nonlocal completed
        text = example["text"]
        ground_truth = example["ground_truth"]

        classification = await classify_comment_async(client, text, system_prompt, semaphore)

        result = {
            "id": idx,
            "text": text,
            "ground_truth": ground_truth,
            "ground_truth_toxicity": example["toxicity"],
            "predicted_label": classification.get("label"),
            "reasoning_trace": classification.get("reasoning_trace", ""),
            "reasoning_tokens": classification.get("reasoning_tokens", 0),
            "content": classification.get("content", ""),
            "error": classification.get("error")
        }
        results[idx] = result

        completed += 1
        # Print progress every 10 items or at completion
        if completed % 10 == 0 or completed == total:
            print(f"  [{completed}/{total}] completed", flush=True)

        return result

    # Run all classifications concurrently
    async with httpx.AsyncClient() as client:
        tasks = [
            classify_with_progress(i, ex, client)
            for i, ex in enumerate(all_examples)
        ]
        await asyncio.gather(*tasks)

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics."""
    # Filter out errors
    valid = [r for r in results if r["predicted_label"] is not None]

    if not valid:
        return {"error": "No valid predictions"}

    # Convert ground truth to binary
    tp = fp = tn = fn = 0
    for r in valid:
        gt_toxic = r["ground_truth"] == "toxic"
        pred_toxic = r["predicted_label"] == 1

        if gt_toxic and pred_toxic:
            tp += 1
        elif not gt_toxic and pred_toxic:
            fp += 1
        elif not gt_toxic and not pred_toxic:
            tn += 1
        else:
            fn += 1

    total = len(valid)
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total,
        "errors": len(results) - len(valid),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
    }


def extract_reasoning(categories: list[dict]) -> str:
    """Extract combined reasoning from categories list (full format with category and policy)."""
    if not categories:
        return ""
    parts = []
    for cat in categories:
        category = cat.get("category", "Unknown")
        reasoning = cat.get("reasoning", "")
        policy_source = cat.get("policy_source", "")
        parts.append(f"[{category}] {reasoning} (Policy: {policy_source})")
    return " | ".join(parts)


def save_results_csv(results: list[dict], path: Path) -> None:
    """Save results to CSV with reasoning trace column."""
    fieldnames = [
        "id",
        "text",
        "ground_truth",
        "ground_truth_toxicity",
        "predicted_label",
        "reasoning_trace",
        "reasoning_tokens",
        "content",
        "error"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                "id": r["id"],
                "text": r["text"],
                "ground_truth": r["ground_truth"],
                "ground_truth_toxicity": r["ground_truth_toxicity"],
                "predicted_label": r["predicted_label"],
                "reasoning_trace": r.get("reasoning_trace", ""),
                "reasoning_tokens": r.get("reasoning_tokens", 0),
                "content": r.get("content", ""),
                "error": r.get("error", "")
            }
            writer.writerow(row)


async def main_async(max_samples: int | None = None):
    """
    Main async entry point.

    Args:
        max_samples: Limit per category (None for all 500 toxic + 500 safe = 1000)
    """
    # Load policy and dataset
    print("Loading policy...")
    policy = load_policy()
    system_prompt = build_system_prompt(policy)

    print("Loading dataset...")
    dataset = load_dataset()
    print(f"  {len(dataset['toxic'])} toxic, {len(dataset['safe'])} safe examples")

    # Classify with async concurrency
    print("\n" + "="*60)
    print("CLASSIFYING WITH REASONING TRACE EXTRACTION")
    print("="*60)
    results = await classify_dataset_async(
        dataset,
        system_prompt,
        max_samples=max_samples  # None = all samples
    )

    # Save results
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    save_results_csv(results, OUTPUT_CSV_PATH)
    print(f"\nSaved CSV to: {OUTPUT_CSV_PATH}")

    # Save as JSON (for full data)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON to: {OUTPUT_JSON_PATH}")

    # Compute metrics
    print("\n" + "="*60)
    print("METRICS:")
    print("="*60)
    metrics = compute_metrics(results)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Show reasoning trace stats
    traces_with_content = sum(1 for r in results if r.get("reasoning_trace"))
    avg_tokens = sum(r.get("reasoning_tokens", 0) for r in results) / len(results) if results else 0
    print(f"\nReasoning traces extracted: {traces_with_content}/{len(results)}")
    print(f"Average reasoning tokens: {avg_tokens:.1f}")

    return results


def main():
    """Synchronous entry point."""
    asyncio.run(main_async(max_samples=None))


if __name__ == "__main__":
    main()
