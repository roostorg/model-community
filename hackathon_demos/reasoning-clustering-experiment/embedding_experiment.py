"""Embedding generation and clustering experiments on Civil Comments dataset."""
import json
import sys
import time
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings import embed_texts
from src.config import EMBEDDING_DIM

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Dataset paths
COMMENTS_DATASET = DATA_DIR / "civil_comments_balanced_toxic_0.9.json"
CLASSIFIED_DATASET = DATA_DIR / "civil_comments_classified_1000.json"

# Output paths
COMMENTS_ONLY_EMBEDDINGS = EMBEDDINGS_DIR / "comments_only_1000.npz"
COMMENTS_WITH_REASONING_EMBEDDINGS = EMBEDDINGS_DIR / "comments_with_reasoning_1000.npz"
COMMENTS_ONLY_SUBSET_EMBEDDINGS = EMBEDDINGS_DIR / "comments_only_subset_1000.npz"


def load_comments_dataset() -> dict:
    """Load the civil comments dataset (1000 samples, toxicity >= 0.9)."""
    with open(COMMENTS_DATASET) as f:
        return json.load(f)


def load_classified_dataset() -> list[dict]:
    """Load the classified dataset with reasoning (100 samples)."""
    with open(CLASSIFIED_DATASET) as f:
        return json.load(f)


def extract_reasoning_text(item: dict) -> str:
    """Extract reasoning trace from a classified item.

    Uses the model's actual chain-of-thought reasoning from the reasoning channel,
    not the structured output categories.
    """
    return item.get("reasoning_trace", "")


def generate_embeddings_batched(
    texts: list[str],
    batch_size: int = 50,
    delay: float = 0.5,
    progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings in batches with rate limiting.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call
        delay: Seconds between batches
        progress: Show progress

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM)
    """
    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]

        if progress:
            print(f"  Embedding batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
                  f"({i+1}-{min(i+len(batch), total)}/{total})...")

        embeddings = embed_texts(batch)
        all_embeddings.append(embeddings)

        # Rate limiting between batches
        if delay > 0 and i + batch_size < total:
            time.sleep(delay)

    return np.vstack(all_embeddings)


def save_embeddings(
    embeddings: np.ndarray,
    texts: list[str],
    labels: list[str],
    metadata: dict,
    path: Path
) -> None:
    """Save embeddings with metadata to .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        embeddings=embeddings,
        texts=np.array(texts, dtype=object),
        labels=np.array(labels, dtype=object),
        metadata=json.dumps(metadata)
    )
    print(f"Saved embeddings to: {path}")


def load_embeddings(path: Path) -> dict:
    """Load embeddings from .npz file."""
    data = np.load(path, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "texts": data["texts"].tolist(),
        "labels": data["labels"].tolist(),
        "metadata": json.loads(str(data["metadata"]))
    }


def generate_comments_only_embeddings(force: bool = False) -> dict:
    """
    Generate embeddings for all 1000 comments (comments text only).

    Args:
        force: If True, regenerate even if cache exists

    Returns:
        dict with embeddings, texts, labels, metadata
    """
    if COMMENTS_ONLY_EMBEDDINGS.exists() and not force:
        print(f"Loading cached embeddings from: {COMMENTS_ONLY_EMBEDDINGS}")
        return load_embeddings(COMMENTS_ONLY_EMBEDDINGS)

    print("Loading comments dataset...")
    dataset = load_comments_dataset()

    # Combine toxic and safe examples
    texts = []
    labels = []
    toxicity_scores = []

    for item in dataset["toxic"]:
        texts.append(item["text"])
        labels.append("toxic")
        toxicity_scores.append(item["toxicity"])

    for item in dataset["safe"]:
        texts.append(item["text"])
        labels.append("safe")
        toxicity_scores.append(item["toxicity"])

    print(f"Generating embeddings for {len(texts)} comments...")
    embeddings = generate_embeddings_batched(texts, batch_size=50, delay=0.3)

    metadata = {
        "source": str(COMMENTS_DATASET),
        "n_toxic": len(dataset["toxic"]),
        "n_safe": len(dataset["safe"]),
        "embedding_dim": EMBEDDING_DIM,
        "type": "comments_only"
    }

    save_embeddings(embeddings, texts, labels, metadata, COMMENTS_ONLY_EMBEDDINGS)

    return {
        "embeddings": embeddings,
        "texts": texts,
        "labels": labels,
        "metadata": metadata
    }


def generate_comments_with_reasoning_embeddings(force: bool = False) -> dict:
    """
    Generate embeddings for classified samples (comment + reasoning trace combined).

    Uses the model's actual chain-of-thought reasoning from the reasoning channel.

    Args:
        force: If True, regenerate even if cache exists

    Returns:
        dict with embeddings, texts, labels, metadata
    """
    if COMMENTS_WITH_REASONING_EMBEDDINGS.exists() and not force:
        print(f"Loading cached embeddings from: {COMMENTS_WITH_REASONING_EMBEDDINGS}")
        return load_embeddings(COMMENTS_WITH_REASONING_EMBEDDINGS)

    print("Loading classified dataset...")
    classified = load_classified_dataset()

    # Filter to items that have reasoning traces (no errors)
    valid_items = [
        item for item in classified
        if item.get("reasoning_trace") and not item.get("error")
    ]
    print(f"Found {len(valid_items)} items with reasoning traces (out of {len(classified)} total)")

    combined_texts = []
    labels = []
    original_texts = []

    for item in valid_items:
        original_text = item["text"]
        reasoning = extract_reasoning_text(item)

        # Combine comment and reasoning trace
        combined = f"{original_text}\n\nReasoning: {reasoning}"

        combined_texts.append(combined)
        original_texts.append(original_text)
        labels.append(item["ground_truth"])

    print(f"Generating embeddings for {len(combined_texts)} comment+reasoning pairs...")
    embeddings = generate_embeddings_batched(combined_texts, batch_size=50, delay=0.3)

    metadata = {
        "source": str(CLASSIFIED_DATASET),
        "n_samples": len(combined_texts),
        "n_toxic": sum(1 for l in labels if l == "toxic"),
        "n_safe": sum(1 for l in labels if l == "safe"),
        "embedding_dim": EMBEDDING_DIM,
        "type": "comments_with_reasoning_trace"
    }

    save_embeddings(embeddings, combined_texts, labels, metadata, COMMENTS_WITH_REASONING_EMBEDDINGS)

    # Also generate comment-only embeddings for the same samples (for fair comparison)
    print(f"\nGenerating comment-only embeddings for the same {len(original_texts)} samples...")
    comment_only_embeddings = generate_embeddings_batched(original_texts, batch_size=50, delay=0.3)

    comment_only_metadata = {
        "source": str(CLASSIFIED_DATASET),
        "n_samples": len(original_texts),
        "n_toxic": sum(1 for l in labels if l == "toxic"),
        "n_safe": sum(1 for l in labels if l == "safe"),
        "embedding_dim": EMBEDDING_DIM,
        "type": "comments_only_subset"
    }

    save_embeddings(comment_only_embeddings, original_texts, labels, comment_only_metadata, COMMENTS_ONLY_SUBSET_EMBEDDINGS)

    return {
        "embeddings": embeddings,
        "texts": combined_texts,
        "labels": labels,
        "metadata": metadata
    }


def main():
    """Run embedding generation for all datasets."""
    print("=" * 60)
    print("EMBEDDING EXPERIMENT - Generating Embeddings")
    print("=" * 60)

    # Step 1: Generate comments-only embeddings (1000 samples)
    print("\n[1/2] Comments Only (1000 samples)")
    print("-" * 40)
    comments_data = generate_comments_only_embeddings()
    print(f"  Shape: {comments_data['embeddings'].shape}")
    print(f"  Toxic: {sum(1 for l in comments_data['labels'] if l == 'toxic')}")
    print(f"  Safe: {sum(1 for l in comments_data['labels'] if l == 'safe')}")

    # Step 2: Generate comments+reasoning embeddings (with reasoning traces)
    print("\n[2/2] Comments + Reasoning Traces")
    print("-" * 40)
    reasoning_data = generate_comments_with_reasoning_embeddings()
    print(f"  Shape: {reasoning_data['embeddings'].shape}")
    print(f"  Toxic: {sum(1 for l in reasoning_data['labels'] if l == 'toxic')}")
    print(f"  Safe: {sum(1 for l in reasoning_data['labels'] if l == 'safe')}")

    print("\n" + "=" * 60)
    print("DONE! Embeddings saved to:")
    print(f"  - {COMMENTS_ONLY_EMBEDDINGS}")
    print(f"  - {COMMENTS_WITH_REASONING_EMBEDDINGS}")
    print(f"  - {COMMENTS_ONLY_SUBSET_EMBEDDINGS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
