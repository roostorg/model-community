#!/usr/bin/env python3
"""
Prepare Civil Comments dataset for clustering experiments.

Downloads the Civil Comments dataset and creates a balanced subset
of toxic (toxicity >= 0.9) and safe (toxicity < 0.1) examples.

Usage:
    uv run python prepare_dataset.py --samples 1000

Requirements:
    pip install datasets
"""
import argparse
import json
import sys
from pathlib import Path

# Try importing datasets, provide helpful error if missing
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed")
    print("Install with: uv add datasets")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"


def prepare_civil_comments(
    n_samples: int = 1000,
    toxic_threshold: float = 0.9,
    safe_threshold: float = 0.1,
    output_name: str | None = None
) -> Path:
    """
    Download and prepare Civil Comments dataset.

    Args:
        n_samples: Total samples (split evenly between toxic/safe)
        toxic_threshold: Minimum toxicity score for "toxic" class
        safe_threshold: Maximum toxicity score for "safe" class
        output_name: Custom output filename (default: auto-generated)

    Returns:
        Path to the created dataset file
    """
    samples_per_class = n_samples // 2

    print(f"Loading Civil Comments dataset from HuggingFace...")
    dataset = load_dataset("google/civil_comments", split="train")

    print(f"Dataset loaded: {len(dataset)} examples")

    # Filter toxic examples (toxicity >= threshold)
    print(f"Filtering toxic examples (toxicity >= {toxic_threshold})...")
    toxic_examples = [
        {"text": ex["text"], "toxicity": ex["toxicity"], "label": "toxic"}
        for ex in dataset
        if ex["toxicity"] >= toxic_threshold
    ]
    print(f"  Found {len(toxic_examples)} toxic examples")

    # Filter safe examples (toxicity < threshold)
    print(f"Filtering safe examples (toxicity < {safe_threshold})...")
    safe_examples = [
        {"text": ex["text"], "toxicity": ex["toxicity"], "label": "safe"}
        for ex in dataset
        if ex["toxicity"] < safe_threshold
    ]
    print(f"  Found {len(safe_examples)} safe examples")

    # Sample balanced subset
    import random
    random.seed(42)  # Reproducibility

    if len(toxic_examples) < samples_per_class:
        print(f"Warning: Only {len(toxic_examples)} toxic examples available")
        samples_per_class = min(len(toxic_examples), len(safe_examples))

    toxic_sample = random.sample(toxic_examples, min(samples_per_class, len(toxic_examples)))
    safe_sample = random.sample(safe_examples, min(samples_per_class, len(safe_examples)))

    # Create output
    output_data = {
        "toxic": toxic_sample,
        "safe": safe_sample,
        "metadata": {
            "source": "google/civil_comments",
            "toxic_threshold": toxic_threshold,
            "safe_threshold": safe_threshold,
            "n_toxic": len(toxic_sample),
            "n_safe": len(safe_sample)
        }
    }

    # Save
    if output_name is None:
        output_name = f"civil_comments_balanced_toxic_{toxic_threshold}.json"

    output_path = DATA_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"  Toxic: {len(toxic_sample)}")
    print(f"  Safe: {len(safe_sample)}")
    print(f"  Total: {len(toxic_sample) + len(safe_sample)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Civil Comments dataset for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python experiments/prepare_dataset.py
    uv run python experiments/prepare_dataset.py --samples 500
    uv run python experiments/prepare_dataset.py --toxic-threshold 0.8
        """
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="Total number of samples (split between toxic/safe, default: 1000)"
    )
    parser.add_argument(
        "--toxic-threshold",
        type=float,
        default=0.9,
        help="Minimum toxicity score for 'toxic' class (default: 0.9)"
    )
    parser.add_argument(
        "--safe-threshold",
        type=float,
        default=0.1,
        help="Maximum toxicity score for 'safe' class (default: 0.1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Custom output filename"
    )

    args = parser.parse_args()

    prepare_civil_comments(
        n_samples=args.samples,
        toxic_threshold=args.toxic_threshold,
        safe_threshold=args.safe_threshold,
        output_name=args.output
    )


if __name__ == "__main__":
    main()
