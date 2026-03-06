#!/usr/bin/env python3
"""
Reasoning Trace Clustering Analysis

Compares how toxic/safe content clusters when using:
1. Raw comment text embeddings
2. Comment + model reasoning trace embeddings

Usage:
    uv run python clustering_analysis.py --samples 100
    uv run python clustering_analysis.py --samples 1000 --clusters 3,5,7,10

Requirements:
    - GROQ_API_KEY environment variable (for gpt-oss-safeguard classification)
    - GEMINI_API_KEY environment variable (for embeddings)
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import GROQ_API_KEY, GEMINI_API_KEY

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SOURCE_DATASET = DATA_DIR / "civil_comments_balanced_toxic_0.9.json"


@dataclass
class ClusteringResult:
    """Results from a clustering run."""
    n_clusters: int
    silhouette: float
    purity: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    composition: dict


def get_cache_paths(n_samples: int) -> dict:
    """Get cache file paths for a given sample count."""
    return {
        "classified": DATA_DIR / f"classified_{n_samples}.json",
        "embeddings_comments": EMBEDDINGS_DIR / f"comments_only_{n_samples}.npz",
        "embeddings_reasoning": EMBEDDINGS_DIR / f"comments_with_reasoning_{n_samples}.npz",
    }


def check_cache(n_samples: int) -> dict:
    """Check which cached files exist."""
    paths = get_cache_paths(n_samples)
    return {key: path.exists() for key, path in paths.items()}


# =============================================================================
# STEP 1: Classification with Reasoning Traces
# =============================================================================

async def run_classification(n_samples: int, force: bool = False) -> list[dict]:
    """
    Classify comments using gpt-oss-safeguard and extract reasoning traces.

    Args:
        n_samples: Number of samples (split evenly between toxic/safe)
        force: If True, re-run even if cache exists

    Returns:
        List of classification results with reasoning traces
    """
    from classify_toxicity import (
        load_policy, load_dataset, build_system_prompt,
        classify_dataset_async, save_results_csv
    )

    paths = get_cache_paths(n_samples)
    cache_path = paths["classified"]

    # Check cache
    if cache_path.exists() and not force:
        print(f"  Loading cached classifications from {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)

    # Run classification
    print(f"  Classifying {n_samples} samples with gpt-oss-safeguard...")
    policy = load_policy()
    system_prompt = build_system_prompt(policy)
    dataset = load_dataset()

    samples_per_class = n_samples // 2
    results = await classify_dataset_async(
        dataset,
        system_prompt,
        max_samples=samples_per_class
    )

    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save CSV for inspection
    csv_path = cache_path.with_suffix(".csv")
    save_results_csv(results, csv_path)

    return results


# =============================================================================
# STEP 2: Embedding Generation
# =============================================================================

def run_embedding_generation(
    classified_data: list[dict],
    n_samples: int,
    force: bool = False
) -> tuple[dict, dict]:
    """
    Generate embeddings for comments-only and comments+reasoning.

    Returns:
        Tuple of (comments_only_data, comments_with_reasoning_data)
    """
    from embedding_experiment import (
        generate_embeddings_batched, save_embeddings, load_embeddings
    )
    from src.config import EMBEDDING_DIM

    paths = get_cache_paths(n_samples)

    # Filter to items with reasoning traces
    valid_items = [
        item for item in classified_data
        if item.get("reasoning_trace") and not item.get("error")
    ]

    print(f"  Found {len(valid_items)} samples with valid reasoning traces")

    # Prepare texts
    original_texts = [item["text"] for item in valid_items]
    combined_texts = [
        f"{item['text']}\n\nReasoning: {item['reasoning_trace']}"
        for item in valid_items
    ]
    labels = [item["ground_truth"] for item in valid_items]

    # Generate/load comments-only embeddings
    comments_path = paths["embeddings_comments"]
    if comments_path.exists() and not force:
        print(f"  Loading cached comment embeddings from {comments_path.name}")
        comments_data = load_embeddings(comments_path)
    else:
        print(f"  Generating comment-only embeddings...")
        embeddings = generate_embeddings_batched(original_texts, batch_size=50, delay=0.3)
        comments_data = {
            "embeddings": embeddings,
            "texts": original_texts,
            "labels": labels,
            "metadata": {"type": "comments_only", "n_samples": len(original_texts)}
        }
        comments_path.parent.mkdir(parents=True, exist_ok=True)
        save_embeddings(embeddings, original_texts, labels, comments_data["metadata"], comments_path)

    # Generate/load comments+reasoning embeddings
    reasoning_path = paths["embeddings_reasoning"]
    if reasoning_path.exists() and not force:
        print(f"  Loading cached reasoning embeddings from {reasoning_path.name}")
        reasoning_data = load_embeddings(reasoning_path)
    else:
        print(f"  Generating comment+reasoning embeddings...")
        embeddings = generate_embeddings_batched(combined_texts, batch_size=50, delay=0.3)
        reasoning_data = {
            "embeddings": embeddings,
            "texts": combined_texts,
            "labels": labels,
            "metadata": {"type": "comments_with_reasoning", "n_samples": len(combined_texts)}
        }
        save_embeddings(embeddings, combined_texts, labels, reasoning_data["metadata"], reasoning_path)

    return comments_data, reasoning_data


# =============================================================================
# STEP 3: Clustering Analysis
# =============================================================================

def compute_purity(cluster_ids: np.ndarray, labels: list[str]) -> float:
    """Compute cluster purity (= accuracy of cluster-based classifier)."""
    n_samples = len(labels)
    labels_arr = np.array(labels)

    total_correct = 0
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        cluster_labels = labels_arr[mask]
        toxic_count = (cluster_labels == "toxic").sum()
        safe_count = (cluster_labels == "safe").sum()
        total_correct += max(toxic_count, safe_count)

    return total_correct / n_samples


def compute_classification_metrics(
    cluster_ids: np.ndarray,
    labels: list[str]
) -> dict:
    """Compute precision/recall/F1 treating clusters as classifiers."""
    labels_arr = np.array(labels)

    # Assign each cluster to majority class
    cluster_to_class = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        cluster_labels = labels_arr[mask]
        toxic_count = (cluster_labels == "toxic").sum()
        safe_count = (cluster_labels == "safe").sum()
        cluster_to_class[cid] = "toxic" if toxic_count > safe_count else "safe"

    # Make predictions
    predictions = np.array([cluster_to_class[cid] for cid in cluster_ids])

    # Compute confusion matrix for 'toxic' class
    tp = ((predictions == "toxic") & (labels_arr == "toxic")).sum()
    fp = ((predictions == "toxic") & (labels_arr == "safe")).sum()
    fn = ((predictions == "safe") & (labels_arr == "toxic")).sum()
    tn = ((predictions == "safe") & (labels_arr == "safe")).sum()

    accuracy = (tp + tn) / len(labels_arr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
    }


def compute_composition(cluster_ids: np.ndarray, labels: list[str]) -> dict:
    """Compute per-cluster composition."""
    labels_arr = np.array(labels)
    composition = {}

    for cid in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == cid
        cluster_labels = labels_arr[mask]
        composition[int(cid)] = {
            "toxic": int((cluster_labels == "toxic").sum()),
            "safe": int((cluster_labels == "safe").sum()),
            "total": int(mask.sum())
        }

    return composition


def run_clustering_analysis(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int,
    random_state: int = 42
) -> ClusteringResult:
    """Run KMeans clustering and compute all metrics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_ids = kmeans.fit_predict(embeddings)

    silhouette = silhouette_score(embeddings, cluster_ids) if n_clusters > 1 else 0.0
    purity = compute_purity(cluster_ids, labels)
    metrics = compute_classification_metrics(cluster_ids, labels)
    composition = compute_composition(cluster_ids, labels)

    return ClusteringResult(
        n_clusters=n_clusters,
        silhouette=silhouette,
        purity=purity,
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        composition=composition
    )


# =============================================================================
# Main Pipeline
# =============================================================================

async def run_pipeline(
    n_samples: int,
    clusters: list[int],
    force: bool = False
) -> dict:
    """
    Run the full pipeline: classify -> embed -> cluster -> analyze.

    Args:
        n_samples: Total number of samples (split between toxic/safe)
        clusters: List of k values to test
        force: If True, regenerate all cached data

    Returns:
        Dict with all results
    """
    results = {
        "n_samples": n_samples,
        "clusters_tested": clusters,
        "comparisons": []
    }

    # Step 1: Classification
    print("\n[1/3] Classification with Reasoning Trace Extraction")
    print("-" * 50)
    classified = await run_classification(n_samples, force=force)

    valid_count = sum(1 for r in classified if r.get("reasoning_trace") and not r.get("error"))
    error_count = sum(1 for r in classified if r.get("error"))
    print(f"  Valid samples: {valid_count}, Errors: {error_count}")

    # Step 2: Embeddings
    print("\n[2/3] Embedding Generation")
    print("-" * 50)
    comments_data, reasoning_data = run_embedding_generation(classified, n_samples, force=force)

    actual_samples = len(comments_data["labels"])
    results["actual_samples"] = actual_samples
    print(f"  Embeddings generated for {actual_samples} samples")

    # Step 3: Clustering Analysis
    print("\n[3/3] Clustering Analysis")
    print("-" * 50)

    for k in clusters:
        comments_result = run_clustering_analysis(
            comments_data["embeddings"],
            comments_data["labels"],
            n_clusters=k
        )
        reasoning_result = run_clustering_analysis(
            reasoning_data["embeddings"],
            reasoning_data["labels"],
            n_clusters=k
        )

        comparison = {
            "k": k,
            "comments_only": {
                "silhouette": comments_result.silhouette,
                "purity": comments_result.purity,
                "accuracy": comments_result.accuracy,
                "precision": comments_result.precision,
                "recall": comments_result.recall,
                "f1": comments_result.f1
            },
            "comments_with_reasoning": {
                "silhouette": reasoning_result.silhouette,
                "purity": reasoning_result.purity,
                "accuracy": reasoning_result.accuracy,
                "precision": reasoning_result.precision,
                "recall": reasoning_result.recall,
                "f1": reasoning_result.f1
            }
        }
        results["comparisons"].append(comparison)

    return results


def print_results(results: dict):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("RESULTS: Reasoning Trace Clustering Analysis")
    print("=" * 70)
    print(f"Samples: {results['actual_samples']}")

    print("\n" + "-" * 70)
    print(f"{'k':>3} | {'Metric':<12} | {'Comments Only':>14} | {'+ Reasoning':>14} | {'Winner':<10}")
    print("-" * 70)

    for comp in results["comparisons"]:
        k = comp["k"]
        c = comp["comments_only"]
        r = comp["comments_with_reasoning"]

        metrics = [
            ("Silhouette", c["silhouette"], r["silhouette"]),
            ("Purity", c["purity"], r["purity"]),
            ("Precision", c["precision"], r["precision"]),
            ("Recall", c["recall"], r["recall"]),
            ("F1", c["f1"], r["f1"]),
        ]

        for i, (name, cv, rv) in enumerate(metrics):
            k_str = str(k) if i == 0 else ""
            winner = "Comments" if cv > rv + 0.001 else "+Reasoning" if rv > cv + 0.001 else "Tie"
            print(f"{k_str:>3} | {name:<12} | {cv:>14.3f} | {rv:>14.3f} | {winner:<10}")

        print("-" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze how reasoning traces affect content clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python clustering_analysis.py --samples 100
    uv run python clustering_analysis.py --samples 500 --clusters 3,5,10
    uv run python clustering_analysis.py --samples 100 --force

Environment Variables Required:
    GROQ_API_KEY    - For gpt-oss-safeguard classification
    GEMINI_API_KEY  - For text embeddings
        """
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Number of samples to analyze (default: 100)"
    )
    parser.add_argument(
        "--clusters", "-k",
        type=str,
        default="3,5,7",
        help="Comma-separated list of k values for clustering (default: 3,5,7)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force regeneration of cached data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Parse clusters
    clusters = [int(k.strip()) for k in args.clusters.split(",")]

    # Check source dataset exists
    if not SOURCE_DATASET.exists():
        print(f"Error: Source dataset not found: {SOURCE_DATASET}")
        print("\nTo create it, run:")
        print("  uv run python prepare_dataset.py --samples 1000")
        sys.exit(1)

    # Check API keys
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY environment variable not set")
        sys.exit(1)
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)

    print("=" * 70)
    print("Reasoning Trace Clustering Analysis")
    print("=" * 70)
    print(f"Samples: {args.samples}")
    print(f"Clusters: {clusters}")
    print(f"Force regenerate: {args.force}")

    # Check cache status
    cache_status = check_cache(args.samples)
    print(f"\nCache status:")
    for key, exists in cache_status.items():
        status = "found" if exists else "will generate"
        print(f"  {key}: {status}")

    # Run pipeline
    start_time = time.time()
    results = asyncio.run(run_pipeline(args.samples, clusters, force=args.force))
    elapsed = time.time() - start_time

    # Print results
    print_results(results)
    print(f"\nCompleted in {elapsed:.1f} seconds")

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
