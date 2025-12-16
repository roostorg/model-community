"""Run GPT-OSS-Safeguard policy evaluation on the dataset."""

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Set random seed for reproducibility
random.seed(42)

# Configuration
POLICY_DIR = "./policy_pak/policy"
DATA_DIR = "./policy_pak/data"
RUNS_DIR = "./policy_pak/runs"
MODEL = "openai/gpt-oss-safeguard-20b"

# Categories to evaluate
CATEGORIES = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def get_latest_policy():
    """Find and load the latest policy version."""
    policy_dir = Path(POLICY_DIR)

    if not policy_dir.exists():
        raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

    # Find all policy files matching pattern policy_v###.md
    policy_files = list(policy_dir.glob("policy_v*.md"))

    if not policy_files:
        raise FileNotFoundError(f"No policy files found in {POLICY_DIR}")

    # Extract version numbers and sort
    def extract_version(filepath):
        match = re.search(r'policy_v(\d+)', filepath.name)
        return int(match.group(1)) if match else 0

    latest_policy = max(policy_files, key=extract_version)
    policy_version = f"policy_v{extract_version(latest_policy):03d}"

    print(f"Loading policy: {latest_policy.name}")

    with open(latest_policy, 'r') as f:
        policy_content = f.read()

    return policy_content, policy_version


def load_dataset_data(num_samples=None):
    """Load the evaluation dataset and optionally sample random rows.

    Args:
        num_samples: If provided, randomly sample this many rows from the dataset.
                    If None, return the full dataset.
    """
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. "
            "Please run a02_download_hf_dataset.py first."
        )

    print(f"Loading dataset from: {DATA_DIR}")
    dataset = load_from_disk(str(data_path))

    # The dataset has a 'train' split
    if 'train' in dataset:
        dataset = dataset['train']

    # If num_samples is specified, randomly sample that many rows
    if num_samples is not None and num_samples < len(dataset):
        print(f"Randomly sampling {num_samples} rows from {len(dataset)} total rows...")
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
        print(f"Sampled {len(dataset)} rows")

    return dataset


def parse_json_response(response_text):
    """Extract and parse JSON from model response."""
    try:
        # Try to find JSON in the response
        # Sometimes the model includes markdown code blocks
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find a plain JSON object
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text

        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {response_text}")
        print(f"Error: {e}")
        return None


def run_evaluation(client, policy_content, dataset):
    """Run policy evaluation on dataset samples."""
    results = []
    total = len(dataset)

    print(f"\nRunning evaluation on {total} samples...")

    for idx, item in enumerate(dataset):
        prompt_text = item['prompt']

        # Get ground truth labels
        ground_truth = {cat: item.get(cat, None) for cat in CATEGORIES}

        try:
            # Call GPT-OSS-Safeguard
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": policy_content,
                    },
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                model=MODEL,
                temperature=0,  # Use deterministic outputs
            )

            response_text = chat_completion.choices[0].message.content
            predictions = parse_json_response(response_text)

            if predictions is None:
                predictions = {cat: None for cat in CATEGORIES}
                parse_error = True
            else:
                parse_error = False

            result = {
                "index": idx,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "predictions": predictions,
                "raw_response": response_text,
                "parse_error": parse_error,
            }

            results.append(result)

            # Progress indicator
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{total} samples...")

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            result = {
                "index": idx,
                "prompt": prompt_text,
                "ground_truth": ground_truth,
                "predictions": {cat: None for cat in CATEGORIES},
                "raw_response": None,
                "error": str(e),
            }
            results.append(result)

    print(f"Evaluation complete! Processed {len(results)} samples.")
    return results


def save_results(results, policy_version, num_rows_eval=None):
    """Save evaluation results with versioning."""
    # Create runs directory if it doesn't exist
    runs_dir = Path(RUNS_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Generate run identifier with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{policy_version}_{timestamp}"

    # Save detailed results as JSON
    results_file = runs_dir / f"{run_id}_results.json"

    output_data = {
        "run_id": run_id,
        "policy_version": policy_version,
        "timestamp": timestamp,
        "model": MODEL,
        "num_rows_to_eval": num_rows_eval,
        "total_samples": len(results),
        "results": results,
    }

    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Calculate and save summary statistics
    calculate_metrics(results, run_id, runs_dir)

    return run_id


def calculate_metrics(results, run_id, runs_dir):
    """Calculate accuracy, precision, recall, F1 for each category."""
    metrics = {}

    for cat in CATEGORIES:
        tp = fp = tn = fn = 0
        valid_samples = 0

        for result in results:
            gt = result['ground_truth'].get(cat)
            pred = result['predictions'].get(cat) if result['predictions'] else None

            # Skip if ground truth or prediction is missing
            if gt is None or pred is None:
                continue

            valid_samples += 1

            if gt == 1 and pred == 1:
                tp += 1
            elif gt == 0 and pred == 1:
                fp += 1
            elif gt == 0 and pred == 0:
                tn += 1
            elif gt == 1 and pred == 0:
                fn += 1

        if valid_samples == 0:
            continue

        accuracy = (tp + tn) / valid_samples if valid_samples > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cat] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "valid_samples": valid_samples,
        }

    # Save metrics
    metrics_file = runs_dir / f"{run_id}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {metrics_file}")

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION METRICS SUMMARY")
    print("="*60)
    for cat in CATEGORIES:
        if cat in metrics:
            m = metrics[cat]
            print(f"\n{cat}:")
            print(f"  Accuracy:  {m['accuracy']:.3f}")
            print(f"  Precision: {m['precision']:.3f}")
            print(f"  Recall:    {m['recall']:.3f}")
            print(f"  F1 Score:  {m['f1']:.3f}")
            print(f"  (TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']})")


def main():
    """Main execution function."""
    print("="*60)
    print("GPT-OSS-Safeguard Policy Evaluation")
    print("="*60)

    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    # Load policy
    policy_content, policy_version = get_latest_policy()
    print(f"Policy version: {policy_version}")

    # Get number of rows to evaluate from environment
    num_rows_str = os.environ.get("NUM_ROWS_TO_EVAL")
    num_rows = int(num_rows_str) if num_rows_str and num_rows_str.strip() else None

    if num_rows:
        print(f"NUM_ROWS_TO_EVAL set to: {num_rows}")
    else:
        print("NUM_ROWS_TO_EVAL not set - evaluating full dataset")

    # Load dataset
    dataset = load_dataset_data(num_samples=num_rows)
    print(f"Dataset size: {len(dataset)} samples")

    # Run evaluation on the sampled dataset
    results = run_evaluation(client, policy_content, dataset)

    # Save results
    run_id = save_results(results, policy_version, num_rows_eval=num_rows)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Run ID: {run_id}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
