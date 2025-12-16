"""Use LLM to automatically improve policy based on evaluation results."""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Configuration
POLICY_DIR = "./policy_pak/policy"
RUNS_DIR = "./policy_pak/runs"
JUDGE_MODEL = "openai/gpt-oss-120b"  # Using the larger model for policy modification

# Categories
CATEGORIES = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


def get_latest_run_files():
    """Find and load the latest run results and metrics files."""
    runs_dir = Path(RUNS_DIR)

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

    # Find all results files
    results_files = list(runs_dir.glob("*_results.json"))

    if not results_files:
        raise FileNotFoundError(f"No results files found in {RUNS_DIR}")

    # Sort by modification time to get the latest
    latest_results_file = max(results_files, key=lambda f: f.stat().st_mtime)

    # Get corresponding metrics file
    run_id = latest_results_file.stem.replace("_results", "")
    metrics_file = runs_dir / f"{run_id}_metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    print(f"Loading latest run files:")
    print(f"  Results: {latest_results_file.name}")
    print(f"  Metrics: {metrics_file.name}")

    with open(latest_results_file, 'r') as f:
        results_data = json.load(f)

    with open(metrics_file, 'r') as f:
        metrics_data = json.load(f)

    return results_data, metrics_data, run_id


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
    policy_version = extract_version(latest_policy)

    print(f"\nCurrent policy: {latest_policy.name} (version {policy_version})")

    with open(latest_policy, 'r') as f:
        policy_content = f.read()

    return policy_content, policy_version


def analyze_failures(results_data, metrics_data):
    """Analyze failures to provide context for policy improvement."""
    results = results_data['results']

    failures_by_category = {cat: {"false_positives": [], "false_negatives": []} for cat in CATEGORIES}

    for result in results:
        if result.get('parse_error') or result.get('error'):
            continue

        ground_truth = result['ground_truth']
        predictions = result['predictions']

        if not predictions:
            continue

        for cat in CATEGORIES:
            gt = ground_truth.get(cat)
            pred = predictions.get(cat)

            if gt is None or pred is None:
                continue

            # False positive: predicted 1, but ground truth is 0
            if gt == 0 and pred == 1:
                failures_by_category[cat]["false_positives"].append({
                    "index": result['index'],
                    "prompt": result['prompt'],
                    "raw_response": result.get('raw_response', ''),
                })

            # False negative: predicted 0, but ground truth is 1
            elif gt == 1 and pred == 0:
                failures_by_category[cat]["false_negatives"].append({
                    "index": result['index'],
                    "prompt": result['prompt'],
                    "raw_response": result.get('raw_response', ''),
                })

    return failures_by_category


def create_policy_improvement_prompt(policy_content, metrics_data, failures_by_category, max_examples_per_type=5):
    """Create a detailed prompt for the LLM to improve the policy."""

    prompt = f"""You are an expert in content moderation policy design. Your task is to improve a content moderation policy based on evaluation results.

## CURRENT POLICY

{policy_content}

## EVALUATION METRICS

The policy was evaluated on a dataset and achieved the following metrics:

"""

    # Add metrics summary
    for cat in CATEGORIES:
        if cat in metrics_data:
            m = metrics_data[cat]
            prompt += f"""
### Category: {cat}
- Accuracy:  {m['accuracy']:.3f}
- Precision: {m['precision']:.3f}
- Recall:    {m['recall']:.3f}
- F1 Score:  {m['f1']:.3f}
- True Positives:  {m['tp']}
- False Positives: {m['fp']}
- True Negatives:  {m['tn']}
- False Negatives: {m['fn']}
"""

    prompt += "\n## FAILURE ANALYSIS\n\n"
    prompt += "Below are examples of misclassifications that need to be addressed:\n\n"

    # Add failure examples
    for cat in CATEGORIES:
        fp_examples = failures_by_category[cat]["false_positives"]
        fn_examples = failures_by_category[cat]["false_negatives"]

        if fp_examples or fn_examples:
            prompt += f"### Category: {cat}\n\n"

            if fp_examples:
                prompt += f"**False Positives** (predicted violation, but should be safe): {len(fp_examples)} total\n\n"
                for i, example in enumerate(fp_examples[:max_examples_per_type]):
                    prompt += f"Example {i+1}:\n"
                    prompt += f"Content: {example['prompt'][:500]}{'...' if len(example['prompt']) > 500 else ''}\n\n"

            if fn_examples:
                prompt += f"**False Negatives** (predicted safe, but should be violation): {len(fn_examples)} total\n\n"
                for i, example in enumerate(fn_examples[:max_examples_per_type]):
                    prompt += f"Example {i+1}:\n"
                    prompt += f"Content: {example['prompt'][:500]}{'...' if len(example['prompt']) > 500 else ''}\n\n"

    prompt += """
## YOUR TASK

Based on the evaluation metrics and failure examples above, improve the content moderation policy to:

1. **Reduce false positives**: Make criteria more precise to avoid flagging safe content
2. **Reduce false negatives**: Expand criteria to catch violations that were missed
3. **Maintain clarity**: Keep the policy clear, specific, and unambiguous
4. **Preserve structure**: Maintain the four-section format (Instructions, Definitions, Criteria, Examples)
5. **Focus on failures**: Prioritize fixes for categories with the lowest F1 scores

## GUIDELINES FOR IMPROVEMENT

- If precision is low (many false positives), make the VIOLATES criteria more specific and narrow
- If recall is low (many false negatives), expand the VIOLATES criteria to cover more cases
- Update examples to better illustrate edge cases from the failure analysis
- Clarify definitions where ambiguity led to errors
- Add specific guidance for borderline cases

## OUTPUT INSTRUCTIONS

Return ONLY the complete improved policy in markdown format. Do not include explanations, meta-commentary, or anything else - just the policy text that can be directly saved to a file.

Start your response with "# Content Moderation Policy" and include all sections.
"""

    return prompt


def generate_improved_policy(client, prompt):
    """Use LLM to generate an improved policy based on the prompt."""
    print("\nGenerating improved policy with LLM...")
    print(f"Using model: {JUDGE_MODEL}")

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=JUDGE_MODEL,
            temperature=0.3,  # Some creativity but mostly deterministic
            max_tokens=8000,  # Need enough tokens for a full policy
        )

        improved_policy = chat_completion.choices[0].message.content

        # Clean up any markdown code blocks if present
        improved_policy = re.sub(r'^```markdown\s*', '', improved_policy)
        improved_policy = re.sub(r'^```\s*', '', improved_policy)
        improved_policy = re.sub(r'\s*```$', '', improved_policy)

        return improved_policy

    except Exception as e:
        print(f"Error generating improved policy: {e}")
        raise


def save_new_policy(policy_content, current_version):
    """Save the improved policy as a new version."""
    policy_dir = Path(POLICY_DIR)
    policy_dir.mkdir(parents=True, exist_ok=True)

    # Increment version
    new_version = current_version + 1
    new_policy_file = policy_dir / f"policy_v{new_version:03d}.md"

    # Check if file already exists
    if new_policy_file.exists():
        print(f"\nWarning: {new_policy_file.name} already exists!")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled. Policy not saved.")
            return None

    with open(new_policy_file, 'w') as f:
        f.write(policy_content)

    print(f"\nNew policy saved: {new_policy_file.name}")
    return new_policy_file


def main():
    """Main execution function."""
    print("="*60)
    print("Automatic Policy Improvement")
    print("="*60)

    # Initialize Groq client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    # Load latest run results
    results_data, metrics_data, run_id = get_latest_run_files()
    print(f"Run ID: {run_id}")
    print(f"Total samples: {results_data['total_samples']}")

    # Load current policy
    policy_content, policy_version = get_latest_policy()

    # Analyze failures
    print("\nAnalyzing failures...")
    failures_by_category = analyze_failures(results_data, metrics_data)

    # Count failures
    total_fp = sum(len(failures_by_category[cat]["false_positives"]) for cat in CATEGORIES)
    total_fn = sum(len(failures_by_category[cat]["false_negatives"]) for cat in CATEGORIES)
    print(f"Total false positives: {total_fp}")
    print(f"Total false negatives: {total_fn}")

    # Create improvement prompt
    prompt = create_policy_improvement_prompt(
        policy_content,
        metrics_data,
        failures_by_category,
        max_examples_per_type=5
    )

    # Print prompt length for debugging
    print(f"\nPrompt length: {len(prompt)} characters")

    # Generate improved policy
    improved_policy = generate_improved_policy(client, prompt)

    # Preview the improved policy
    print("\n" + "="*60)
    print("IMPROVED POLICY PREVIEW")
    print("="*60)
    print(improved_policy[:1000] + "..." if len(improved_policy) > 1000 else improved_policy)
    print("="*60)

    # Ask for confirmation
    response = input("\nSave this improved policy as a new version? (y/n): ")

    if response.lower() == 'y':
        new_policy_file = save_new_policy(improved_policy, policy_version)
        if new_policy_file:
            print(f"\n{'='*60}")
            print(f"Success! New policy version created: {new_policy_file.name}")
            print(f"You can now run a03_run_policy.py to evaluate the new policy.")
            print(f"{'='*60}")
    else:
        print("\nPolicy improvement cancelled. No new version created.")


if __name__ == "__main__":
    main()
