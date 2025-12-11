"""Policy evaluation and modification tools."""

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from dotenv import load_dotenv
from groq import Groq
from datasets import load_from_disk

from policy_cli.tools.base import BaseTool
from policy_cli.core.types import ToolResult, SimpleToolResult

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# Configuration
POLICY_DIR = "./policy_pak/policy"
DATA_DIR = "./policy_pak/data"
RUNS_DIR = "./policy_pak/runs"
EVAL_MODEL = "openai/gpt-oss-safeguard-20b"
JUDGE_MODEL = "openai/gpt-oss-120b"
CATEGORIES = ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"]


class RunPolicyTool(BaseTool):
    """Tool for running policy evaluation against the dataset."""

    def __init__(self):
        super().__init__(
            name="run_policy",
            description="Run the latest policy evaluation against the dataset. Returns metrics (accuracy, precision, recall, F1) for each category."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_samples": {
                    "type": "integer",
                    "description": "Number of random samples to evaluate (default: from NUM_ROWS_TO_EVAL env var or all)",
                    "default": None
                }
            },
            "required": []
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            # Get Groq API key
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                return SimpleToolResult(
                    content="Error: GROQ_API_KEY not found in environment variables",
                    error="Missing API key"
                )

            client = Groq(api_key=api_key)

            # Load latest policy
            policy_content, policy_version = self._get_latest_policy()

            # Get number of samples
            num_samples = parameters.get("num_samples")
            if num_samples is None:
                num_rows_str = os.environ.get("NUM_ROWS_TO_EVAL")
                num_samples = int(num_rows_str) if num_rows_str and num_rows_str.strip() else None

            # Load dataset
            dataset = self._load_dataset(num_samples)

            # Run evaluation
            results = self._run_evaluation(client, policy_content, dataset)

            # Save results
            run_id, metrics = self._save_results(results, policy_version, num_samples)

            # Format output
            summary = self._format_summary(run_id, policy_version, len(results), metrics)

            return SimpleToolResult(
                content=summary,
                display_content=f"Policy evaluation complete. Run ID: {run_id}"
            )

        except Exception as e:
            logger.error(f"Error running policy: {e}", exc_info=True)
            return SimpleToolResult(
                content=f"Error running policy evaluation: {str(e)}",
                error=str(e)
            )

    def _get_latest_policy(self):
        """Find and load the latest policy version."""
        policy_dir = Path(POLICY_DIR)
        if not policy_dir.exists():
            raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

        policy_files = list(policy_dir.glob("policy_v*.md"))
        if not policy_files:
            raise FileNotFoundError(f"No policy files found in {POLICY_DIR}")

        def extract_version(filepath):
            match = re.search(r'policy_v(\d+)', filepath.name)
            return int(match.group(1)) if match else 0

        latest_policy = max(policy_files, key=extract_version)
        policy_version = f"policy_v{extract_version(latest_policy):03d}"

        with open(latest_policy, 'r') as f:
            policy_content = f.read()

        return policy_content, policy_version

    def _load_dataset(self, num_samples=None):
        """Load the evaluation dataset and optionally sample random rows."""
        data_path = Path(DATA_DIR)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {DATA_DIR}")

        dataset = load_from_disk(str(data_path))
        if 'train' in dataset:
            dataset = dataset['train']

        if num_samples is not None and num_samples < len(dataset):
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)

        return dataset

    def _run_evaluation(self, client, policy_content, dataset):
        """Run policy evaluation on dataset samples."""
        results = []
        total = len(dataset)

        for idx, item in enumerate(dataset):
            prompt_text = item['prompt']
            ground_truth = {cat: item.get(cat, None) for cat in CATEGORIES}

            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": policy_content},
                        {"role": "user", "content": prompt_text}
                    ],
                    model=EVAL_MODEL,
                    temperature=0,
                )

                response_text = chat_completion.choices[0].message.content
                predictions = self._parse_json_response(response_text)

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

            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                result = {
                    "index": idx,
                    "prompt": prompt_text,
                    "ground_truth": ground_truth,
                    "predictions": {cat: None for cat in CATEGORIES},
                    "raw_response": None,
                    "error": str(e),
                }
                results.append(result)

        return results

    def _parse_json_response(self, response_text):
        """Extract and parse JSON from model response."""
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text

            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def _save_results(self, results, policy_version, num_rows_eval=None):
        """Save evaluation results with versioning."""
        runs_dir = Path(RUNS_DIR)
        runs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{policy_version}_{timestamp}"

        results_file = runs_dir / f"{run_id}_results.json"
        output_data = {
            "run_id": run_id,
            "policy_version": policy_version,
            "timestamp": timestamp,
            "model": EVAL_MODEL,
            "num_rows_to_eval": num_rows_eval,
            "total_samples": len(results),
            "results": results,
        }

        with open(results_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Calculate metrics
        metrics = self._calculate_metrics(results)
        metrics_file = runs_dir / f"{run_id}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        return run_id, metrics

    def _calculate_metrics(self, results):
        """Calculate accuracy, precision, recall, F1 for each category."""
        metrics = {}

        for cat in CATEGORIES:
            tp = fp = tn = fn = 0
            valid_samples = 0

            for result in results:
                gt = result['ground_truth'].get(cat)
                pred = result['predictions'].get(cat) if result['predictions'] else None

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

        return metrics

    def _format_summary(self, run_id, policy_version, total_samples, metrics):
        """Format evaluation results as a summary."""
        summary = f"# Policy Evaluation Results\n\n"
        summary += f"**Run ID:** {run_id}\n"
        summary += f"**Policy:** {policy_version}\n"
        summary += f"**Samples:** {total_samples}\n\n"
        summary += "## Metrics by Category\n\n"

        for cat in CATEGORIES:
            if cat in metrics:
                m = metrics[cat]
                summary += f"### {cat}\n"
                summary += f"- Accuracy: {m['accuracy']:.3f}\n"
                summary += f"- Precision: {m['precision']:.3f}\n"
                summary += f"- Recall: {m['recall']:.3f}\n"
                summary += f"- F1 Score: {m['f1']:.3f}\n"
                summary += f"- TP={m['tp']}, FP={m['fp']}, TN={m['tn']}, FN={m['fn']}\n\n"

        return summary


class ModifyPolicyTool(BaseTool):
    """Tool for automatically improving policy based on evaluation results."""

    def __init__(self):
        super().__init__(
            name="modify_policy",
            description="Automatically improve the policy based on the latest evaluation results. Uses AI to analyze failures and generate an improved policy version."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_examples_per_type": {
                    "type": "integer",
                    "description": "Maximum number of failure examples to include per type (default: 5)",
                    "default": 5
                }
            },
            "required": []
        }

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            # Get Groq API key
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                return SimpleToolResult(
                    content="Error: GROQ_API_KEY not found in environment variables",
                    error="Missing API key"
                )

            client = Groq(api_key=api_key)

            # Load latest run results
            results_data, metrics_data, run_id = self._get_latest_run_files()

            # Load current policy
            policy_content, policy_version = self._get_latest_policy()

            # Analyze failures
            failures_by_category = self._analyze_failures(results_data, metrics_data)

            # Create improvement prompt
            max_examples = parameters.get("max_examples_per_type", 5)
            prompt = self._create_improvement_prompt(
                policy_content, metrics_data, failures_by_category, max_examples
            )

            # Generate improved policy
            improved_policy = self._generate_improved_policy(client, prompt)

            # Save new policy version
            new_version = policy_version
            new_policy_file = self._save_new_policy(improved_policy, policy_version)

            summary = f"# Policy Improvement Complete\n\n"
            summary += f"**Original Policy:** {policy_version}\n"
            summary += f"**New Policy:** {new_policy_file.name}\n"
            summary += f"**Based on Run:** {run_id}\n\n"
            summary += f"The improved policy has been saved to: {new_policy_file}\n\n"
            summary += "You can now run the new policy with the run_policy tool to see if it performs better."

            return SimpleToolResult(
                content=summary,
                display_content=f"Policy improved and saved as {new_policy_file.name}"
            )

        except Exception as e:
            logger.error(f"Error modifying policy: {e}", exc_info=True)
            return SimpleToolResult(
                content=f"Error modifying policy: {str(e)}",
                error=str(e)
            )

    def _get_latest_run_files(self):
        """Find and load the latest run results and metrics files."""
        runs_dir = Path(RUNS_DIR)
        if not runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {RUNS_DIR}")

        results_files = list(runs_dir.glob("*_results.json"))
        if not results_files:
            raise FileNotFoundError(f"No results files found in {RUNS_DIR}")

        latest_results_file = max(results_files, key=lambda f: f.stat().st_mtime)
        run_id = latest_results_file.stem.replace("_results", "")
        metrics_file = runs_dir / f"{run_id}_metrics.json"

        if not metrics_file.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

        with open(latest_results_file, 'r') as f:
            results_data = json.load(f)

        with open(metrics_file, 'r') as f:
            metrics_data = json.load(f)

        return results_data, metrics_data, run_id

    def _get_latest_policy(self):
        """Find and load the latest policy version."""
        policy_dir = Path(POLICY_DIR)
        if not policy_dir.exists():
            raise FileNotFoundError(f"Policy directory not found: {POLICY_DIR}")

        policy_files = list(policy_dir.glob("policy_v*.md"))
        if not policy_files:
            raise FileNotFoundError(f"No policy files found in {POLICY_DIR}")

        def extract_version(filepath):
            match = re.search(r'policy_v(\d+)', filepath.name)
            return int(match.group(1)) if match else 0

        latest_policy = max(policy_files, key=extract_version)
        policy_version = extract_version(latest_policy)

        with open(latest_policy, 'r') as f:
            policy_content = f.read()

        return policy_content, policy_version

    def _analyze_failures(self, results_data, metrics_data):
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

                if gt == 0 and pred == 1:
                    failures_by_category[cat]["false_positives"].append({
                        "index": result['index'],
                        "prompt": result['prompt'],
                        "raw_response": result.get('raw_response', ''),
                    })
                elif gt == 1 and pred == 0:
                    failures_by_category[cat]["false_negatives"].append({
                        "index": result['index'],
                        "prompt": result['prompt'],
                        "raw_response": result.get('raw_response', ''),
                    })

        return failures_by_category

    def _create_improvement_prompt(self, policy_content, metrics_data, failures_by_category, max_examples_per_type=5):
        """Create a detailed prompt for the LLM to improve the policy."""
        prompt = f"""You are an expert in content moderation policy design. Your task is to improve a content moderation policy based on evaluation results.

## CURRENT POLICY

{policy_content}

## EVALUATION METRICS

The policy was evaluated on a dataset and achieved the following metrics:

"""
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

    def _generate_improved_policy(self, client, prompt):
        """Use LLM to generate an improved policy based on the prompt."""
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=JUDGE_MODEL,
            temperature=0.3,
            max_tokens=8000,
        )

        improved_policy = chat_completion.choices[0].message.content

        # Clean up any markdown code blocks
        improved_policy = re.sub(r'^```markdown\s*', '', improved_policy)
        improved_policy = re.sub(r'^```\s*', '', improved_policy)
        improved_policy = re.sub(r'\s*```$', '', improved_policy)

        return improved_policy

    def _save_new_policy(self, policy_content, current_version):
        """Save the improved policy as a new version."""
        policy_dir = Path(POLICY_DIR)
        policy_dir.mkdir(parents=True, exist_ok=True)

        new_version = current_version + 1
        new_policy_file = policy_dir / f"policy_v{new_version:03d}.md"

        with open(new_policy_file, 'w') as f:
            f.write(policy_content)

        return new_policy_file
