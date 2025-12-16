from gpt_oss_utils_dactile.classify import classify
from tqdm import tqdm
import json
from itertools import islice
from ..model_predict import InferenceBackend, Model
from pathlib import Path
from gpt_oss_utils_dactile.policies import toxic_simple
from gpt_oss_utils_dactile.policies.hackathon import is_good
from gpt_oss_utils_dactile.hackathon.analyize_comparison import compare_results, print_analysis


cur_path = Path(__file__).parent
controversial_path = cur_path / "controversial-micro.jsonl"
results_path = cur_path / "results.json"

def read_all_controversial():
    with open(controversial_path, "r") as f:
        for line in f:
            data = json.loads(line)
            yield data
    

def run_all_controversial(save=True):
    results = []
    all_data = list(read_all_controversial())
    for data in tqdm(all_data):
        r = {**data}
        for model in [Model.GPT_OSS_20B, Model.GPT_OSS_safeguard_20B]:
            text = data["text"]
            print(text)
            result = classify(
                text=text,
                policy_module=is_good,
                model=model,
                backend=InferenceBackend.API_INJECT_HARMONY,
            )
            print(f"Classification Results: {result.float_label}")
            r[f"model_{model.value}"] = result.float_label
        results.append(r)
    print(results)
    
    if save:
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    return results


def load_results():
    """Load previously saved results."""
    if results_path.exists():
        with open(results_path, "r") as f:
            return json.load(f)
    return None


def analyze_saved_results():
    """Load and analyze previously saved results."""
    results = load_results()
    if results is None:
        print("No saved results found. Run run_all_controversial() first.")
        return None
    
    analysis = compare_results(results)
    print_analysis(analysis)
    return analysis


def hello_controversial():
    opinion = "Giving your kid an iPad is lazy parenting, full stop."
    
    print(f"Opinion to classify: {opinion}\n")
    
    # Classify the opinion using the toxic_simple policy
    result = classify(
        text=opinion,
        policy_module=is_good,
        model=Model.GPT_OSS_20B,
        backend=InferenceBackend.API_INJECT_HARMONY,
    )
    
    print("Classification Results:")
    print(f"  Binary Label: {result.binary_label}")
    print(f"  Fine-grain Label: {result.fine_grain_label}")
    print(f"  Parsed Successfully: {result.parsed_successfully}")
    print(f"\nModel Response:")
    print(f"  {result.model_response.response}")
    
    return result


if __name__ == "__main__":
    #hello_controversial()
    #exit()
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # Just analyze saved results
        analyze_saved_results()
    else:
        # Run classification and then analyze
        results = load_results()
        if results is None:
            results = run_all_controversial()
        print("\n" + "=" * 70)
        print("ANALYSIS")
        print("=" * 70)
        analysis = compare_results(results)
        print_analysis(analysis)
        