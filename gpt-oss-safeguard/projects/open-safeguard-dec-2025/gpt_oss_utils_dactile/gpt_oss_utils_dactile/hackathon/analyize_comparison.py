"""Analyze comparison between two model classifications."""
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def compare_results(results: list[dict], model1_key: str = None, model2_key: str = None) -> dict:
    """
    Analyze and compare classification results from two models.
    
    Args:
        results: List of dicts, each containing:
            - "text": the input text
            - model keys with float_label values
        model1_key: Key for first model (auto-detected if None)
        model2_key: Key for second model (auto-detected if None)
    
    Returns:
        Dictionary with analysis results
    """
    # Auto-detect model keys if not provided
    if model1_key is None or model2_key is None:
        if results:
            model_keys = [k for k in results[0].keys() if k.startswith("model_")]
            if len(model_keys) >= 2:
                model1_key = model_keys[0]
                model2_key = model_keys[1]
            else:
                raise ValueError(f"Could not find two model keys. Found: {model_keys}")
        else:
            raise ValueError("No results to analyze")
    
    print(f"Comparing: {model1_key} vs {model2_key}")
    
    # Extract scores (filter out None values)
    model1_scores = []
    model2_scores = []
    texts = []
    
    for r in results:
        s1 = r.get(model1_key)
        s2 = r.get(model2_key)
        if s1 is not None and s2 is not None:
            model1_scores.append(s1)
            model2_scores.append(s2)
            texts.append(r["text"])
    
    model1_scores = np.array(model1_scores)
    model2_scores = np.array(model2_scores)
    
    analysis = {}
    
    # =========================================================================
    # 1. Distribution of labels from each model
    # =========================================================================
    # Convert back to 1-5 scale for interpretability
    model1_likert = (model1_scores * 4 + 1).round().astype(int)
    model2_likert = (model2_scores * 4 + 1).round().astype(int)
    
    analysis["distribution"] = {
        "model1": {
            "name": model1_key,
            "counts": {int(k): int(v) for k, v in Counter(model1_likert).items()},
            "mean": float(np.mean(model1_scores)),
            "std": float(np.std(model1_scores)),
            "mean_likert": float(np.mean(model1_likert)),
        },
        "model2": {
            "name": model2_key, 
            "counts": {int(k): int(v) for k, v in Counter(model2_likert).items()},
            "mean": float(np.mean(model2_scores)),
            "std": float(np.std(model2_scores)),
            "mean_likert": float(np.mean(model2_likert)),
        }
    }
    
    # =========================================================================
    # 1b. Test for specific Likert value differences (e.g., proportion of "3"s)
    # =========================================================================
    # Test if proportion of a specific Likert value differs between models
    likert_value_tests = {}
    for likert_val in [1, 2, 3, 4, 5]:
        n1 = np.sum(model1_likert == likert_val)
        n2 = np.sum(model2_likert == likert_val)
        total1 = len(model1_likert)
        total2 = len(model2_likert)
        p1 = n1 / total1 if total1 > 0 else 0
        p2 = n2 / total2 if total2 > 0 else 0
        
        # Two-proportion z-test
        # H0: p1 = p2, H1: p1 != p2
        if total1 > 0 and total2 > 0:
            # Pooled proportion for standard error
            p_pooled = (n1 + n2) / (total1 + total2)
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total1 + 1/total2))
            if se > 0:
                z_stat = (p1 - p2) / se
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            else:
                z_stat = 0
                p_value = 1.0
        else:
            z_stat = np.nan
            p_value = np.nan
        
        likert_value_tests[likert_val] = {
            "model1_count": int(n1),
            "model1_proportion": float(p1),
            "model2_count": int(n2),
            "model2_proportion": float(p2),
            "z_statistic": float(z_stat) if not np.isnan(z_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
            "difference": float(p2 - p1),  # model2 - model1
        }
    
    analysis["likert_value_tests"] = likert_value_tests
    
    # =========================================================================
    # 2. Typical differences
    # =========================================================================
    differences = model1_scores - model2_scores
    abs_differences = np.abs(differences)
    
    # Any difference at all (accounting for floating point)
    has_difference = abs_differences > 0.001
    
    analysis["differences"] = {
        "mean_difference": float(np.mean(differences)),  # Signed: positive = model1 higher
        "mean_abs_difference": float(np.mean(abs_differences)),
        "std_difference": float(np.std(differences)),
        "fraction_different": float(np.mean(has_difference)),
        "n_different": int(np.sum(has_difference)),
        "n_total": len(differences),
    }
    
    # =========================================================================
    # 3. Statistical tests
    # =========================================================================
    # Paired t-test (parametric) - tests if mean difference is significantly != 0
    if len(model1_scores) >= 2:
        t_stat, t_pvalue = stats.ttest_rel(model1_scores, model2_scores)
    else:
        t_stat, t_pvalue = np.nan, np.nan
    
    # Wilcoxon signed-rank test (non-parametric) - better for small samples
    # Tests if the distribution of differences is symmetric around zero
    if len(model1_scores) >= 5 and np.any(differences != 0):
        try:
            w_stat, w_pvalue = stats.wilcoxon(model1_scores, model2_scores)
        except ValueError:
            # All differences are zero
            w_stat, w_pvalue = np.nan, 1.0
    else:
        w_stat, w_pvalue = np.nan, np.nan
    
    # Sign test - simplest non-parametric, just counts which is higher
    n_model1_higher = np.sum(differences > 0.001)
    n_model2_higher = np.sum(differences < -0.001)
    n_ties = np.sum(np.abs(differences) <= 0.001)
    
    # Binomial test for sign test
    if n_model1_higher + n_model2_higher > 0:
        sign_pvalue = stats.binom_test(
            n_model1_higher, 
            n_model1_higher + n_model2_higher, 
            0.5, 
            alternative='two-sided'
        ) if hasattr(stats, 'binom_test') else stats.binomtest(
            n_model1_higher,
            n_model1_higher + n_model2_higher,
            0.5,
            alternative='two-sided'
        ).pvalue
    else:
        sign_pvalue = 1.0
    
    analysis["statistical_tests"] = {
        "paired_ttest": {
            "description": "Tests if mean difference is significantly different from 0",
            "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
            "p_value": float(t_pvalue) if not np.isnan(t_pvalue) else None,
            "interpretation": "Assumes normal distribution of differences",
        },
        "wilcoxon_signed_rank": {
            "description": "Non-parametric test for paired samples",
            "w_statistic": float(w_stat) if not np.isnan(w_stat) else None,
            "p_value": float(w_pvalue) if not np.isnan(w_pvalue) else None,
            "interpretation": "Better for small samples, doesn't assume normality",
        },
        "sign_test": {
            "description": "Counts which model scores higher more often",
            "n_model1_higher": int(n_model1_higher),
            "n_model2_higher": int(n_model2_higher),
            "n_ties": int(n_ties),
            "p_value": float(sign_pvalue),
            "interpretation": "Simplest test - just compares direction of differences",
        }
    }
    
    # =========================================================================
    # 5. Inter-rater reliability (ICC and correlations)
    # =========================================================================
    # Prepare data for ICC calculation (needs long format)
    n = len(model1_scores)
    icc_data = pd.DataFrame({
        'targets': list(range(n)) * 2,
        'raters': ['model1'] * n + ['model2'] * n,
        'ratings': np.concatenate([model1_scores, model2_scores])
    })
    
    # Calculate ICC using pingouin
    icc_results = pg.intraclass_corr(data=icc_data, targets='targets', 
                                      raters='raters', ratings='ratings')
    
    # Extract ICC3 (two-way mixed, absolute agreement, single rater)
    icc3_row = icc_results[icc_results['Type'] == 'ICC3'].iloc[0]
    
    # Calculate correlations
    from scipy.stats import pearsonr, spearmanr
    r_pearson, p_pearson = pearsonr(model1_scores, model2_scores)
    r_spearman, p_spearman = spearmanr(model1_scores, model2_scores)
    
    analysis["inter_rater_reliability"] = {
        "icc3": {
            "description": "ICC(3,1): Two-way mixed effects, absolute agreement, single rater",
            "icc": float(icc3_row['ICC']),
            "ci_95_lower": float(icc3_row['CI95%'][0]),
            "ci_95_upper": float(icc3_row['CI95%'][1]),
            "p_value": float(icc3_row['pval']),
            "interpretation": _interpret_icc(float(icc3_row['ICC'])),
            "meaning": "Proportion of variance due to true differences between items (vs rater disagreement)"
        },
        "pearson_correlation": {
            "description": "Linear correlation between ratings",
            "r": float(r_pearson),
            "p_value": float(p_pearson),
            "interpretation": "Measures linear relationship (doesn't penalize systematic bias)"
        },
        "spearman_correlation": {
            "description": "Rank-order correlation between ratings",
            "rho": float(r_spearman),
            "p_value": float(p_spearman),
            "interpretation": "Measures agreement on ranking (robust to outliers)"
        }
    }
    
    # =========================================================================
    # 4. Instances with largest differences
    # =========================================================================
    # Sort by absolute difference
    sorted_indices = np.argsort(-abs_differences)
    
    most_different = []
    for idx in sorted_indices[:10]:  # Top 10 most different
        most_different.append({
            "text": texts[idx],
            "model1_score": float(model1_scores[idx]),
            "model2_score": float(model2_scores[idx]),
            "model1_likert": int(model1_likert[idx]),
            "model2_likert": int(model2_likert[idx]),
            "difference": float(differences[idx]),
            "abs_difference": float(abs_differences[idx]),
        })
    
    analysis["most_different_instances"] = most_different
    
    return analysis


def _short_model_name(name: str) -> str:
    """Extract a short model name from the full key."""
    # e.g., "model_openai/gpt-oss-20b" -> "gpt-oss-20b"
    if name.startswith("model_"):
        name = name[6:]
    if "/" in name:
        name = name.split("/")[-1]
    return name


def _interpret_icc(icc: float) -> str:
    """Interpret ICC value according to standard guidelines."""
    if icc < 0.5:
        return "Poor reliability"
    elif icc < 0.75:
        return "Moderate reliability"
    elif icc < 0.9:
        return "Good reliability"
    else:
        return "Excellent reliability"


def plot_likert_distribution(analysis: dict, output_path: str = None, figsize=(8, 5)):
    """
    Create a bar plot comparing Likert rating distributions between two models.
    Tufte-style: minimal ink, transparent background, clean design.
    
    Args:
        analysis: Dictionary from compare_results()
        output_path: Path to save figure (if None, displays instead)
        figsize: Figure size tuple (width, height)
    
    Returns:
        matplotlib figure object
    """
    model1_name = _short_model_name(analysis["distribution"]["model1"]["name"])
    model2_name = _short_model_name(analysis["distribution"]["model2"]["name"])
    
    model1_counts = analysis["distribution"]["model1"]["counts"]
    model2_counts = analysis["distribution"]["model2"]["counts"]
    
    # Get total counts for each model to compute proportions
    model1_total = sum(model1_counts.values())
    model2_total = sum(model2_counts.values())
    
    # Prepare data for all Likert values (1-5)
    likert_values = [1, 2, 3, 4, 5]
    model1_props = [model1_counts.get(v, 0) / model1_total for v in likert_values]
    model2_props = [model2_counts.get(v, 0) / model2_total for v in likert_values]
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=figsize, facecolor='none')
    ax.set_facecolor('none')
    
    # Set bar width and positions
    bar_width = 0.35
    x = np.arange(len(likert_values))
    
    # Tufte-style colors: muted, distinguishable
    color1 = '#4C72B0'  # Muted blue
    color2 = '#C44E52'  # Muted red
    
    # Create bars with minimal styling
    bars1 = ax.bar(x - bar_width/2, model1_props, bar_width, 
                   label=model1_name, alpha=0.85, color=color1,
                   edgecolor='none')
    bars2 = ax.bar(x + bar_width/2, model2_props, bar_width,
                   label=model2_name, alpha=0.85, color=color2,
                   edgecolor='none')
    
    # Minimal axis styling
    ax.set_xlabel('Likert Rating', fontsize=11)
    ax.set_ylabel('Proportion', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(likert_values)
    
    # Remove top and right spines (Tufte style)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Subtle grid on y-axis only
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)  # Grid behind bars
    
    # Legend with minimal styling
    legend = ax.legend(fontsize=10, frameon=False, loc='upper left')
    
    # Add value labels on bars (only for significant values)
    def add_value_labels(bars, props):
        for bar, prop in zip(bars, props):
            height = bar.get_height()
            if height > 0.05:  # Only show label if bar is visible enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{prop:.0%}',
                       ha='center', va='bottom', fontsize=9, color='#333333')
    
    add_value_labels(bars1, model1_props)
    add_value_labels(bars2, model2_props)
    
    # Set y-axis limits with some padding
    ax.set_ylim(0, max(max(model1_props), max(model2_props)) * 1.15)
    
    # Format y-axis as percentages
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    
    plt.tight_layout()
    
    if output_path:
        # Save as SVG with transparent background
        plt.savefig(output_path, format='svg', bbox_inches='tight', 
                   facecolor='none', edgecolor='none')
        print(f"Figure saved to: {output_path}")
    
    return fig


def print_analysis(analysis: dict):
    """Pretty print the analysis results."""
    model1_name = _short_model_name(analysis["distribution"]["model1"]["name"])
    model2_name = _short_model_name(analysis["distribution"]["model2"]["name"])
    
    print("=" * 70)
    print("DISTRIBUTION OF LABELS")
    print("=" * 70)
    
    for model_key in ["model1", "model2"]:
        m = analysis["distribution"][model_key]
        print(f"\n{m['name']}:")
        print(f"  Likert counts: {m['counts']}")
        print(f"  Mean (0-1 scale): {m['mean']:.3f}")
        print(f"  Mean (1-5 scale): {m['mean_likert']:.2f}")
        print(f"  Std (0-1 scale): {m['std']:.3f}")
    
    print("\n" + "=" * 70)
    print("PROPORTION TESTS FOR EACH LIKERT VALUE")
    print("=" * 70)
    
    lvt = analysis["likert_value_tests"]
    print("\nTesting if proportions differ between models (two-proportion z-test):")
    for likert_val in sorted(lvt.keys()):
        test = lvt[likert_val]
        print(f"\nLikert value {likert_val}:")
        print(f"  {model1_name}: {test['model1_count']} ({test['model1_proportion']:.1%})")
        print(f"  {model2_name}: {test['model2_count']} ({test['model2_proportion']:.1%})")
        print(f"  Difference ({model2_name} - {model1_name}): {test['difference']:+.1%}")
        if test['p_value'] is not None:
            p = test['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  p-value: {p:.4f} {sig}")
            if p < 0.05:
                if test['difference'] > 0:
                    print(f"  → {model2_name} has significantly MORE {likert_val}s")
                else:
                    print(f"  → {model1_name} has significantly MORE {likert_val}s")
    
    print("\n" + "=" * 70)
    print("DIFFERENCES BETWEEN MODELS")
    print("=" * 70)
    
    d = analysis["differences"]
    print(f"\nMean difference ({model1_name} - {model2_name}): {d['mean_difference']:+.3f}")
    print(f"Mean absolute difference: {d['mean_abs_difference']:.3f}")
    print(f"Std of differences: {d['std_difference']:.3f}")
    print(f"Fraction with any difference: {d['fraction_different']:.1%} ({d['n_different']}/{d['n_total']})")
    
    print("\n" + "=" * 70)
    print("INTER-RATER RELIABILITY")
    print("=" * 70)
    
    irr = analysis["inter_rater_reliability"]
    
    print("\nICC(3,1) - Intraclass Correlation Coefficient:")
    icc_info = irr["icc3"]
    print(f"  ICC = {icc_info['icc']:.3f} (95% CI: [{icc_info['ci_95_lower']:.3f}, {icc_info['ci_95_upper']:.3f}])")
    print(f"  Interpretation: {icc_info['interpretation']}")
    print(f"  p-value: {icc_info['p_value']:.4f}")
    print(f"  → {icc_info['icc']*100:.1f}% of variance is true differences between items")
    print(f"  → {(1-icc_info['icc'])*100:.1f}% of variance is rater disagreement")
    
    print("\nCorrelations:")
    print(f"  Pearson r = {irr['pearson_correlation']['r']:.3f} (p = {irr['pearson_correlation']['p_value']:.4f})")
    print(f"  Spearman ρ = {irr['spearman_correlation']['rho']:.3f} (p = {irr['spearman_correlation']['p_value']:.4f})")
    
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Mean Differences)")
    print("=" * 70)
    
    for test_name, test in analysis["statistical_tests"].items():
        print(f"\n{test_name}:")
        print(f"  {test['description']}")
        if test.get('p_value') is not None:
            p = test['p_value']
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  p-value: {p:.4f} {sig}")
        if 'n_model1_higher' in test:
            print(f"  {model1_name} higher: {test['n_model1_higher']}, {model2_name} higher: {test['n_model2_higher']}, Ties: {test['n_ties']}")
    
    print("\n" + "=" * 70)
    print("MOST DIFFERENT INSTANCES")
    print("=" * 70)
    
    for i, inst in enumerate(analysis["most_different_instances"], 1):
        if inst["abs_difference"] < 0.001:
            break
        max_len = 120
        print(f"\n{i}. \"{inst['text'][:max_len]}...\"" if len(inst['text']) > max_len else f"\n{i}. \"{inst['text']}\"")
        print(f"   {model1_name}: {inst['model1_likert']}/5 ({inst['model1_score']:.2f})")
        print(f"   {model2_name}: {inst['model2_likert']}/5 ({inst['model2_score']:.2f})")
        print(f"   Difference: {inst['difference']:+.2f}")


if __name__ == "__main__":
    import json
    import sys
    
    # Load real results if available
    try:
        with open("results.json", "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from results.json")
    except FileNotFoundError:
        print("results.json not found, using mock data")
        results = [
            {"text": "Test 1", "model_openai/gpt-oss-20b": 0.5, "model_openai/gpt-oss-safeguard-20b": 0.75},
            {"text": "Test 2", "model_openai/gpt-oss-20b": 0.25, "model_openai/gpt-oss-safeguard-20b": 0.25},
            {"text": "Test 3", "model_openai/gpt-oss-20b": 1.0, "model_openai/gpt-oss-safeguard-20b": 0.5},
        ]
    
    # Run analysis
    analysis = compare_results(results)
    print_analysis(analysis)
    
    # Create and save figure
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)
    fig = plot_likert_distribution(analysis, output_path="likert_distribution_comparison.svg")
    plt.show()
