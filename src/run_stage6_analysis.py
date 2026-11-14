"""
Stage 6: Statistical analysis and interpretation of Stage 5 SAE interventions.

Input:
  results/stage5/stage5_results.csv

Outputs:
  results/stage6/
    stage6_overall_summary.csv
    stage6_prompt_family_summary.csv
    stage6_feature_summary.csv
    stage6_paired_prompt_comparison.csv
    stage6_data_quality.json
    stage6_report.md
    pass_rate_by_intervention.png
    pass_rate_by_prompt_family.png

Run:
  python src/run_stage6_analysis.py --input results/stage5/stage5_results.csv
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

REQUIRED = {
    "experiment_id", "prompt_sha256", "task_id", "prompt_family",
    "split", "intervention_type", "feature_group", "feature_id",
    "intervention_passed", "hook_calls", "max_residual_delta"
}

SELECTED_TYPES = {"ablation", "positive_steering", "negative_steering"}
RANDOM_TYPES = {
    "random_ablation", "random_positive_steering",
    "random_negative_steering"
}

def wilcoxon_safe(a, b):
    """Paired prompt-level Wilcoxon test; returns None when not testable."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2 or np.allclose(a, b):
        return {"n_prompts": int(len(a)), "statistic": 0.0, "p_value": 1.0}
    try:
        res = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {
            "n_prompts": int(len(a)),
            "statistic": float(res.statistic),
            "p_value": float(res.pvalue)
        }
    except Exception:
        return {"n_prompts": int(len(a)), "statistic": None, "p_value": None}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/stage5/stage5_results.csv")
    ap.add_argument("--output-dir", default="results/stage6")
    args = ap.parse_args()

    source = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    df = pd.read_csv(source)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Normalize analysis columns without changing the original input file.
    df["intervention_passed"] = pd.to_numeric(
        df["intervention_passed"], errors="coerce"
    )
    df["hook_calls"] = pd.to_numeric(df["hook_calls"], errors="coerce")
    df["max_residual_delta"] = pd.to_numeric(
        df["max_residual_delta"], errors="coerce"
    )
    df["feature_id"] = pd.to_numeric(df["feature_id"], errors="coerce")
    df["task_id"] = df["task_id"].astype(str)
    df["prompt_sha256"] = df["prompt_sha256"].astype(str)
    df["intervention_type"] = df["intervention_type"].astype(str)
    df["prompt_family"] = df["prompt_family"].astype(str)

    valid_outcomes = df["intervention_passed"].isin([0, 1])
    if not valid_outcomes.all():
        df = df.loc[valid_outcomes].copy()

    # Overall descriptive results.
    overall = (
        df.groupby("intervention_type", dropna=False)
        .agg(
            experiment_runs=("intervention_passed", "size"),
            passed=("intervention_passed", "sum"),
            pass_rate=("intervention_passed", "mean"),
            mean_execution_time_sec=("execution_time_sec", "mean"),
            mean_max_residual_delta=("max_residual_delta", "mean"),
            zero_delta_fraction=("max_residual_delta", lambda x: (x.fillna(0) == 0).mean()),
            mean_hook_calls=("hook_calls", "mean")
        )
        .reset_index()
    )
    overall["pass_rate_percent"] = overall["pass_rate"] * 100
    overall.to_csv(out / "stage6_overall_summary.csv", index=False)

    # Breakdown by prompt family and condition.
    family = (
        df.groupby(["prompt_family", "intervention_type"])
        .agg(
            experiment_runs=("intervention_passed", "size"),
            passed=("intervention_passed", "sum"),
            pass_rate=("intervention_passed", "mean")
        )
        .reset_index()
    )
    family["pass_rate_percent"] = family["pass_rate"] * 100
    family.to_csv(out / "stage6_prompt_family_summary.csv", index=False)

    # Feature-level outcomes for selected and random feature conditions.
    feature_df = df[df["feature_id"].notna()].copy()
    feature = (
        feature_df.groupby(
            ["feature_id", "feature_group", "intervention_type"]
        )
        .agg(
            experiment_runs=("intervention_passed", "size"),
            passed=("intervention_passed", "sum"),
            pass_rate=("intervention_passed", "mean"),
            mean_max_residual_delta=("max_residual_delta", "mean"),
            zero_delta_fraction=("max_residual_delta", lambda x: (x.fillna(0) == 0).mean())
        )
        .reset_index()
    )
    feature["pass_rate_percent"] = feature["pass_rate"] * 100
    feature.to_csv(out / "stage6_feature_summary.csv", index=False)

    # Prompt-level aggregation prevents treating repeated feature conditions
    # on the same prompt as independent observations in the paired test.
    prompt_type = (
        df.groupby(["prompt_sha256", "task_id", "prompt_family", "intervention_type"])
        .agg(prompt_pass_fraction=("intervention_passed", "mean"))
        .reset_index()
    )
    wide = prompt_type.pivot(
        index=["prompt_sha256", "task_id", "prompt_family"],
        columns="intervention_type",
        values="prompt_pass_fraction"
    ).reset_index()

    rows = []
    if "baseline" in wide.columns:
        baseline = wide["baseline"]
        for condition in sorted(set(wide.columns) - {
            "prompt_sha256", "task_id", "prompt_family", "baseline"
        }):
            valid = wide[["baseline", condition]].dropna()
            diff = valid[condition] - valid["baseline"]
            test = wilcoxon_safe(valid[condition], valid["baseline"])
            rows.append({
                "comparison": f"{condition} vs baseline",
                "n_paired_prompts": int(len(valid)),
                "baseline_mean_prompt_pass_fraction": float(valid["baseline"].mean()) if len(valid) else np.nan,
                "condition_mean_prompt_pass_fraction": float(valid[condition].mean()) if len(valid) else np.nan,
                "mean_paired_difference": float(diff.mean()) if len(diff) else np.nan,
                "median_paired_difference": float(diff.median()) if len(diff) else np.nan,
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_p_value": test["p_value"]
            })

        # Compare the average selected-feature intervention result with
        # average random-control result per prompt, where both exist.
        selected_cols = [c for c in SELECTED_TYPES if c in wide.columns]
        random_cols = [c for c in RANDOM_TYPES if c in wide.columns]
        if selected_cols and random_cols:
            selected_mean = wide[selected_cols].mean(axis=1, skipna=True)
            random_mean = wide[random_cols].mean(axis=1, skipna=True)
            valid = pd.DataFrame({
                "selected": selected_mean,
                "random": random_mean
            }).dropna()
            test = wilcoxon_safe(valid["selected"], valid["random"])
            diff = valid["selected"] - valid["random"]
            rows.append({
                "comparison": "selected-feature interventions vs random controls",
                "n_paired_prompts": int(len(valid)),
                "baseline_mean_prompt_pass_fraction": np.nan,
                "condition_mean_prompt_pass_fraction": np.nan,
                "mean_paired_difference": float(diff.mean()) if len(diff) else np.nan,
                "median_paired_difference": float(diff.median()) if len(diff) else np.nan,
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_p_value": test["p_value"]
            })

    paired = pd.DataFrame(rows)
    paired.to_csv(out / "stage6_paired_prompt_comparison.csv", index=False)

    # Data quality / provenance diagnostics.
    baseline_rows = df[df["intervention_type"] == "baseline"]
    noop_rows = df[df["intervention_type"] == "noop_control"]
    selected_ids = sorted(
        feature_df.loc[feature_df["feature_group"].isin(
            ["success_associated", "failure_associated"]
        ), "feature_id"].dropna().astype(int).unique().tolist()
    )
    random_ids = sorted(
        feature_df.loc[feature_df["feature_group"] == "random",
                       "feature_id"].dropna().astype(int).unique().tolist()
    )
    quality = {
        "source_file": str(source),
        "total_rows": int(len(df)),
        "unique_experiment_ids": int(df["experiment_id"].nunique()),
        "duplicate_experiment_id_rows": int(df["experiment_id"].duplicated().sum()),
        "unique_prompt_instances": int(df["prompt_sha256"].nunique()),
        "unique_task_ids": int(df["task_id"].nunique()),
        "prompt_families": df["prompt_family"].value_counts().to_dict(),
        "splits": df["split"].value_counts(dropna=False).to_dict(),
        "intervention_type_counts": df["intervention_type"].value_counts().to_dict(),
        "selected_feature_ids_observed": selected_ids,
        "random_feature_ids_observed": random_ids,
        "baseline_rows": int(len(baseline_rows)),
        "baseline_nonzero_hook_call_rows": int((baseline_rows["hook_calls"].fillna(0) > 0).sum()),
        "noop_rows": int(len(noop_rows)),
        "noop_nonzero_hook_call_rows": int((noop_rows["hook_calls"].fillna(0) > 0).sum()),
        "noop_nonzero_residual_delta_rows": int((noop_rows["max_residual_delta"].fillna(0) != 0).sum()),
        "missing_outcome_rows_removed": int((~valid_outcomes).sum()),
        "feature_group_counts": df["feature_group"].value_counts(dropna=False).to_dict()
    }
    (out / "stage6_data_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )

    # Charts.
    plot_df = overall.sort_values("pass_rate", ascending=False)
    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["intervention_type"], plot_df["pass_rate_percent"])
    plt.ylabel("Pass rate (%)")
    plt.xlabel("Intervention condition")
    plt.title("MBPP pass rate by intervention condition")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out / "pass_rate_by_intervention.png", dpi=180)
    plt.close()

    pivot_family = family.pivot(
        index="intervention_type", columns="prompt_family",
        values="pass_rate_percent"
    )
    ax = pivot_family.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Pass rate (%)")
    ax.set_xlabel("Intervention condition")
    ax.set_title("Pass rate by intervention condition and prompt family")
    ax.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    plt.savefig(out / "pass_rate_by_prompt_family.png", dpi=180)
    plt.close()

    # Concise report generated from the actual uploaded data.
    report_lines = [
        "# Stage 6 — Results Analysis and Interpretation",
        "",
        "## Dataset",
        f"- Valid experiment records analyzed: **{len(df)}**",
        f"- Unique validation prompt instances: **{df['prompt_sha256'].nunique()}**",
        f"- Unique MBPP task IDs: **{df['task_id'].nunique()}**",
        f"- Duplicate experiment IDs: **{df['experiment_id'].duplicated().sum()}**",
        f"- Selected feature IDs observed: **{len(selected_ids)}**",
        f"- Random feature IDs observed: **{len(random_ids)}**",
        "",
        "The file contains multiple feature configurations / partial runs. "
        "Results below describe the uploaded CSV as a whole; they must not be "
        "described as exclusively one top-k setting.",
        "",
        "## Overall outcomes",
        "",
        "| Condition | Runs | Passed | Pass rate | Mean residual delta |",
        "|---|---:|---:|---:|---:|"
    ]
    for _, r in overall.iterrows():
        report_lines.append(
            f"| {r['intervention_type']} | {int(r['experiment_runs'])} | "
            f"{int(r['passed'])} | {r['pass_rate']*100:.2f}% | "
            f"{r['mean_max_residual_delta']:.6f} |"
        )
    report_lines += [
        "",
        "## Prompt-family results",
        "",
        "| Prompt family | Condition | Runs | Pass rate |",
        "|---|---|---:|---:|"
    ]
    for _, r in family.iterrows():
        report_lines.append(
            f"| {r['prompt_family']} | {r['intervention_type']} | "
            f"{int(r['experiment_runs'])} | {r['pass_rate']*100:.2f}% |"
        )
    report_lines += [
        "",
        "## Paired prompt-level comparisons",
        "",
        "The paired analysis first averages repeated feature-condition outcomes "
        "within each prompt. This avoids treating every feature run on the same "
        "prompt as an independent prompt. Wilcoxon p-values are exploratory.",
        "",
        "| Comparison | Paired prompts | Mean paired difference | Wilcoxon p-value |",
        "|---|---:|---:|---:|"
    ]
    for _, r in paired.iterrows():
        pval = r["wilcoxon_p_value"]
        ptxt = "NA" if pd.isna(pval) else f"{pval:.6g}"
        diff = r["mean_paired_difference"]
        dtxt = "NA" if pd.isna(diff) else f"{diff:.4f}"
        report_lines.append(
            f"| {r['comparison']} | {int(r['n_paired_prompts'])} | "
            f"{dtxt} | {ptxt} |"
        )
    report_lines += [
        "",
        "## Data-quality observations",
        f"- Baseline rows with nonzero hook calls: **{quality['baseline_nonzero_hook_call_rows']}**.",
        f"- No-op rows with nonzero hook calls: **{quality['noop_nonzero_hook_call_rows']}**.",
        f"- No-op rows with nonzero residual delta: **{quality['noop_nonzero_residual_delta_rows']}**.",
        "",
        "Baseline rows with hook calls indicate that some baseline records came "
        "from an earlier run/configuration. They are retained as recorded, but "
        "this inconsistency must be disclosed.",
        "",
        "## Interpretation and limitations",
        "- Pass rate is the proportion of generated solutions that passed the recorded MBPP tests.",
        "- A zero residual delta means that the recorded intervention did not change the residual activation for that run; it should not be interpreted as evidence that the feature has no effect in every context.",
        "- Mixed feature counts and partial runs mean the dataset is not a balanced, single top-k experiment.",
        "- Repeated feature interventions share prompts; run-level counts are descriptive and are not independent observations.",
        "- The validation split was used for this intervention analysis. The test split was not included in this file and remains reserved for held-out evaluation.",
        "- Statistical associations in this analysis do not by themselves establish a general causal mechanism. Interpret results in light of actual residual changes and experimental coverage.",
        "",
        "## Files",
        "- `stage6_overall_summary.csv`: aggregate condition results.",
        "- `stage6_prompt_family_summary.csv`: canonical / paraphrase / OOD breakdown.",
        "- `stage6_feature_summary.csv`: individual feature outcomes and residual deltas.",
        "- `stage6_paired_prompt_comparison.csv`: paired prompt-level comparisons.",
        "- `stage6_data_quality.json`: dataset diagnostics.",
        "- `pass_rate_by_intervention.png` and `pass_rate_by_prompt_family.png`: figures.",
        ""
    ]
    (out / "stage6_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(f"Stage 6 analysis complete. Outputs: {out.resolve()}")
    print(f"Rows analyzed: {len(df)}")
    print(f"Prompts: {df['prompt_sha256'].nunique()}")
    print(f"Tasks: {df['task_id'].nunique()}")
    print(overall[["intervention_type", "experiment_runs", "passed", "pass_rate_percent"]].to_string(index=False))

if __name__ == "__main__":
    main()
