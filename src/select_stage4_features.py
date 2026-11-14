
"""
Stage 4: SAE Feature Selection

Uses:
- results/activation_stats.csv
- results/baseline_train_eval.csv
- activations/sae_latents/<prompt_sha256[:16]>.npy

Produces:
- results/stage4/feature_statistics.csv
- results/stage4/ranked_features.csv
- results/stage4/selected_features.json
- results/stage4/feature_selection_summary.json

Feature discovery uses TRAIN data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

ACTIVATION_STATS = ROOT / "results" / "activation_stats.csv"
BASELINE_CSV = ROOT / "results" / "baseline_train_eval.csv"

LATENT_DIR = ROOT / "activations" / "sae_latents"

OUTPUT_DIR = ROOT / "results" / "stage4"

N_FEATURES = 16384
TOP_K = 50

SPLIT = "train"


# --------------------------------------------------
# BENJAMINI-HOCHBERG FDR CORRECTION
# --------------------------------------------------

def benjamini_hochberg(p_values):

    p_values = np.asarray(p_values, dtype=np.float64)

    n = len(p_values)

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted = sorted_p * n / np.arange(1, n + 1)

    # Enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n)

    result[order] = adjusted

    return result


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():

    print("Loading activation statistics...")

    activations = pd.read_csv(ACTIVATION_STATS)

    print("Loading baseline evaluation...")

    baseline = pd.read_csv(BASELINE_CSV)

    required_activation = [
        "prompt_sha256",
        "task_id",
        "prompt_family",
        "split",
        "sae_latent_path"
    ]

    required_baseline = [
        "prompt_sha256",
        "task_id",
        "prompt_family",
        "split",
        "passed"
    ]

    for col in required_activation:
        if col not in activations.columns:
            raise ValueError(f"Missing activation column: {col}")

    for col in required_baseline:
        if col not in baseline.columns:
            raise ValueError(f"Missing baseline column: {col}")

    # Feature discovery uses training data only.
    activations = activations[
        activations["split"] == SPLIT
    ].copy()

    baseline = baseline[
        baseline["split"] == SPLIT
    ].copy()

    if activations["prompt_sha256"].duplicated().any():
        raise ValueError("Duplicate activation prompt hashes found.")

    if baseline["prompt_sha256"].duplicated().any():
        raise ValueError("Duplicate baseline prompt hashes found.")

    # Merge on unique prompt hash.
    merged = activations.merge(
        baseline,
        on="prompt_sha256",
        how="inner",
        suffixes=("_act", "_base"),
        validate="one_to_one"
    )

    if len(merged) != len(baseline):
        missing = len(baseline) - len(merged)

        raise ValueError(
            f"{missing} baseline prompts have no matching activation."
        )

    # Check task and prompt-family consistency.
    for col in ["task_id", "prompt_family", "split"]:
        left = f"{col}_act"
        right = f"{col}_base"

        if not (merged[left] == merged[right]).all():
            raise ValueError(
                f"Mismatch detected in column: {col}"
            )

    merged["passed"] = pd.to_numeric(
        merged["passed"], errors="raise"
    ).astype(int)

    if not merged["passed"].isin([0, 1]).all():
        raise ValueError("passed must contain only 0 and 1.")

    print(f"Matched prompts: {len(merged)}")

    print(
        "Successful:",
        int((merged["passed"] == 1).sum())
    )

    print(
        "Failed:",
        int((merged["passed"] == 0).sum())
    )

    return merged


# --------------------------------------------------
# LOAD SAE LATENT VECTORS
# --------------------------------------------------

def load_latents(merged):

    print("\nLoading SAE latent vectors...")

    vectors = []

    for i, row in enumerate(
        merged.itertuples(index=False), start=1
    ):

        # Use the stable prompt hash as the filename stem.
        prompt_hash = row.prompt_sha256

        latent_path = (
            LATENT_DIR / f"{prompt_hash[:16]}.npy"
        )

        if not latent_path.exists():
            raise FileNotFoundError(
                f"Missing latent vector: {latent_path}"
            )

        vector = np.load(
            latent_path,
            allow_pickle=False
        )

        if vector.shape != (N_FEATURES,):
            raise ValueError(
                f"Unexpected shape for {prompt_hash}: "
                f"{vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Non-finite activation in {latent_path}"
            )

        vectors.append(
            vector.astype(np.float32, copy=False)
        )

        if i % 100 == 0:
            print(f"Loaded {i}/{len(merged)}")

    X = np.stack(vectors, axis=0)

    print("Activation matrix:", X.shape)

    return X


# --------------------------------------------------
# FEATURE STATISTICS
# --------------------------------------------------

def calculate_statistics(X, y):

    print("\nCalculating feature statistics...")

    passed = X[y == 1].astype(np.float64)

    failed = X[y == 0].astype(np.float64)

    if len(passed) < 2 or len(failed) < 2:
        raise ValueError(
            "At least two successful and two failed "
            "prompts are required."
        )

    # Mean activations
    mean_pass = passed.mean(axis=0)

    mean_fail = failed.mean(axis=0)

    # Differential activation
    difference = mean_pass - mean_fail

    # Welch's independent-samples t-test
    t_result = ttest_ind(
        passed,
        failed,
        axis=0,
        equal_var=False,
        nan_policy="raise"
    )

    t_stat = np.nan_to_num(
        t_result.statistic,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    p_values = np.nan_to_num(
        t_result.pvalue,
        nan=1.0,
        posinf=1.0,
        neginf=1.0
    )

    # AUROC using the Mann-Whitney rank formula.
    # Positive class = passed prompts.
    ranks = rankdata(X, axis=0)

    n_pass = len(passed)
    n_fail = len(failed)

    positive_rank_sum = ranks[y == 1].sum(axis=0)

    auc = (
        positive_rank_sum
        - n_pass * (n_pass + 1) / 2
    ) / (n_pass * n_fail)

    # Correct for any numerical rounding.
    auc = np.clip(auc, 0.0, 1.0)

    # Multiple-testing correction
    q_values = benjamini_hochberg(p_values)

    statistics = pd.DataFrame({

        "feature_id": np.arange(N_FEATURES),

        "mean_activation_pass": mean_pass,

        "mean_activation_fail": mean_fail,

        "differential_activation": difference,

        "t_statistic": t_stat,

        "p_value": p_values,

        "adjusted_p_value": q_values,

        "auroc": auc,

        "auroc_distance": np.abs(auc - 0.5),

        "direction": np.where(
            difference > 0,
            "success_associated",
            np.where(
                difference < 0,
                "failure_associated",
                "neutral"
            )
        )
    })

    return statistics


# --------------------------------------------------
# RANK FEATURES
# --------------------------------------------------

def select_features(statistics):

    print("\nRanking SAE features...")

    ranked = statistics.sort_values(
        by=["auroc_distance", "adjusted_p_value"],
        ascending=[False, True]
    ).reset_index(drop=True)

    ranked["rank"] = np.arange(1, len(ranked) + 1)

    success_features = (
        statistics[
            (statistics["auroc"] > 0.5)
            & (statistics["differential_activation"] > 0)
        ]
        .sort_values(
            ["auroc_distance", "adjusted_p_value"],
            ascending=[False, True]
        )
        .head(TOP_K)
    )

    failure_features = (
        statistics[
            (statistics["auroc"] < 0.5)
            & (statistics["differential_activation"] < 0)
        ]
        .sort_values(
            ["auroc_distance", "adjusted_p_value"],
            ascending=[False, True]
        )
        .head(TOP_K)
    )

    return ranked, success_features, failure_features


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    merged = load_data()

    X = load_latents(merged)

    y = merged["passed"].to_numpy()

    statistics = calculate_statistics(X, y)

    ranked, success_features, failure_features = (
        select_features(statistics)
    )

    # Save full statistics
    statistics.to_csv(
        OUTPUT_DIR / "feature_statistics.csv",
        index=False
    )

    ranked.to_csv(
        OUTPUT_DIR / "ranked_features.csv",
        index=False
    )

    selected = {

        "success_associated": (
            success_features.to_dict(orient="records")
        ),

        "failure_associated": (
            failure_features.to_dict(orient="records")
        ),

        "selection_method": (
            "AUROC distance from 0.5, "
            "then adjusted p-value"
        ),

        "top_k_per_direction": TOP_K,

        "discovery_split": SPLIT,

        "n_prompts": len(merged),

        "n_success": int((y == 1).sum()),

        "n_failure": int((y == 0).sum())
    }

    with open(
        OUTPUT_DIR / "selected_features.json",
        "w"
    ) as f:

        json.dump(
            selected,
            f,
            indent=4
        )

    summary = {

        "total_prompts": len(merged),

        "successful_prompts": int((y == 1).sum()),

        "failed_prompts": int((y == 0).sum()),

        "total_features": N_FEATURES,

        "significant_features_fdr_0.05": int(
            (statistics["adjusted_p_value"] < 0.05).sum()
        ),

        "success_associated_selected": len(success_features),

        "failure_associated_selected": len(failure_features),

        "mean_baseline_pass_rate": float(y.mean()),

        "discovery_split": SPLIT
    }

    with open(
        OUTPUT_DIR / "feature_selection_summary.json",
        "w"
    ) as f:

        json.dump(summary, f, indent=4)

    print("\nFEATURE SELECTION COMPLETED")

    print(json.dumps(summary, indent=4))

    print("\nTop success-associated features:")

    print(
        success_features[
            [
                "feature_id",
                "auroc",
                "t_statistic",
                "differential_activation",
                "adjusted_p_value"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nTop failure-associated features:")

    print(
        failure_features[
            [
                "feature_id",
                "auroc",
                "t_statistic",
                "differential_activation",
                "adjusted_p_value"
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
