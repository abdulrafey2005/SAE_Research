"""
Sanity-checks results/activation_stats.csv after a Stage 3 run against
data/processed/mbpp_prompt_instances.csv:
  - every prompt_sha256 in the instances file has a corresponding stats row
  - no zero-activation (degenerate) extractions
  - L0-norm distribution looks reasonable per prompt_family and per split

Usage:
    python src/verify_stage3_extraction.py
"""
import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT_INSTANCES_CSV = ROOT / "data" / "processed" / "mbpp_prompt_instances.csv"
STATS_CSV = ROOT / "results" / "activation_stats.csv"


def main():
    with open(PROMPT_INSTANCES_CSV, newline="", encoding="utf-8") as f:
        instance_ids = {row["prompt_sha256"] for row in csv.DictReader(f)}

    with open(STATS_CSV, newline="", encoding="utf-8") as f:
        stats_rows = list(csv.DictReader(f))
    stats_ids = {row["prompt_sha256"] for row in stats_rows}

    missing = instance_ids - stats_ids
    print(f"Prompt instances: {len(instance_ids)} | Extracted: {len(stats_ids)} | Missing: {len(missing)}")
    if missing:
        print(f"  (re-run extract_stage3_features.py — resume will only process these {len(missing)})")

    by_family = defaultdict(list)
    by_split = defaultdict(list)
    zero_l0 = []
    for row in stats_rows:
        l0 = int(row["l0_norm"])
        by_family[row["prompt_family"]].append(l0)
        by_split[row["split"]].append(l0)
        if l0 == 0:
            zero_l0.append(row["prompt_sha256"])

    def report(label, groups):
        print(f"\nL0 norm (active SAE features per prompt) by {label}:")
        for key, values in sorted(groups.items()):
            if not values:
                continue
            print(f"  {key:<12} n={len(values):<4} mean={st.mean(values):6.2f} "
                  f"median={st.median(values):6.1f} min={min(values)} max={max(values)}")

    report("prompt_family", by_family)
    report("split", by_split)

    if zero_l0:
        print(f"\nWARNING: {len(zero_l0)} prompts had zero active features "
              f"(check for a dtype mismatch between the model output and the SAE's expected input). "
              f"First few: {zero_l0[:5]}")
    else:
        print("\nNo zero-activation prompts found.")

    families_present = set(by_family.keys())
    expected_families = {"canonical", "paraphrase", "ood"}
    if families_present != expected_families:
        print(f"\nWARNING: expected prompt families {expected_families}, found {families_present}")

    splits_present = set(by_split.keys())
    expected_splits = {"train", "validation", "test"}
    if splits_present != expected_splits:
        print(f"\nWARNING: expected splits {expected_splits}, found {splits_present}")


if __name__ == "__main__":
    main()
