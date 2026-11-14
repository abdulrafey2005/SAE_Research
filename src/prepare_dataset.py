"""Stage 1: download, sample, split, and create prompt-family records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_prompts(task_statement: str) -> dict[str, str]:
    # All three keep the task specification unchanged. The wrappers vary.
    # The paraphrase-style wrapper is not a human-validated paraphrase of the
    # task statement; document this limitation in the Phase 3 report.
    return {
        "canonical": (
            "Write a Python solution for the following problem:\n"
            f"{task_statement}\n"
            "Return only executable Python code."
        ),
        "paraphrase": (
            "Please implement a Python function that solves the task described here:\n"
            f"{task_statement}\n"
            "Provide only the Python code, without an explanation."
        ),
        "ood": (
            "You are completing an automated programming challenge. Produce a correct "
            "Python implementation that satisfies this specification. Do not include "
            "markdown or commentary.\n\n"
            f"Specification:\n{task_statement}\n\nPython implementation:"
        ),
    }


def normalize_row(row: dict) -> dict:
    statement = row.get("prompt") or row.get("text") or ""
    tests = row.get("test_list") or []
    imports = row.get("test_imports") or []
    return {
        "task_id": int(row["task_id"]),
        "task_statement": str(statement).strip(),
        "reference_solution": str(row.get("code") or ""),
        "test_imports": imports,
        "test_list": tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tasks", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for directory in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)

        
    print("Downloading all MBPP sanitized splits...")

    try:
        dataset = load_dataset(
            "google-research-datasets/mbpp", "sanitized"
        )
        dataset_id = "google-research-datasets/mbpp"

    except Exception as first_error:
        print(f"Primary dataset ID failed: {first_error}")
        print("Trying canonical datasets alias...")
        dataset = load_dataset("mbpp", "sanitized")
        dataset_id = "mbpp"

    rows = []

    # Combine all original dataset splits.
    for source_split, split_dataset in dataset.items():

        print(f"Loading {source_split}: {len(split_dataset)} records")

        for original_row in split_dataset:

            row = normalize_row(original_row)

            # Keep original dataset split for traceability.
            row["source_split"] = source_split

            rows.append(row)

    print("Total records loaded:", len(rows))

    rows = [
        row for row in rows
        if row["task_statement"] and row["reference_solution"] and row["test_list"]
    ]

    # Remove duplicate task IDs and duplicate task statements deterministically.
    unique = {}
    seen_statements = set()
    for row in rows:
        statement_key = row["task_statement"].casefold()
        if row["task_id"] not in unique and statement_key not in seen_statements:
            unique[row["task_id"]] = row
            seen_statements.add(statement_key)
    rows = list(unique.values())

    if args.n_tasks > len(rows):
        raise ValueError(
            f"Requested {args.n_tasks} tasks, but only {len(rows)} valid unique "
            "sanitized tasks were loaded. Reduce --n_tasks."
        )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    selected = [rows[i] for i in order[:args.n_tasks]]

    # Task-level split: every prompt family for a task inherits this split.
    n_train = int(args.n_tasks * 0.70)
    n_val = int(args.n_tasks * 0.15)
    n_test = args.n_tasks - n_train - n_val
    
    if n_test <= 0 and args.n_tasks >= 3:
        n_test = 1
        n_val = max(1, n_val - 1)

    split_names = (
        ["train"] * n_train
        + ["validation"] * n_val
        + ["test"] * n_test
    )
    split_order = rng.permutation(args.n_tasks)
    split_by_id = {
        selected[idx]["task_id"]: split_names[pos]
        for pos, idx in enumerate(split_order)
    }

    raw_path = RAW_DIR / "mbpp_sanitized_source.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_records = []
    prompt_records = []
    for row in tqdm(selected, desc="Creating task and prompt records"):
        split = split_by_id[row["task_id"]]
        prompt_map = make_prompts(row["task_statement"])
        task_records.append({
            **row,
            "split": split,
            "task_statement_sha256": sha256_text(row["task_statement"]),
        })
        for family, prompt in prompt_map.items():
            prompt_records.append({
                "task_id": row["task_id"],
                "split": split,
                "prompt_family": family,
                "prompt": prompt,
                "prompt_sha256": sha256_text(prompt),
                "task_statement_sha256": sha256_text(row["task_statement"]),
            })

    tasks_df = pd.DataFrame(task_records)
    prompts_df = pd.DataFrame(prompt_records)
    split_df = tasks_df[["task_id", "split", "task_statement_sha256"]].copy()

    tasks_df.to_csv(PROCESSED_DIR / "mbpp_300_tasks.csv", index=False)
    prompts_df.to_csv(PROCESSED_DIR / "mbpp_prompt_instances.csv", index=False)
    split_df.to_csv(PROCESSED_DIR / "task_split_manifest.csv", index=False)

    manifest = {
        "dataset_id_used": dataset_id,
        "dataset_config": "sanitized",
        "source_split": "all_sanitized_splits",
        "sample_size_requested": args.n_tasks,
        "sample_size_saved": len(tasks_df),
        "random_seed": args.seed,
        "split_counts": tasks_df["split"].value_counts().to_dict(),
        "prompt_instance_counts": prompts_df["prompt_family"].value_counts().to_dict(),
        "task_level_split": True,
        "prompt_family_note": (
            "Instruction-wrapper variants preserve the task statement. The "
            "paraphrase-style wrapper is not an independently human-validated "
            "semantic paraphrase; validate or replace it before final evaluation "
            "if strict paraphrase semantics are required."
        ),
        "files": [
            "data/raw/mbpp_sanitized_source.jsonl",
            "data/processed/mbpp_300_tasks.csv",
            "data/processed/mbpp_prompt_instances.csv",
            "data/processed/task_split_manifest.csv",
        ],
    }
    (RESULTS_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nStage 1 complete.")
    print(tasks_df["split"].value_counts().to_string())
    print(f"Tasks: {len(tasks_df)}")
    print(f"Prompt instances: {len(prompts_df)}")
    print(f"Manifest: {RESULTS_DIR / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
