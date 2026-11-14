"""Validate Stage 1 files and task-level split integrity."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tasks_path = ROOT / "data/processed/mbpp_300_tasks.csv"
prompts_path = ROOT / "data/processed/mbpp_prompt_instances.csv"
splits_path = ROOT / "data/processed/task_split_manifest.csv"
manifest_path = ROOT / "results/dataset_manifest.json"

for path in (tasks_path, prompts_path, splits_path, manifest_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")

tasks = pd.read_csv(tasks_path)
prompts = pd.read_csv(prompts_path)
splits = pd.read_csv(splits_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

assert len(tasks) == manifest["sample_size_saved"], "Task count mismatch"
assert tasks["task_id"].is_unique, "Duplicate task IDs found"
assert len(prompts) == len(tasks) * 3, "Expected exactly 3 prompt families per task"
assert set(prompts["prompt_family"]) == {"canonical", "paraphrase", "ood"}
assert prompts.groupby("task_id")["split"].nunique().max() == 1, (
    "Leakage: one task appears in multiple splits"
)
assert set(tasks["task_id"]) == set(splits["task_id"])
assert tasks["test_list"].notna().all()
assert tasks["task_statement"].notna().all()

print("DATASET VALIDATION PASSED")
print(f"Unique tasks: {len(tasks)}")
print(f"Prompt instances: {len(prompts)}")
print("\nSplit counts:")
print(tasks["split"].value_counts().to_string())
print("\nPrompt family counts:")
print(prompts["prompt_family"].value_counts().to_string())
print("\nManifest:")
print(json.dumps(manifest, indent=2))
