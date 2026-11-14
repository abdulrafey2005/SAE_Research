
"""
Stage 5: SAE causal intervention experiments.

Experiments:
1. Baseline
2. Selected feature ablation
3. Selected feature positive steering
4. Selected feature negative steering
5. Random feature ablation
6. Random feature positive/negative steering
7. No-op control

Default split: validation.

The test split is reserved for final held-out evaluation.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import random
import sys
import time

from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList
)

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage4_baseline_eval as stage4

from stage5_intervention_engine import SAEIntervention


PROMPTS_CSV = ROOT / "data/processed/mbpp_prompt_instances.csv"
TASKS_CSV = ROOT / "data/processed/mbpp_300_tasks.csv"

FEATURES_JSON = ROOT / "results/stage4/selected_features.json"

OUTPUT_DIR = ROOT / "results/stage5"

MODEL_ID = "google/gemma-2-2b"

RESULT_FIELDS = [
    "experiment_id",
    "prompt_sha256",
    "task_id",
    "prompt_family",
    "split",
    "intervention_type",
    "feature_group",
    "feature_id",
    "steering_strength",
    "baseline_passed",
    "intervention_passed",
    "generated_code",
    "error_message",
    "execution_time_sec",
    "hook_calls",
    "max_residual_delta",
    "random_seed"
]


def read_csv(path):

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_rows(path, rows):

    if not rows:
        return

    write_header = (
        not path.exists()
        or path.stat().st_size == 0
    )

    with open(path, "a", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=RESULT_FIELDS,
            extrasaction="ignore"
        )

        if write_header:
            writer.writeheader()

        writer.writerows(rows)
        f.flush()


def load_tasks():

    tasks = {}

    for row in read_csv(TASKS_CSV):

        tasks[str(row["task_id"])] = {
            "test_imports": row.get("test_imports", "") or "",
            "test_list": ast.literal_eval(row["test_list"])
        }

    return tasks


def load_prompts(split):

    return [
        row for row in read_csv(PROMPTS_CSV)
        if row["split"] == split
    ]


def load_selected_features(top_k):

    with open(FEATURES_JSON, encoding="utf-8") as f:
        data = json.load(f)

    success = data["success_associated"][:top_k]
    failure = data["failure_associated"][:top_k]

    selected = []

    seen = set()

    for group, records in [
        ("success_associated", success),
        ("failure_associated", failure)
    ]:

        for record in records:

            fid = int(record["feature_id"])

            if fid not in seen:

                selected.append({
                    "feature_id": fid,
                    "feature_group": group
                })

                seen.add(fid)

    if not selected:
        raise RuntimeError("No selected features found.")

    return selected


class Stage5Runner:

    def __init__(self, quant4):

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            padding_side="left"
        )

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading Gemma-2-2B...")

        if quant4:

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                attn_implementation="eager",
                device_map={"": 0}
            )

        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map={"": 0}
            )

        self.model.eval()

        print("Loading GemmaScope SAE...")

        self.intervention = SAEIntervention(
            self.model,
            layer_index=12
        )

        print("Model and SAE loaded successfully.")

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens):

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to("cuda")

        start_length = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            stopping_criteria=StoppingCriteriaList([
                stage4.StopOnFence(
                    self.tokenizer,
                    start_length
                )
            ]),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        generated = outputs[:, start_length:]

        raw = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True
        )

        return stage4.extract_code_block(raw)

    def close(self):

        self.intervention.close()


def experiment_id(prompt_hash, kind, feature_id, alpha):

    value = (
        f"{prompt_hash}|{kind}|{feature_id}|{alpha}"
    )

    return hashlib.sha256(
        value.encode()
    ).hexdigest()[:20]


def run_experiment(
    runner,
    row,
    task,
    prompt,
    kind,
    feature_id,
    feature_group,
    alpha,
    baseline_passed,
    random_seed,
    max_new_tokens
):

    modes = {
        # Baseline and control
        "baseline": "disabled",
        "noop_control": "noop",

        # Selected feature interventions
        "ablation": "ablation",
        "positive_steering": "steer_positive",
        "negative_steering": "steer_negative",

        # Random feature controls
        "random_ablation": "ablation",
        "random_positive_steering": "steer_positive",
        "random_negative_steering": "steer_negative",
    }

    mode = modes[kind]

    runner.intervention.configure(
        mode=mode,
        feature_id=feature_id,
        alpha=alpha
    )

    start = time.time()

    code = runner.generate(
        prompt,
        max_new_tokens
    )

    passed, error, execution_time = stage4.execute_with_timeout(
        code,
        task["test_imports"],
        task["test_list"],
        3.0
    )

    if kind == "baseline":
        baseline_passed = int(passed)

    return {
        "experiment_id": experiment_id(
            row["prompt_sha256"],
            kind,
            feature_id,
            alpha
        ),

        "prompt_sha256": row["prompt_sha256"],
        "task_id": row["task_id"],
        "prompt_family": row["prompt_family"],
        "split": row["split"],

        "intervention_type": kind,
        "feature_group": feature_group,

        "feature_id": (
            "" if feature_id is None else feature_id
        ),

        "steering_strength": alpha,

        "baseline_passed": (
            "" if baseline_passed is None else baseline_passed
        ),

        "intervention_passed": int(passed),
        "generated_code": code,
        "error_message": error,
        "execution_time_sec": execution_time,

        "hook_calls": runner.intervention.hook_calls,
        "max_residual_delta": runner.intervention.last_max_delta,

        "random_seed": random_seed
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation"]
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0
    )

    parser.add_argument(
        "--top-k-per-direction",
        type=int,
        default=1
    )

    parser.add_argument(
        "--steer-alpha",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--quant4",
        action="store_true"
    )

    args = parser.parse_args()

    if args.top_k_per_direction < 1:
        parser.error("--top-k-per-direction must be positive.")

    if args.steer_alpha < 0:
        parser.error("--steer-alpha cannot be negative.")

    if args.split == "train":

        print(
            "WARNING: Feature discovery used train data. "
            "Train intervention results are exploratory."
        )

    else:

        print(
            "Validation split selected. "
            "Test split remains untouched."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prompts = load_prompts(args.split)

    if args.limit > 0:
        prompts = prompts[:args.limit]

    tasks = load_tasks()

    selected = load_selected_features(
        args.top_k_per_direction
    )

    selected_ids = [
        item["feature_id"]
        for item in selected
    ]

    rng = random.Random(args.random_seed)

    available_ids = [
        i for i in range(16384)
        if i not in set(selected_ids)
    ]

    random_ids = rng.sample(
        available_ids,
        len(selected_ids)
    )

    print("\nSelected features:")
    print(selected)

    print("\nRandom control features:")
    print(random_ids)

    all_results_path = (
        OUTPUT_DIR / "stage5_results.csv"
    )

    existing = set()

    if all_results_path.exists():

        existing = {
            r["experiment_id"]
            for r in read_csv(all_results_path)
        }

    runner = Stage5Runner(
        quant4=args.quant4
    )

    try:

        for index, row in enumerate(prompts, start=1):

            task = tasks.get(
                str(row["task_id"])
            )

            if task is None:
                print("Unknown task:", row["task_id"])
                continue

            description = stage4.extract_description(
                row["prompt"]
            )

            final_prompt = stage4.build_prompt(
                description,
                task["test_list"]
            )

            # ---------------- BASELINE ----------------

            baseline_key = experiment_id(
                row["prompt_sha256"],
                "baseline",
                None,
                0.0
            )

            baseline_passed = None

            if baseline_key not in existing:

                baseline_record = run_experiment(
                    runner,
                    row,
                    task,
                    final_prompt,
                    "baseline",
                    None,
                    "none",
                    0.0,
                    None,
                    args.random_seed,
                    args.max_new_tokens
                )

                append_rows(
                    all_results_path,
                    [baseline_record]
                )

                existing.add(
                    baseline_record["experiment_id"]
                )

                baseline_passed = (
                    baseline_record["intervention_passed"]
                )

            else:

                for old in read_csv(all_results_path):

                    if old["experiment_id"] == baseline_key:

                        baseline_passed = int(
                            old["intervention_passed"]
                        )

                        break

            # ---------------- EXPERIMENTS ----------------

            conditions = [
                ("noop_control", None, "none", 0.0)
            ]

            # Selected features, one at a time.
            for item in selected:

                fid = item["feature_id"]
                group = item["feature_group"]

                conditions.extend([
                    ("ablation", fid, group, 0.0),

                    (
                        "positive_steering",
                        fid,
                        group,
                        args.steer_alpha
                    ),

                    (
                        "negative_steering",
                        fid,
                        group,
                        args.steer_alpha
                    )
                ])

            # Random controls, one feature at a time.
            for fid in random_ids:

                conditions.extend([
                    ("random_ablation", fid, "random", 0.0),

                    (
                        "random_positive_steering",
                        fid,
                        "random",
                        args.steer_alpha
                    ),

                    (
                        "random_negative_steering",
                        fid,
                        "random",
                        args.steer_alpha
                    )
                ])

            new_records = []

            for kind, fid, group, alpha in conditions:

                key = experiment_id(
                    row["prompt_sha256"],
                    kind,
                    fid,
                    alpha
                )

                if key in existing:
                    continue

                record = run_experiment(
                    runner,
                    row,
                    task,
                    final_prompt,
                    kind,
                    fid,
                    group,
                    alpha,
                    baseline_passed,
                    args.random_seed,
                    args.max_new_tokens
                )

                new_records.append(record)
                existing.add(record["experiment_id"])

            append_rows(
                all_results_path,
                new_records
            )

            print(
                f"[{index}/{len(prompts)}] "
                f"Task={row['task_id']} "
                f"New experiments={len(new_records)}"
            )

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:

        runner.close()

    # ---------------- EXPORT CSV FILES ----------------

    all_rows = read_csv(all_results_path)

    exports = {
        "baseline_results.csv": {
            "baseline"
        },

        "ablation_results.csv": {
            "ablation"
        },

        "steering_results.csv": {
            "positive_steering",
            "negative_steering"
        },

        "random_control_results.csv": {
            "random_ablation",
            "random_positive_steering",
            "random_negative_steering"
        },

        "noop_control_results.csv": {
            "noop_control"
        }
    }

    for filename, kinds in exports.items():

        selected_rows = [
            r for r in all_rows
            if r["intervention_type"] in kinds
        ]

        path = OUTPUT_DIR / filename

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=RESULT_FIELDS
            )

            writer.writeheader()
            writer.writerows(selected_rows)

    print("\nStage 5 completed.")
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
