"""
Phase 3 - Stage 4: Baseline Generation and Evaluation (v4).

Changes from v3:
- Prompt now contains the MBPP tests + few-shot examples (fixes function-name NameErrors)
- Generation stops at the closing ``` fence (no more 256 tokens of repetition)
- bf16 + eager attention by default (use --quant4 to fall back to 4-bit)
- inputs=None bug fixed in the OOM handler
- Debug printing behind --verbose

Run:
    python src/run_stage4_baseline_eval.py --limit 20 --verbose
    python src/run_stage4_baseline_eval.py --batch-size 8
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

ROOT = Path(__file__).resolve().parents[1]

PROMPT_INSTANCES_CSV = ROOT / "data" / "processed" / "mbpp_prompt_instances.csv"
MBPP_TASKS_CSV = ROOT / "data" / "processed" / "mbpp_300_tasks.csv"
OUTPUT_CSV = ROOT / "results" / "baseline_train_eval.csv"
WORKER = Path(__file__).resolve().parent / "sandbox_worker.py"

MODEL_ID = "google/gemma-2-2b"

FIELDS = [
    "prompt_sha256",
    "task_id",
    "prompt_family",
    "split",
    "generated_code",
    "passed",
    "error_message",
    "execution_time_sec",
]

# --------------------------------------------------
# PROMPT BUILDING
# --------------------------------------------------

# Standard MBPP few-shot examples (tasks 2, 3, 4). Make sure these task ids
# are NOT in the set you evaluate on.
FEWSHOT = '''You are an expert Python programmer. Your task is: Write a function to find the similar elements from the given two tuple lists.
Your code should pass these tests:
assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)
```python
def similar_elements(test_tup1, test_tup2):
    return tuple(sorted(set(test_tup1) & set(test_tup2)))
```

You are an expert Python programmer. Your task is: Write a python function to identify non-prime numbers.
Your code should pass these tests:
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
```python
import math
def is_not_prime(n):
    if n < 2:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return True
    return False
```

You are an expert Python programmer. Your task is: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
Your code should pass these tests:
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
```python
import heapq
def heap_queue_largest(nums, n):
    return heapq.nlargest(n, nums)
```

'''

_WRAPPER_HEAD = re.compile(
    r"^\s*Write a Python solution for the following problem:\s*", re.IGNORECASE
)
_WRAPPER_TAIL = re.compile(
    r"\s*Return only executable Python code\.?\s*$", re.IGNORECASE
)


def extract_description(prompt_text: str) -> str:
    """Strip the wrapper text from a stored prompt to recover the raw task description."""
    text = _WRAPPER_HEAD.sub("", prompt_text)
    text = _WRAPPER_TAIL.sub("", text)
    return text.strip()


def build_prompt(description: str, test_list: list[str]) -> str:
    tests = "\n".join(test_list)
    return (
        FEWSHOT
        + f"You are an expert Python programmer. Your task is: {description}\n"
        + f"Your code should pass these tests:\n{tests}\n```python\n"
    )


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

def load_tasks(path=MBPP_TASKS_CSV):
    tasks = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tasks[str(row["task_id"])] = {
                "reference_solution": row["reference_solution"],
                "test_imports": row.get("test_imports", "") or "",
                "test_list": ast.literal_eval(row["test_list"]),
            }
    return tasks


def load_train_instances(path=PROMPT_INSTANCES_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["split"] == "train"]


# --------------------------------------------------
# RESUME SUPPORT
# --------------------------------------------------

def already_done_ids(csv_path=OUTPUT_CSV):
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {
            row["prompt_sha256"]
            for row in csv.DictReader(f)
            if row.get("prompt_sha256")
        }


def append_eval_rows(rows, csv_path=OUTPUT_CSV):
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()


# --------------------------------------------------
# CODE EXTRACTION
# --------------------------------------------------

def extract_code_block(raw):
    text = raw.strip()
    if not text:
        return ""

    # The prompt already ends with ```python\n, so the model writes code first
    # and closes with ```. Take everything before the first closing fence.
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        text = "\n".join(lines)

    text = text.split("```", 1)[0].strip()

    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return ""


# --------------------------------------------------
# CODE EXECUTION
# --------------------------------------------------

def execute_with_timeout(code, test_imports, test_list, timeout_sec=3.0):
    start_time = time.time()

    if not code.strip():
        return False, "NoCodeGenerated: model did not generate valid Python code", 0.0

    payload = {
        "code": code,
        "test_imports": test_imports,
        "test_list": test_list,
    }

    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_sec + 2.0,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return False, f"TimeoutError: worker exceeded {timeout_sec + 2.0}s", round(elapsed, 3)
    except Exception as e:
        elapsed = time.time() - start_time
        return False, f"SubprocessError: {type(e).__name__}: {e}", round(elapsed, 3)

    elapsed = time.time() - start_time
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    lines = [line for line in stdout.splitlines() if line.strip()]

    if not lines:
        return (
            False,
            f"WorkerNoOutput | returncode={proc.returncode} | stderr={stderr[:1000]!r}",
            round(elapsed, 3),
        )

    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError:
        return (
            False,
            (
                f"WorkerParseError | returncode={proc.returncode} | "
                f"stdout={stdout[-1000:]!r} | stderr={stderr[-1000:]!r}"
            ),
            round(elapsed, 3),
        )

    if proc.returncode != 0:
        return (
            False,
            (
                f"WorkerExitError | returncode={proc.returncode} | "
                f"worker_error={result.get('error', '')} | stderr={stderr[:500]!r}"
            ),
            round(elapsed, 3),
        )

    return bool(result.get("passed", False)), result.get("error", ""), round(elapsed, 3)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

class StopOnFence(StoppingCriteria):
    """Stop each sequence once it has produced a closing ``` fence."""

    def __init__(self, tokenizer, start_len):
        self.tokenizer = tokenizer
        self.start_len = start_len

    def __call__(self, input_ids, scores, **kwargs):
        # Only decode the last few tokens; the fence is 1-3 tokens.
        tails = self.tokenizer.batch_decode(
            input_ids[:, max(self.start_len, input_ids.shape[1] - 4):],
            skip_special_tokens=True,
        )
        return torch.tensor(
            ["```" in t for t in tails], device=input_ids.device
        )


class PipelineRunner:
    def __init__(self, model_id=MODEL_ID, quant4=False):
        print(f"Loading model: {model_id} ({'4-bit' if quant4 else 'bf16'})")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if quant4:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                attn_implementation="eager",
                device_map={"": 0},
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map={"": 0},
            )

        self.model.eval()
        print("Model loaded successfully.")

    @torch.inference_mode()
    def generate_batch(self, prompts, max_new_tokens=256):
        inputs = None
        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to("cuda")

            start = inputs["input_ids"].shape[1]
            torch.cuda.synchronize()
            start_time = time.time()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                stopping_criteria=StoppingCriteriaList(
                    [StopOnFence(self.tokenizer, start)]
                ),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            torch.cuda.synchronize()
            generation_time = time.time() - start_time

            generated = outputs[:, start:]
            new_tokens = generated.shape[1]
            print(
                f"Generation time: {generation_time:.2f}s | "
                f"Generated tokens (longest): {new_tokens} | "
                f"Speed: {new_tokens / max(generation_time, 0.001):.2f} tokens/sec"
            )

            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        except torch.cuda.OutOfMemoryError:
            inputs = None
            gc.collect()
            torch.cuda.empty_cache()

            if len(prompts) == 1:
                raise

            print(f"CUDA OOM with batch size {len(prompts)}. Splitting batch...")
            middle = len(prompts) // 2
            first = self.generate_batch(prompts[:middle], max_new_tokens)
            second = self.generate_batch(prompts[middle:], max_new_tokens)
            return first + second


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--test-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit prompts for a small test run. 0 means all.")
    parser.add_argument("--quant4", action="store_true",
                        help="Load the model in 4-bit instead of bf16.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompts, raw generations and extracted code.")
    args = parser.parse_args()

    if args.batch_size < 1 or args.test_workers < 1:
        parser.error("Batch size and test workers must be positive.")

    if not WORKER.exists():
        raise FileNotFoundError(f"Worker not found: {WORKER}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    tasks = load_tasks()
    train_instances = load_train_instances()
    done = already_done_ids()

    todo = [r for r in train_instances if r["prompt_sha256"] not in done]

    # Build the final prompt for every row (description + tests + few-shot).
    for row in todo:
        task = tasks.get(str(row["task_id"]))
        if task is None:
            row["final_prompt"] = None
            continue
        desc = extract_description(row["prompt"])
        row["final_prompt"] = build_prompt(desc, task["test_list"])

    # Group similar-length prompts to reduce padding.
    todo.sort(key=lambda r: len(r["final_prompt"] or ""))

    if args.limit > 0:
        todo = todo[: args.limit]

    print("-------------------------------------")
    print(f"Training instances: {len(train_instances)}")
    print(f"Previously evaluated: {len(done)}")
    print(f"Remaining: {len(todo)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Test workers: {args.test_workers}")
    print("-------------------------------------")

    if not todo:
        print("No prompts remaining.")
        return

    runner = PipelineRunner(quant4=args.quant4)

    pass_count = 0
    n_done = 0
    start_all = time.time()

    with ThreadPoolExecutor(max_workers=args.test_workers) as pool:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b : b + args.batch_size]

            # ---------------- GENERATION ----------------
            generation_start = time.time()

            gen_idx = [i for i, r in enumerate(batch) if r["final_prompt"]]
            raw_generations = [""] * len(batch)

            if gen_idx:
                try:
                    outs = runner.generate_batch(
                        [batch[i]["final_prompt"] for i in gen_idx],
                        args.max_new_tokens,
                    )
                    for i, text in zip(gen_idx, outs):
                        raw_generations[i] = text
                except Exception as e:
                    print(f"Generation failed for batch {b}: {type(e).__name__}: {e}")

            generation_time = time.time() - generation_start
            codes = [extract_code_block(t) for t in raw_generations]

            if args.verbose:
                for row, raw, code in zip(batch, raw_generations, codes):
                    print("\nTASK ID:", row["task_id"])
                    print("RAW GENERATION:", repr(raw))
                    print("EXTRACTED CODE:\n" + code)

            # ---------------- EXECUTION ----------------
            futures = []
            for row, code in zip(batch, codes):
                task = tasks.get(str(row["task_id"]))
                if task is None or not code:
                    futures.append(None)
                    continue
                futures.append(
                    pool.submit(
                        execute_with_timeout,
                        code,
                        task["test_imports"],
                        task["test_list"],
                        3.0,
                    )
                )

            # ---------------- COLLECT ----------------
            rows = []
            for row, code, future in zip(batch, codes, futures):
                if str(row["task_id"]) not in tasks:
                    passed, error, execution_time = False, "DatasetError: task_id not found", 0.0
                elif not code:
                    passed, error, execution_time = (
                        False,
                        "NoCodeGenerated: no valid Python code extracted",
                        0.0,
                    )
                else:
                    passed, error, execution_time = future.result()

                pass_count += int(passed)

                rows.append({
                    "prompt_sha256": row["prompt_sha256"],
                    "task_id": row["task_id"],
                    "prompt_family": row["prompt_family"],
                    "split": row["split"],
                    "generated_code": code,
                    "passed": 1 if passed else 0,
                    "error_message": error,
                    "execution_time_sec": execution_time,
                })

            append_eval_rows(rows)
            n_done += len(batch)

            # ---------------- PROGRESS ----------------
            elapsed = time.time() - start_all
            rate = n_done / max(elapsed, 0.001)
            eta_min = (len(todo) - n_done) / max(rate, 0.001) / 60

            print(
                f"[{n_done}/{len(todo)}] "
                f"Generation={generation_time:.2f}s | "
                f"Accuracy={pass_count / n_done:.2%} | "
                f"Speed={rate:.2f} prompts/sec | "
                f"ETA={eta_min:.1f} min"
            )

            if n_done % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    print("-------------------------------------")
    print("Evaluation completed.")
    print(f"Total prompts: {n_done}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {n_done - pass_count}")
    print(f"Accuracy: {pass_count / n_done:.2%}")
    print(f"Output: {OUTPUT_CSV}")
    print("-------------------------------------")


if __name__ == "__main__":
    main()