"""
Phase 3 - Stage 3: Batch Feature Activation Extraction.

Reads data/processed/mbpp_prompt_instances.csv (written by prepare_dataset.py)
directly — columns: task_id, split, prompt_family, prompt, prompt_sha256,
task_statement_sha256. No join against mbpp_300_tasks.csv is needed here:
extraction only needs the prompt text. Test execution against
reference_solution/test_list belongs to the later causal-intervention stage,
not this one.

For each prompt instance:
  1. Run Gemma-2-2B (4-bit) and capture the Layer-12 residual-stream
     activation at the final prompt token via a forward hook (NOT
     output_hidden_states=True, which keeps all 26 layers alive and is
     the easiest way to OOM a 4GB card).
  2. Pass that activation through the GemmaScope SAE to get the 16k-wide
     latent feature vector.
  3. Save both vectors to disk, keyed by prompt_sha256, and append summary
     stats (L0 norm, mean/max activation) to results/activation_stats.csv.

Built for a 4GB card: batch size is always 1, VRAM is logged and the CUDA
cache is cleared every --empty-cache-every prompts, and the run is
resumable — prompt_sha256 values already present in activation_stats.csv
are skipped, so a crash partway through costs nothing.

Usage (real run, after Stage 2 verification passes):
    python src/extract_stage3_features.py

Usage (dry run with no GPU/model/SAE — validates I/O, resume, and stats
logic before spending GPU time):
    python src/extract_stage3_features.py --mock --limit 36
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROMPT_INSTANCES_CSV = ROOT / "data" / "processed" / "mbpp_prompt_instances.csv"
STATS_CSV = ROOT / "results" / "activation_stats.csv"
RAW_DIR = ROOT / "activations" / "raw"
SAE_LATENT_DIR = ROOT / "activations" / "sae_latents"

STATS_FIELDS = [
    "prompt_sha256", "task_id", "prompt_family", "split",
    "l0_norm", "mean_activation", "max_activation", "n_active_features",
    "raw_act_path", "sae_latent_path", "extraction_seconds",
]


def short_id(prompt_sha256: str) -> str:
    # 16 hex chars (64 bits) is more than enough to stay collision-free
    # across ~900 instances, and keeps filenames shorter than the full hash.
    return prompt_sha256[:16]


def read_prompt_instances(path=PROMPT_INSTANCES_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(
            f"{path} has no rows — run src/prepare_dataset.py first (Stage 1)."
        )
    return rows


def already_done_ids(stats_csv=STATS_CSV):
    if not stats_csv.exists():
        return set()
    with open(stats_csv, newline="", encoding="utf-8") as f:
        return {row["prompt_sha256"] for row in csv.DictReader(f)}


def append_stats_row(row, stats_csv=STATS_CSV):
    write_header = not stats_csv.exists()
    with open(stats_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Real backend: Gemma-2-2B (4-bit) + GemmaScope SAE, hook-based extraction
# ---------------------------------------------------------------------------
class RealExtractor:
    def __init__(self, model_id="google/gemma-2-2b", layer_index=12,
                sae_release = "gemma-scope-2b-pt-res",sae_id = "layer_12/width_16k/average_l0_82"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from sae_lens import SAE

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": 0},  # Forces all layers onto cuda:0, preventing CPU offload errors
            torch_dtype=torch.float16,
        )
        self.model.eval()

        self.sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=self.device)

        self._captured = {}
        self.model.model.layers[layer_index].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._captured["act"] = hidden[:, -1, :].detach()  # last token only

    def extract(self, prompt_text):
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            self.model(**inputs)
        raw_act = self._captured["act"].float()          # [1, d_model]
        sae_latent = self.sae.encode(raw_act)             # cast to fp32 before the SAE
        
        # Explicitly detach both tensors from the computation graph before converting to NumPy
        return (
            raw_act.squeeze(0).detach().cpu().numpy(), 
            sae_latent.squeeze(0).float().detach().cpu().numpy()
        )
    def memory_report(self):
        if self.torch.cuda.is_available():
            alloc = self.torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = self.torch.cuda.memory_reserved() / (1024 ** 2)
            return f"allocated={alloc:.1f}MB reserved={reserved:.1f}MB"
        return "cuda unavailable"

    def empty_cache(self):
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Mock backend: deterministic synthetic activations, no GPU/model/SAE needed
# ---------------------------------------------------------------------------
class MockExtractor:
    def __init__(self, d_model=2304, m_latents=16384, avg_l0=70):
        self.d_model = d_model
        self.m_latents = m_latents
        self.avg_l0 = avg_l0

    def extract(self, prompt_text):
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt"
        ).to(self.device)

        with self.torch.inference_mode():
            self.model(**inputs)

            # Extract captured Layer-12 activation
            raw_act = self._captured["act"].detach().float()

            # Encode activation using GemmaScope SAE
            sae_latent = self.sae.encode(raw_act)

            # Explicitly detach both tensors
            raw_act = raw_act.detach()
            sae_latent = sae_latent.detach()

        return (
            raw_act.squeeze(0).cpu().numpy(),
            sae_latent.squeeze(0).float().cpu().numpy()
        )
    def memory_report(self):
        return "mock mode — no GPU used"

    def empty_cache(self):
        pass


def summarize(sae_latent):
    active_mask = sae_latent > 0
    return {
        "l0_norm": int(active_mask.sum()),
        "mean_activation": float(sae_latent[active_mask].mean()) if active_mask.any() else 0.0,
        "max_activation": float(sae_latent.max()),
        "n_active_features": int(active_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="dry run with synthetic activations, no GPU needed")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N prompt instances")
    ap.add_argument("--empty-cache-every", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SAE_LATENT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_CSV.parent.mkdir(parents=True, exist_ok=True)

    instances = read_prompt_instances()
    if args.limit:
        instances = instances[: args.limit]
    done = already_done_ids()
    todo = [row for row in instances if row["prompt_sha256"] not in done]

    print(f"Prompt instances: {len(instances)} | already done: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("Nothing to do — all prompt instances already extracted.")
        return

    extractor = MockExtractor() if args.mock else RealExtractor()
    print(f"Backend: {'MOCK' if args.mock else 'REAL (Gemma-2-2B 4-bit + GemmaScope SAE)'}")

    for i, row in enumerate(todo, start=1):
        t0 = time.time()
        sid = short_id(row["prompt_sha256"])
        try:
            raw_act, sae_latent = extractor.extract(row["prompt"])
        except Exception as e:  # noqa: BLE001 — keep the batch alive, log and continue
            print(f"  [{i}/{len(todo)}] FAILED prompt_sha256={sid} ({e}); skipping.")
            continue

        raw_path = RAW_DIR / f"{sid}.npy"
        latent_path = SAE_LATENT_DIR / f"{sid}.npy"
        np.save(raw_path, raw_act)
        np.save(latent_path, sae_latent)

        stats = summarize(sae_latent)
        append_stats_row({
            "prompt_sha256": row["prompt_sha256"], "task_id": row["task_id"],
            "prompt_family": row["prompt_family"], "split": row["split"],
            "raw_act_path": str(raw_path.relative_to(ROOT)),
            "sae_latent_path": str(latent_path.relative_to(ROOT)),
            "extraction_seconds": round(time.time() - t0, 3),
            **stats,
        })

        if i % args.empty_cache_every == 0:
            extractor.empty_cache()
        if i % args.log_every == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] split={row['split']:<10} family={row['prompt_family']:<10} "
                  f"L0={stats['l0_norm']:<4} {extractor.memory_report()}")

    print(f"\nDone. Stats: {STATS_CSV}  |  raw activations: {RAW_DIR}/  |  SAE latents: {SAE_LATENT_DIR}/")
    print("Next: python src/verify_stage3_extraction.py, then fill in context.md's Stage 3 notes section.")


if __name__ == "__main__":
    main()
