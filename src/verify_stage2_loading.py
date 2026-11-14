"""
Stage 2: Model Loading and SAE Feature Extraction Verification (4-bit Quantized)
Model: Gemma-2-2B
Constraints: NVIDIA Quadro T1000 (4.0 GB VRAM)
"""
from __future__ import annotations

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sae_lens import SAE

def check_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"[GPU Memory] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Total: {total:.2f} MB")
    else:
        print("[GPU Memory] CUDA not available.")

def test_model_loading(model_id: str = "google/gemma-2-2b"):
    print(f"--- Step 1: Loading Tokenizer & Model in 4-bit ({model_id}) ---")
    check_gpu_memory()

    # Configure 4-bit quantization with double quantization and direct GPU mapping to fit within 4GB VRAM
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True
        )
        model.eval()
        print("Model loaded successfully in 4-bit!")
        check_gpu_memory()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def test_sae_loading(release: str = "gemma-scope-2b-pt-res", sae_id: str = "layer_12/width_16k/canonical"):
    print(f"\n--- Step 2: Loading SAE ({release} / {sae_id}) ---")
    check_gpu_memory()

    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("SAE loaded successfully!")
        check_gpu_memory()
        return sae
    except Exception as e:
        print(f"Error loading SAE: {e}")
        return None

def test_feature_extraction(model, tokenizer, sae):
    print("\n--- Step 3: Running Feature Extraction Sanity Check ---")
    prompt = "Write a Python function to calculate the factorial of a number."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        target_layer_idx = 12
        hidden_states = outputs.hidden_states[target_layer_idx]  # [batch, seq_len, d_model]
        
        feature_activations = sae.encode(hidden_states)
        print(f"Input Prompt: '{prompt}'")
        print(f"Hidden States Shape: {hidden_states.shape}")
        print(f"SAE Feature Activations Shape: {feature_activations.shape}")
        
        l0_norm = (feature_activations > 0).float().sum(dim=-1).mean().item()
        print(f"Mean L0 Norm (Active features per token): {l0_norm:.2f}")

    check_gpu_memory()

def main():
    print("==================================================")
    print("   Stage 2: 4-Bit Model & SAE Loading Verification")
    print("==================================================")

    model_id = "google/gemma-2-2b"
    tokenizer, model = test_model_loading(model_id)

    if model is None:
        print("Model loading failed. Aborting Stage 2.")
        return

    release = "gemma-scope-2b-pt-res"
    sae_id = "layer_12/width_16k/canonical"
    sae = test_sae_loading(release, sae_id)

    if sae is not None:
        test_feature_extraction(model, tokenizer, sae)

    del model
    del tokenizer
    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nStage 2 verification script complete.")

if __name__ == "__main__":
    main()