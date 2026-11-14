# SAE Research — Phase 3

**Research Title:** Evidence-Grounded Causal Evaluation and Visual Analytics of Sparse Autoencoder Features in Code-Generation Behaviour of Gemma-2-2B.

## 1. Project Overview

This project investigates how Sparse Autoencoder (SAE) features influence the code-generation behaviour of Gemma-2-2B.

The research uses the MBPP programming benchmark to study whether selected SAE features are associated with successful or unsuccessful code generation and whether controlled feature interventions produce measurable behavioural changes.

The experimental workflow includes:

1. Dataset preparation and prompt generation.
2. Baseline code generation and evaluation.
3. Residual-stream activation extraction.
4. SAE feature analysis and selection.
5. Feature ablation, steering, and matched control experiments.
6. Statistical analysis and visual analytics.

The project separates exploratory feature discovery from intervention evaluation to reduce data leakage and improve experimental validity.

---

## 2. Hardware and Execution Plan

The project uses a hybrid local/cloud computing setup.

| Environment | Hardware | Responsibilities |
|---|---|---|
| Local Windows machine | NVIDIA T1000 4 GB | Dataset preparation, code development, statistical analysis, and visualization |
| Google Colab / Linux cloud | Cloud GPU | Gemma-2-2B inference, activation extraction, SAE processing, and intervention experiments |

**Hardware limitation:** Do not attempt full Gemma-2-2B inference and training of three SAEs on the 4 GB T1000 GPU.

The local machine is primarily used for lightweight processing and analysis of saved experimental results.

---

## 3. Environment Setup

Create the Conda environment using Anaconda Prompt or PowerShell:

```powershell
conda create -n sae-research python=3.11 -y
conda activate sae-research

python -m pip install --upgrade pip
```

Install PyTorch separately using the official installation selector:

https://pytorch.org/get-started/locally/

Choose a compatible Windows and CUDA configuration. If CUDA installation fails, CPU PyTorch can be used for local analysis.

Install the remaining dependencies:

```powershell
python -m pip install -r requirements-local.txt
```

For cloud experiments, install the dependencies specified for the Colab environment.

---

## 4. Hugging Face Access

The MBPP dataset is public. Gemma-2-2B is gated and requires accepting Google's model access terms.

- Gemma-2-2B: https://huggingface.co/google/gemma-2-2b
- Hugging Face access tokens: https://huggingface.co/settings/tokens

Authenticate in the cloud environment before loading Gemma.

**Security requirements:**
- Never place Hugging Face tokens directly in source files.
- Never commit access tokens to GitHub.
- Use environment variables or the Hugging Face login mechanism.

---

## 5. Dataset Preparation — Stage 1

Stage 1 downloads the hand-verified MBPP sanitized subset from Hugging Face.

It performs the following operations:

- Selects 300 tasks using a recorded random seed.
- Splits tasks into 70% training, 15% validation, and 15% held-out test sets.
- Generates three prompt instruction formats:
  - Canonical
  - Paraphrase-style instruction
  - OOD-style instruction
- Saves raw and processed datasets.
- Creates a reproducibility manifest.

Run from the project root:

```powershell
python src/prepare_dataset.py --n_tasks 300 --seed 42
python src/validate_dataset.py
```

Expected files:

```text
data/
├── raw/
│   └── mbpp_sanitized_source.jsonl
│
└── processed/
    ├── mbpp_300_tasks.csv
    ├── mbpp_prompt_instances.csv
    └── task_split_manifest.csv

results/
└── dataset_manifest.json
```

### Dataset methodology

The three prompt forms vary the instruction wrapper while preserving the original task statement.

They should be treated as controlled template variants, not independently human-validated semantic paraphrases.

If strict semantic paraphrasing is required, human-reviewed paraphrases should be created and validated before final test evaluation.

All prompt variants belonging to the same task remain in the same split to prevent task-level data leakage.

---

## 6. Experimental Pipeline

### Stage 2 — Baseline Generation and Evaluation

Generate code using Gemma-2-2B without SAE interventions.

Evaluate generated solutions using the MBPP test cases.

The baseline provides the reference performance against which later interventions are compared.

### Stage 3 — Activation Extraction

Extract residual-stream activations from the selected Gemma-2-2B layer.

The current experimental configuration uses:

| Parameter | Configuration |
|---|---|
| Model | `google/gemma-2-2b` |
| SAE release | `gemma-scope-2b-pt-res` |
| SAE ID | `layer_12/width_16k/average_l0_82` |
| Layer | 12 |
| Activation position | Final token |

The activation cache is used for subsequent SAE feature analysis.

### Stage 4 — SAE Feature Analysis and Selection

Analyze SAE feature activations to identify features associated with successful and unsuccessful code generation.

Feature selection uses training data, with validation data used for subsequent validation.

The selected features are exploratory candidates.

**Statistical limitation:** No features were statistically significant after False Discovery Rate (FDR) correction at the 0.05 threshold in the feature-selection analysis.

Therefore, selected features must not be described as statistically confirmed success- or failure-causing features.

### Stage 5 — Feature Intervention Experiments

Perform controlled interventions on selected SAE features.

The experiment conditions include:

| Experiment | Purpose |
|---|---|
| Baseline | Reference model behaviour without intervention |
| No-op control | Verify that the intervention hook does not unintentionally change activations |
| Ablation | Suppress selected feature activations |
| Positive steering | Increase selected feature activations |
| Negative steering | Decrease selected feature activations |
| Random ablation | Matched random-feature control |
| Random positive steering | Random-feature positive steering control |
| Random negative steering | Random-feature negative steering control |

The intervention experiments use saved prompt-level results and record information such as:

- Prompt hash and task ID
- Prompt family and data split
- Intervention type
- Feature ID and feature group
- Steering strength
- Baseline and intervention pass status
- Generated code and execution errors
- Execution time
- Hook calls
- Maximum residual change
- Random seed

The Stage 5 results are saved in:

```text
results/
└── stage5/
    └── stage5_results.csv
```

The actual output path depends on the configured experiment directory.

The experiment results are evaluated using the validation split. The held-out test split must remain untouched until the final evaluation protocol is defined and frozen.

---

## 7. Statistical Analysis and Visual Analytics — Stage 6

Stage 6 analyzes the saved Stage 5 results.

It does not require another Gemma inference run or cloud GPU access.

The analysis includes:

- Overall intervention pass rates
- Prompt-family performance comparisons
- Feature-level intervention summaries
- Paired prompt-level comparisons
- Data-quality checks
- Statistical summaries and visualizations

Run the analysis from the project root:

```powershell
python src/run_stage6_analysis.py
```

The input file should point to the saved Stage 5 results CSV.

### Advanced visualizations

Additional visualizations can provide deeper insight into intervention behaviour.

| Visualization | Research purpose |
|---|---|
| Prompt-level effect distribution | Examine variation in intervention effects across prompts |
| SAE feature-intervention heatmap | Compare feature-specific effects across intervention types |
| Effect-size forest plot | Display effect estimates and bootstrap confidence intervals |
| Prompt-family dumbbell plot | Compare baseline and intervention performance across prompt families |
| Intervention-response landscape | Examine the relationship between residual perturbation magnitude and pass-rate change |
| Selected-versus-random ECDF | Compare the distributions of selected-feature and random-control effects |

The advanced graph-generation script can be saved as:

```text
src/
└── advanced_stage6_graphs.py
```

Run it with:

```powershell
python src/advanced_stage6_graphs.py
```

The script expects `stage5_results.csv` in the configured input location. Adjust `INPUT_FILE` if the CSV is stored elsewhere.

### Important analysis considerations

- Compare intervention outcomes against the matching prompt's baseline.
- Aggregate repeated feature experiments at the prompt level before paired statistical comparisons.
- Treat bootstrap confidence intervals as descriptive unless the statistical testing protocol supports stronger conclusions.
- Residual change measures perturbation magnitude, not semantic feature importance or causal strength.
- The Stage 5 results CSV may contain multiple top-k configurations. Do not describe combined results as exclusively top-k=3.
- Report the number of prompts and experiments contributing to each comparison.
- Avoid treating repeated feature trials on the same prompt as independent observations.

---

## 8. Reproducibility and Research Validity

The following requirements apply throughout the project:

1. Record random seeds and experimental configurations.
2. Preserve raw experimental outputs.
3. Keep training, validation, and held-out test data separate.
4. Use training and validation data for exploratory feature selection.
5. Do not use held-out test outcomes to select features, adjust steering strengths, or modify prompts.
6. Maintain matched random-feature controls.
7. Report unsuccessful experiments and execution errors transparently.
8. Distinguish exploratory associations from statistically supported effects.
9. Document the exact dataset, model, SAE configuration, and intervention settings used in every reported result.
10. Do not report placeholder or demonstration metrics as experimental findings.

---

## 9. Project Structure

The current project directory is organized as follows:

```text
SAE_Research_Phase3_Starter/
│
├── .vscode/
│
├── activations/
│   ├── raw/
│   └── sae_latents/
│
├── config/
│   └── experiment_config.json
│
├── data/
│   ├── processed/
│   │   ├── mbpp_300_tasks.csv
│   │   ├── mbpp_prompt_instances.csv
│   │   └── task_split_manifest.csv
│   │
│   └── raw/
│       └── mbpp_sanitized_source.jsonl
│
├── results/
│   ├── stage4/
│   │   ├── feature_selection_summary.json
│   │   ├── feature_statistics.csv
│   │   ├── ranked_features.csv
│   │   └── selected_features.json
│   │
│   ├── stage5/
│   │   ├── ablation_results.csv
│   │   ├── baseline_results.csv
│   │   ├── noop_control_results.csv
│   │   ├── random_control_results.csv
│   │   ├── stage5_results.csv
│   │   └── steering_results.csv
│   │
│   └── stage6/
│
├── activation_stats.csv
├── baseline_train_eval.csv
├── dataset_manifest.json
│
├── src/
│   ├── __pycache__/
│   ├── .ipynb_checkpoints/
│   │
│   ├── dashboard_stage6.html
│   ├── extract_stage3_features.py
│   ├── prepare_dataset.py
│   ├── run_stage4_baseline_eval.py
│   ├── run_stage5_interventions.py
│   ├── run_stage6_analysis.py
│   ├── sandbox_worker.py
│   ├── select_stage4_features.py
│   ├── stage5_intervention_engine.py
│   ├── validate_dataset.py
│   ├── verify_stage2_loading.py
│   └── verify_stage3_extraction.py
│
├── .gitignore
├── context.md
├── README.md
└── requirements-local.txt
```

### Directory descriptions

| Directory / File | Purpose |
|---|---|
| `activations/raw/` | Stores extracted raw model activations |
| `activations/sae_latents/` | Stores SAE latent activation outputs |
| `config/` | Contains experiment configuration |
| `data/raw/` | Stores the original MBPP dataset |
| `data/processed/` | Contains sampled tasks, prompt instances, and split information |
| `results/stage4/` | Contains SAE feature-selection results |
| `results/stage5/` | Contains baseline, ablation, steering, and control experiment results |
| `results/stage6/` | Stores statistical analysis and visualization outputs |
| `src/` | Contains Python implementation scripts and the Stage 6 dashboard |
| `activation_stats.csv` | Stores activation statistics |
| `baseline_train_eval.csv` | Stores baseline training evaluation results |
| `dataset_manifest.json` | Records dataset preparation information |
| `experiment_config.json` | Stores experimental configuration |
| `context.md` | Contains project context and implementation notes |
| `dashboard_stage6.html` | Stage 6 visualization dashboard |

**Note:** The `results/stage6/` directory is shown without listing its contents because they are not visible in the screenshots. The structure above reflects the files currently visible in your VS Code Explorer.

## 10. Current Project Status

| Stage | Description | Status |
|---|---|---|
| Stage 1 | Dataset preparation and prompt generation | Implemented |
| Stage 2 | Baseline generation and evaluation | Completed |
| Stage 3 | Activation extraction | Completed |
| Stage 4 | SAE feature analysis and selection | Completed; no FDR-significant features |
| Stage 5 | Feature intervention experiments | Completed |
| Stage 6 | Statistical analysis and visualization | completed |

Stage 6 uses the saved Stage 5 results. Local analysis and visualization can continue without access to the Colab GPU.

---

## 11. Research Reporting Principle

The central objective is to evaluate whether controlled interventions on SAE features produce measurable changes in code-generation behaviour.

A change in pass rate alone does not establish that a feature represents a specific semantic concept or that it causally controls successful code generation.

Conclusions must consider:

- Paired intervention effects
- Random-feature control comparisons
- Prompt-family differences
- Statistical uncertainty
- Experimental coverage
- Data-selection limitations

