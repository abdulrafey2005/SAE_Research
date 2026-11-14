# Stage 6 — Results Analysis and Interpretation

## Dataset
- Valid experiment records analyzed: **5919**
- Unique validation prompt instances: **135**
- Unique MBPP task IDs: **45**
- Duplicate experiment IDs: **0**
- Selected feature IDs observed: **10**
- Random feature IDs observed: **17**

The file contains multiple feature configurations / partial runs. Results below describe the uploaded CSV as a whole; they must not be described as exclusively one top-k setting.

## Overall outcomes

| Condition | Runs | Passed | Pass rate | Mean residual delta |
|---|---:|---:|---:|---:|
| ablation | 830 | 294 | 35.42% | 1.870963 |
| baseline | 135 | 47 | 34.81% | 0.000000 |
| negative_steering | 830 | 294 | 35.42% | 0.164996 |
| noop_control | 135 | 47 | 34.81% | 0.000000 |
| positive_steering | 830 | 289 | 34.82% | 0.308087 |
| random_ablation | 1053 | 387 | 36.75% | 0.000000 |
| random_negative_steering | 1053 | 387 | 36.75% | 0.000000 |
| random_positive_steering | 1053 | 387 | 36.75% | 0.111606 |

## Prompt-family results

| Prompt family | Condition | Runs | Pass rate |
|---|---|---:|---:|
| canonical | ablation | 278 | 36.33% |
| canonical | baseline | 45 | 35.56% |
| canonical | negative_steering | 278 | 35.97% |
| canonical | noop_control | 45 | 35.56% |
| canonical | positive_steering | 278 | 35.97% |
| canonical | random_ablation | 354 | 37.01% |
| canonical | random_negative_steering | 354 | 37.01% |
| canonical | random_positive_steering | 354 | 37.85% |
| ood | ablation | 274 | 36.13% |
| ood | baseline | 45 | 35.56% |
| ood | negative_steering | 274 | 36.50% |
| ood | noop_control | 45 | 35.56% |
| ood | positive_steering | 274 | 34.67% |
| ood | random_ablation | 345 | 37.97% |
| ood | random_negative_steering | 345 | 37.97% |
| ood | random_positive_steering | 345 | 37.10% |
| paraphrase | ablation | 278 | 33.81% |
| paraphrase | baseline | 45 | 33.33% |
| paraphrase | negative_steering | 278 | 33.81% |
| paraphrase | noop_control | 45 | 33.33% |
| paraphrase | positive_steering | 278 | 33.81% |
| paraphrase | random_ablation | 354 | 35.31% |
| paraphrase | random_negative_steering | 354 | 35.31% |
| paraphrase | random_positive_steering | 354 | 35.31% |

## Paired prompt-level comparisons

The paired analysis first averages repeated feature-condition outcomes within each prompt. This avoids treating every feature run on the same prompt as an independent prompt. Wilcoxon p-values are exploratory.

| Comparison | Paired prompts | Mean paired difference | Wilcoxon p-value |
|---|---:|---:|---:|
| ablation vs baseline | 135 | 0.0000 | 0.654721 |
| negative_steering vs baseline | 135 | 0.0000 | 1 |
| noop_control vs baseline | 135 | 0.0000 | 1 |
| positive_steering vs baseline | 135 | -0.0062 | 0.317311 |
| random_ablation vs baseline | 135 | 0.0000 | 1 |
| random_negative_steering vs baseline | 135 | 0.0000 | 1 |
| random_positive_steering vs baseline | 135 | -0.0009 | 0.654721 |
| selected-feature interventions vs random controls | 135 | -0.0017 | 0.285049 |

## Data-quality observations
- Baseline rows with nonzero hook calls: **3**.
- No-op rows with nonzero hook calls: **135**.
- No-op rows with nonzero residual delta: **0**.

Baseline rows with hook calls indicate that some baseline records came from an earlier run/configuration. They are retained as recorded, but this inconsistency must be disclosed.

## Interpretation and limitations
- Pass rate is the proportion of generated solutions that passed the recorded MBPP tests.
- A zero residual delta means that the recorded intervention did not change the residual activation for that run; it should not be interpreted as evidence that the feature has no effect in every context.
- Mixed feature counts and partial runs mean the dataset is not a balanced, single top-k experiment.
- Repeated feature interventions share prompts; run-level counts are descriptive and are not independent observations.
- The validation split was used for this intervention analysis. The test split was not included in this file and remains reserved for held-out evaluation.
- Statistical associations in this analysis do not by themselves establish a general causal mechanism. Interpret results in light of actual residual changes and experimental coverage.

## Files
- `stage6_overall_summary.csv`: aggregate condition results.
- `stage6_prompt_family_summary.csv`: canonical / paraphrase / OOD breakdown.
- `stage6_feature_summary.csv`: individual feature outcomes and residual deltas.
- `stage6_paired_prompt_comparison.csv`: paired prompt-level comparisons.
- `stage6_data_quality.json`: dataset diagnostics.
- `pass_rate_by_intervention.png` and `pass_rate_by_prompt_family.png`: figures.
