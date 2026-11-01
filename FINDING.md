## Finding Claim


Post-training uniform asymmetric min-max quantization (dynamic per-tensor scales, round-to-nearest, applied to the recurrent state after every update) of a small SSM (<20 M parameters) preserves associative recall and language-modeling performance down to 4 bits; collapse (defined below) occurs only at 2 bits and below. In addition, the top 15–20 % of channels (ranked by ablation drop on a held-out calibration set) carry nearly all long-range memory: keeping only those channels in full precision while quantizing the rest to 2 bits stays within the collapse thresholds.

## Metric

Primary decision metric: average associative-recall accuracy (exact match of the recalled value) across the controlled synthetic name-recall / multi-query associative-recall tasks at fixed distances 64, 256 and 1024 tokens.

Secondary metric (must also satisfy the bound): TinyStories language-modeling loss in nats.

Collapse is declared if either
  average recall accuracy falls > 10 percentage points below the full-precision baseline, or 
  LM loss rises > 0.05 nats above the full-precision baseline.

## Seeds

All candidate configurations and the baseline are evaluated with at least 3 independent random seeds (different data-order / initialization of any stochastic components of the evaluation pipeline). Report mean ± standard deviation (or 95 % CI) across seeds.

## Baseline

Full-precision (float32) recurrent state, identical model, identical evaluation tasks, identical seeds, and identical dynamic-scale recomputation schedule (but with infinite precision). Static-scale calibration is treated as an explicit experimental arm, not the default baseline.

## Effect size larger than seed spread

The observed degradation (or the improvement from selective full-precision channels) must exceed the seed-to-seed variability. Concretely, the mean difference versus baseline must be larger than the standard deviation across the ≥ 3 seeds (or the confidence interval on the paired difference must exclude both zero and the pre-declared collapse thresholds of 10 points / 0.05 nats).  Any result that fails one or more of the above criteria is recorded only as a preliminary observation, not as a finding.

