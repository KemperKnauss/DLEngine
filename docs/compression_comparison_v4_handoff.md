# Scaled Compression Comparison Handoff

## Status

The scaled four-hour experiment in
[compression_comparison_plan.md](compression_comparison_plan.md) is complete.

- Branch: `codex/fair-compression-comparison`
- Artifact: `comparison_v4`
- Full compute duration: 9,562.8 seconds (2 hours, 39 minutes, 23 seconds)
- Seeds: 8
- Base and distillation training runs: 88
- Held-out condition evaluations: 104 (13 conditions per seed)
- Tests: 10 passed
- Train/validation/test game overlap: zero
- W&B summary:
  [comparison_v4_summary](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression/runs/k06fg5v0)

The previous `comparison_v3` experiment and artifacts were not overwritten.

## Method

The experiment reused the game-grouped Stockfish 18 labels:

| Split | Positions |
|---|---:|
| Train | 59,760 |
| Validation | 7,496 |
| Test | 7,475 |

For each seed (`7, 17, 27, 37, 47, 57, 67, 77`), the pipeline independently trained:

- one FP32 surrogate teacher;
- one direct student;
- nine distilled students covering alpha `{0.25, 0.50, 0.75}` and temperature `{1, 2, 4}`.

Models trained for at most 20 epochs with four-epoch validation early stopping. Direct and
distilled candidates within a seed started from identical student weights. The global
distillation setting was selected by mean validation Stockfish policy cross-entropy. Test metrics
were not used for selection.

Selected distillation configuration:

```text
alpha = 0.50
temperature = 2.0
```

The selected student and teacher were then dynamically INT8-quantized and globally pruned at 25%,
50%, and 75%. Every pruning comparison included an unpruned model receiving the same additional
fine-tuning epoch.

## Command

```powershell
wsl -d Ubuntu --cd /mnt/c/Users/danie/Repositories/DLEngine -- `
  /home/dq24/dlengine-comparison/.venv/bin/python `
  -m scripts.run_scaled_compression_comparison `
  --artifact-name comparison_v4 --epochs 20 --patience 4 `
  --batch-size 256 --latency-repeats 300 --latency-trials 7 `
  --max-runtime-minutes 210
```

The complete training and evaluation run finished inside the four-hour budget. Aggregate W&B
upload initially rejected mixed string/numeric sparsity values after all compute had finished.
The runner now has a tested recovery path:

```powershell
wsl -d Ubuntu --cd /mnt/c/Users/danie/Repositories/DLEngine -- `
  /home/dq24/dlengine-comparison/.venv/bin/python `
  -m scripts.run_scaled_compression_comparison `
  --artifact-name comparison_v4 --finalize-only --elapsed-seconds 9562.8
```

This regenerates aggregation, figures, W&B summary, and handoff metadata without retraining.

## Aggregate Results

Values are means over eight seeds. Top-1 uncertainty is a 95% confidence interval over seeds.
CPU latency is the median of the per-seed latency medians.

| Condition | Top-1 | Top-3 | Value RMSE | Size MB | CPU ms |
|---|---:|---:|---:|---:|---:|
| FP32 teacher | 23.58% +/- 0.99% | 45.37% | 0.222 | 46.59 | 2.324 |
| INT8 teacher | 23.55% +/- 0.92% | 45.32% | 0.222 | 11.72 | 0.496 |
| Teacher pruning control | 24.41% +/- 0.86% | 45.95% | 0.218 | 46.59 | 2.285 |
| Teacher prune 75% | 24.84% +/- 0.30% | 46.23% | 0.235 | 46.59 | 2.146 |
| Direct student | 22.83% +/- 0.79% | 44.14% | 0.220 | 11.33 | 0.355 |
| Distilled student | 23.15% +/- 1.00% | 44.82% | 0.218 | 11.33 | 0.380 |
| Distilled student INT8 | 23.01% +/- 0.85% | 44.65% | 0.220 | 2.90 | 0.233 |
| Student pruning control | 23.22% +/- 0.95% | 45.08% | 0.216 | 11.33 | 0.387 |
| Student prune 25% | 23.46% +/- 0.87% | 44.88% | 0.216 | 11.33 | 0.373 |
| Student prune 75% | 21.27% +/- 0.89% | 43.12% | 0.303 | 11.33 | 0.379 |

The full 13-condition aggregate table is in
`outputs/comparison_v4/aggregate_metrics.csv`; all individual seed rows are in
`outputs/comparison_v4/per_seed_metrics.csv`.

## Paired Interpretation

Paired differences compare conditions within the same seed and are more informative than
comparing separate confidence intervals:

| Comparison | Mean top-1 change | Paired 95% CI |
|---|---:|---:|
| INT8 teacher minus FP32 teacher | -0.03 pp | +/- 0.31 pp |
| Distilled minus direct student | +0.32 pp | +/- 1.33 pp |
| Teacher prune 75% minus matched control | +0.44 pp | +/- 0.80 pp |
| Student prune 25% minus matched control | +0.25 pp | +/- 0.57 pp |

The strongest result is therefore **near-lossless INT8 teacher quantization**:

- 74.9% smaller serialized model;
- approximately 4.7x faster one-thread CPU inference;
- only -0.03 percentage points mean paired top-1 change.

The tuned distilled student was slightly better than direct training on average, reversing the
single-seed v3 ordering, but its paired interval crosses zero. It should be described as
competitive rather than conclusively superior.

Pruning gains also cross zero after matched fine-tuning controls. Unstructured pruning still does
not reduce actual dense checkpoint size, and the larger teacher's apparent 75% pruning gain comes
with worse value RMSE. The distilled student degrades clearly at 75% sparsity.

## Final Poster Assets

The final poster uses exactly three result assets under `figures/comparison_v4/poster/`:

1. `poster_1_compression_method_comparison`
   - aligned top-1, actual serialized size, and CPU latency panels;
   - replaces the confusing v3 size-reduction scatter.
2. `poster_2_accuracy_vs_cpu_latency`
   - all conditions, uncertainty bars, and Pareto frontier.
3. `poster_3_compact_results_table`
   - exact values for FP32 teacher, INT8 teacher, validation-selected pruning conditions, direct
     and distilled students, and the best validation-selected combined condition.

PNG and PDF versions are provided. The pruning-sparsity curve is intentionally retained only as a
diagnostic because the poster has room for three assets.

Additional diagnostics under `figures/comparison_v4/diagnostics/` include:

- pruning tolerance with matched controls;
- the distillation alpha/temperature validation heatmap;
- top-1 distributions across seeds.

## W&B and Artifacts

- W&B project:
  [dlengine-chess-compression](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression)
- Summary run:
  [k06fg5v0](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression/runs/k06fg5v0)
- Individual training links: `outputs/comparison_v4/wandb_runs.json`
- Runner: `scripts/run_scaled_compression_comparison.py`
- Metrics and metadata: `outputs/comparison_v4/`
- Poster and diagnostics: `figures/comparison_v4/`
- Local checkpoints: `checkpoints/comparison_v4/` (ignored by Git)

## Limitations and Next Steps

- This compresses a neural Stockfish surrogate, not Stockfish's shipped NNUE.
- The dataset size was held fixed; scaling was spent on seeds, training, tuning, and reliable
  uncertainty rather than generating more expensive Stockfish labels.
- Top-1 agreement remains an imperfect proxy for playing strength. Engine-match Elo remains a
  follow-up experiment.
- Dynamic quantization uses the deprecated `torch.ao.quantization` API. A maintenance pass should
  migrate to `torchao`.
- A future pruning study should use a sparse runtime or structured pruning if real storage and
  latency improvements are required.
