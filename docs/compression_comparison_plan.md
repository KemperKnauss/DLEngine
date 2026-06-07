# Scaled Four-Hour Fair Compression Experiment Plan

## Purpose

This document is the canonical specification for the fair compression comparison and its poster
assets. Another agent or poster-focused chat should read this file before interpreting results.

The final experiment compares distillation, quantization, and pruning using FP32 neural
surrogates trained on the existing Stockfish labels. It scales the original single-seed run into a
multi-seed benchmark with validation-selected distillation settings, uncertainty estimates, and
poster assets designed for the available presentation space. It must complete training,
compression, evaluation, plots, and handoff documentation within four wall-clock hours.

This experiment does not prune or quantize `stockfish.exe`. Stockfish remains the external expert
that generated the move and value labels. A trainable FP32 surrogate is necessary so all three
compression methods can be evaluated from a controlled neural baseline.

## Data and Models

Reuse the existing leak-free, game-grouped dataset without relabeling:

| Split | Positions | Games |
|---|---:|---:|
| Train | 59,760 | 1,780 |
| Validation | 7,496 | 234 |
| Test | 7,475 | 220 |

The labels were generated with Stockfish 18 at depth 10 and MultiPV 5. Policy targets use the
side-to-move perspective, value targets use White perspective, and illegal moves are masked before
loss calculation and ranking.

Use the existing 18-plane `8x8` board encoding and 20,480-action policy representation.

- FP32 teacher: flattened 1,152-feature input, shared hidden widths `1024 -> 512`, policy and value
  heads.
- Student: flattened input, shared hidden widths `256 -> 128`, identical output heads.
- Both models use ReLU hidden activations and a `tanh` value output.

## Scaled Experimental Design

Use eight deterministic seeds:

```text
7, 17, 27, 37, 47, 57, 67, 77
```

For every seed:

- Train an independently initialized FP32 surrogate teacher and direct student.
- Train for at most 20 epochs with validation early stopping after four non-improving epochs.
- Use the same initial student weights for direct and distilled candidates within a seed.
- Select checkpoints exclusively by validation objective.
- Evaluate final selected conditions once on the held-out test split.

Tune distillation on validation data using the full grid:

```text
alpha:       0.25, 0.50, 0.75
temperature: 1.0, 2.0, 4.0
```

Select one alpha/temperature pair globally using mean validation Stockfish policy
cross-entropy across seeds. This fixed setting defines the final distilled condition for every
seed. Test metrics must not influence this selection.

## Experimental Conditions

Evaluate these conditions on the same held-out test split:

1. FP32 surrogate teacher.
2. Dynamically INT8-quantized teacher.
3. Unpruned teacher fine-tuning control.
4. Teacher globally pruned to 25%, 50%, and 75% sparsity, each with one fine-tuning epoch.
5. Directly trained FP32 student using Stockfish targets only.
6. Distilled FP32 student.
7. Dynamically INT8-quantized distilled student.
8. Unpruned distilled-student fine-tuning control.
9. Distilled student globally pruned to 25%, 50%, and 75% sparsity, each with one fine-tuning
   epoch.

Teacher and direct-student loss:

```text
Stockfish policy soft cross-entropy + Stockfish value MSE
```

Distilled-student loss:

```text
(1 - selected alpha) * Stockfish-target loss + selected alpha * teacher-target loss
```

Teacher-target policy loss is KL divergence at the selected temperature; teacher-target value
loss is MSE. The temperature-squared correction is applied to policy KL. Checkpoints are selected
only by validation loss; test results never influence model, hyperparameter, or compression
selection.

Pruning uses global unstructured L1 magnitude pruning across linear weights. Every pruning
comparison includes an unpruned model that receives the same extra fine-tuning epoch and optimizer
settings. Report actual nonzero weights, dense serialized size, and estimated sparse payload size.

## Environment and Four-Hour Budget

- Work on branch `codex/fair-compression-comparison`.
- Use WSL2 because it exposes the NVIDIA RTX A1000 6 GB and has approximately 952 GB free.
- Install user-space `uv`, a managed Python 3.12 runtime, and an isolated virtual environment.
- Install stable CUDA-enabled PyTorch, preferring CUDA 13.2 and falling back to CUDA 12.6.
- Do not replace or modify the existing Windows CPU-only Python installation.
- Verify CUDA with GPU identity, allocation, matrix multiplication, backward pass, and VRAM usage.
- Use W&B online project `dlengine-chess-compression`, group `comparison_v4`.
- Use mixed precision on CUDA. Start at batch size 256 and retry at 128, then 64 after CUDA OOM.
- Train at most 20 epochs with early stopping after four non-improving validation epochs.
- Reserve 20 minutes for final artifact verification, documentation, commit, and push.
- Stop launching new seeds after 210 elapsed minutes; always finish and aggregate completed seeds.
- The minimum acceptable complete run is five seeds. The target is eight seeds.

Required milestones:

| Elapsed time | Milestone |
|---|---|
| 0:20 | Updated implementation and smoke test complete |
| 1:40 | Multi-seed base models and distillation sweep complete |
| 2:50 | Quantization, pruning, controls, and repeated latency complete |
| 3:30 | Held-out aggregation and poster assets complete |
| 4:00 | Handoff, commit, and push complete |

If a deadline is threatened, stop after the current seed once at least five seeds are complete.
Do not reduce the held-out dataset, use test metrics for selection, or drop a core condition. The
runner must be resumable and skip completed valid artifacts.

## Metrics and Selection

The primary criterion is accuracy-efficiency Pareto dominance. Record:

- Stockfish top-1 and top-3 move agreement.
- Policy cross-entropy.
- Value RMSE and Pearson correlation.
- Total and nonzero parameters.
- Actual sparsity.
- Dense serialized size and estimated sparse payload.
- Fixed-thread batch-one CPU latency.
- GPU latency for FP32 models.
- Peak GPU memory.
- Training and total stage duration.
- Mean, standard deviation, and 95% confidence interval across seeds.
- Median and interquartile range across repeated CPU latency trials.

Store artifacts under:

- `outputs/comparison_v4/`
- `figures/comparison_v4/`
- `checkpoints/comparison_v4/`
- `docs/compression_comparison_v4_handoff.md`

## Poster-Specific Outputs

The poster has room for exactly three result assets. Export each as a 300-DPI PNG and PDF under
`figures/comparison_v4/poster/`. Pruning-sparsity curves remain available as diagnostics but are
not a final poster asset.

### Poster Asset 1: Compression Method Comparison

- Use three aligned horizontal panels or grouped bars for:
  - mean held-out top-1 agreement;
  - actual serialized model size;
  - one-thread CPU latency.
- Include FP32 teacher, INT8 teacher, direct student, selected distilled student, and selected
  combined compressed student.
- Show 95% confidence intervals where the metric varies across seeds.
- Add direct value labels so the chart remains readable without decoding point positions.

This replaces the original size-reduction scatter, which mixed actual dense checkpoint size with
hypothetical sparse payload and produced a misleading negative-size-reduction point. The revised
asset directly compares the three quantities the audience needs: accuracy, storage, and speed.

### Poster Asset 2: Accuracy vs CPU Latency

- X-axis: fixed-thread batch-one CPU latency.
- Y-axis: mean held-out top-1 Stockfish agreement.
- Include every final condition and highlight the Pareto frontier.
- Show 95% confidence intervals across seeds and use median latency across repeated trials.

This shows whether nominal compression becomes practical deployment speed. It prevents reduced
file size or increased sparsity from being mistaken for faster inference. The desirable region is
upper-left: higher accuracy and lower latency.

### Poster Asset 3: Compact Results Table

Rows:

- FP32 teacher.
- INT8 teacher.
- Best pruned teacher.
- Direct student.
- Distilled student.
- Best combined method.

Columns:

- Top-1 agreement.
- Value RMSE.
- Actual serialized size.
- Size reduction.
- CPU latency.
- Actual sparsity.
- Top-1 95% confidence interval.

The table supplies exact values that cannot be read precisely from plots. It identifies the
accuracy leader, smallest model, fastest model, and best accuracy-efficiency compromise.

Together, these three assets communicate accuracy, actual storage, practical speed, uncertainty,
and the strongest deployment tradeoff without spending poster space on a separate pruning curve.

## Diagnostics and Verification

Store report-only plots under `figures/comparison_v4/diagnostics/`:

- Train and validation loss curves.
- Value RMSE versus latency and size.
- GPU latency and peak-memory comparisons.
- Parameter and nonzero-parameter comparisons.
- Direct versus distilled student learning curves.
- Accuracy versus pruning sparsity with matched controls.
- Distillation alpha/temperature validation heatmap.
- Per-seed distributions and confidence intervals.

Verification requirements:

- Test model shapes, legal masking, distillation losses, quantized reload, and pruning sparsity.
- Reconfirm that split game IDs are pairwise disjoint.
- Run a 500-position, two-seed CUDA smoke experiment before the full run.
- Ensure every plotted point corresponds to a consolidated metrics row.
- Verify poster graphs and the compact table at expected poster-column widths.
- Record exact commands, package versions, durations, W&B links, caveats, failures, and resume
  instructions in `docs/compression_comparison_v4_handoff.md`.

## Interpretation Boundaries

- Stockfish's production NNUE is already integer-quantized; this benchmark compresses a trainable
  Stockfish surrogate rather than claiming to quantize the shipped engine from FP32.
- Unstructured pruning may reduce nonzero weights without reducing dense checkpoint size or CPU
  latency. Actual and theoretical storage are both reported.
- Native Stockfish integration and reliable engine-match Elo are deferred because they do not fit
  the four-hour budget.
