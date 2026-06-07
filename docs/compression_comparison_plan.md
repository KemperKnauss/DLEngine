# Four-Hour Fair Compression Experiment Plan

## Purpose

This document is the canonical specification for the fair compression comparison and its poster
assets. Another agent or poster-focused chat should read this file before interpreting results.

The experiment compares distillation, quantization, and pruning using one shared FP32 neural
surrogate trained on the existing Stockfish labels. It must complete environment setup, training,
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
9. Distilled student globally pruned to 50% sparsity with one fine-tuning epoch.

Teacher and direct-student loss:

```text
Stockfish policy soft cross-entropy + Stockfish value MSE
```

Distilled-student loss:

```text
0.3 * Stockfish-target loss + 0.7 * teacher-target loss
```

Teacher-target policy loss is KL divergence at temperature `2.0`; teacher-target value loss is
MSE. The temperature-squared correction is applied to policy KL. Checkpoints are selected only by
validation loss; test results never influence model or compression selection.

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
- Use W&B online project `dlengine-chess-compression`, group `comparison_v3`.
- Use mixed precision on CUDA. Start at batch size 256 and retry at 128, then 64 after CUDA OOM.
- Train at most six epochs with early stopping after two non-improving validation epochs.

Required milestones:

| Elapsed time | Milestone |
|---|---|
| 0:40 | CUDA environment and smoke test complete |
| 1:30 | FP32 teacher complete |
| 2:20 | Direct and distilled students complete |
| 3:20 | Quantization, pruning, and controls complete |
| 3:45 | Held-out evaluation and plots complete |
| 4:00 | Handoff, commit, and push complete |

If a deadline is threatened, reduce training to four completed epochs before reducing the dataset
or dropping a core condition. The runner must be resumable and skip completed valid artifacts.

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

Store artifacts under:

- `outputs/comparison_v3/`
- `figures/comparison_v3/`
- `checkpoints/comparison_v3/`
- `docs/compression_comparison_handoff.md`

## Poster-Specific Outputs

The pipeline may generate additional diagnostic plots, but the following three graphs and one
table are explicitly reserved for the final poster. Export each poster graph as a 300-DPI PNG and
PDF under `figures/comparison_v3/poster/`.

### Poster Graph 1: Accuracy Loss vs Model-Size Reduction

- X-axis: percentage reduction in actual serialized size relative to the FP32 teacher.
- Y-axis: change in held-out top-1 Stockfish agreement relative to the FP32 teacher.
- Include the quantized teacher, best pruned teacher, direct student, distilled student, and
  combined compressed students.
- Distinguish actual dense size from theoretical sparse payload for pruning.

This is the main three-method comparison. It shows which approach produces the greatest storage
reduction for the smallest loss in teacher agreement. The desirable region is upper-right: larger
size reduction and little or no accuracy loss.

### Poster Graph 2: Accuracy vs CPU Latency

- X-axis: fixed-thread batch-one CPU latency.
- Y-axis: held-out top-1 Stockfish agreement.
- Include every final condition and highlight the Pareto frontier.

This shows whether nominal compression becomes practical deployment speed. It prevents reduced
file size or increased sparsity from being mistaken for faster inference. The desirable region is
upper-left: higher accuracy and lower latency.

### Poster Graph 3: Accuracy vs Pruning Sparsity

- X-axis: 0%, 25%, 50%, and 75% target sparsity.
- Y-axis: held-out top-1 Stockfish agreement.
- Use separate lines for the FP32 teacher and distilled student.
- Use equally fine-tuned unpruned models as the 0% controls.

This isolates pruning tolerance. It shows how many weights can be removed before accuracy
collapses and whether distillation changes pruning resistance. Matched controls ensure that gains
from an additional training epoch are not incorrectly attributed to pruning.

### Poster Compact Results Table

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

The table supplies exact values that cannot be read precisely from plots. It identifies the
accuracy leader, smallest model, fastest model, and best accuracy-efficiency compromise.

Together, these poster assets communicate how much compression each method achieved, how much
accuracy it sacrificed, whether compression produced real speed, where pruning failed, and which
condition offered the strongest practical tradeoff.

## Diagnostics and Verification

Store report-only plots under `figures/comparison_v3/diagnostics/`:

- Train and validation loss curves.
- Value RMSE versus latency and size.
- GPU latency and peak-memory comparisons.
- Parameter and nonzero-parameter comparisons.
- Direct versus distilled student learning curves.

Verification requirements:

- Test model shapes, legal masking, distillation losses, quantized reload, and pruning sparsity.
- Reconfirm that split game IDs are pairwise disjoint.
- Run a 500-position CUDA smoke experiment before the full run.
- Ensure every plotted point corresponds to a consolidated metrics row.
- Verify poster graphs and the compact table at expected poster-column widths.
- Record exact commands, package versions, durations, W&B links, caveats, failures, and resume
  instructions in `docs/compression_comparison_handoff.md`.

## Interpretation Boundaries

- Stockfish's production NNUE is already integer-quantized; this benchmark compresses a trainable
  Stockfish surrogate rather than claiming to quantize the shipped engine from FP32.
- Unstructured pruning may reduce nonzero weights without reducing dense checkpoint size or CPU
  latency. Actual and theoretical storage are both reported.
- Native Stockfish integration and reliable engine-match Elo are deferred because they do not fit
  the four-hour budget.
