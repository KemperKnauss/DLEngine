# Fair Compression Comparison Handoff

## Status

The experiment specified in
[compression_comparison_plan.md](compression_comparison_plan.md) is complete. It ran all 13
conditions on the same game-grouped Stockfish dataset, evaluated only on the held-out test split,
synced runs to W&B, and generated the three poster figures, compact table, and diagnostics.

- Branch: `codex/fair-compression-comparison`
- Full run artifact: `comparison_v3`
- Full run duration: 242 seconds
- Completion date: June 6, 2026
- Tests: 8 passed
- Split overlap: 0 train/validation, 0 train/test, 0 validation/test

## Method

Stockfish 18 remains the external expert that produced depth-10, MultiPV-5 labels. The comparison
compresses a trainable FP32 neural surrogate, not `stockfish.exe` or its production NNUE.

| Split | Positions |
|---|---:|
| Train | 59,760 |
| Validation | 7,496 |
| Test | 7,475 |

The FP32 surrogate uses hidden widths `1024 -> 512`; the student uses `256 -> 128`. Both consume
the existing 18-plane board encoding and emit a 20,480-action policy plus a scalar value. The
runner trains the teacher, a direct student, and a distilled student, then evaluates dynamic INT8
quantization and global unstructured magnitude pruning at 25%, 50%, and 75%. Pruned models and
unpruned controls receive the same extra fine-tuning epoch.

## Environment

- WSL2 Ubuntu 26.04
- Managed Python 3.12.13
- PyTorch `2.12.0+cu132`
- CUDA runtime 13.2
- NVIDIA RTX A1000 6 GB Laptop GPU
- Virtual environment: `/home/dq24/dlengine-comparison/.venv`
- W&B project:
  [dlengine-chess-compression](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression)

The Windows CPU-only Python installation was not modified.

## Commands

Smoke validation:

```powershell
wsl -d Ubuntu --cd /mnt/c/Users/danie/Repositories/DLEngine -- `
  /home/dq24/dlengine-comparison/.venv/bin/python `
  -m scripts.run_fair_compression_comparison `
  --artifact-name comparison_v3_smoke --smoke --epochs 1 `
  --batch-size 64 --latency-repeats 20 --max-runtime-minutes 30
```

Full experiment:

```powershell
wsl -d Ubuntu --cd /mnt/c/Users/danie/Repositories/DLEngine -- `
  /home/dq24/dlengine-comparison/.venv/bin/python `
  -m scripts.run_fair_compression_comparison `
  --artifact-name comparison_v3 --epochs 6 --patience 2 `
  --batch-size 256 --latency-repeats 200 --max-runtime-minutes 210
```

Tests:

```powershell
wsl -d Ubuntu --cd /mnt/c/Users/danie/Repositories/DLEngine -- `
  /home/dq24/dlengine-comparison/.venv/bin/python `
  -m unittest discover -s tests -v
```

The runner is resumable. Reusing the same artifact name reloads valid base checkpoints. Pass
`--no-wandb` for local-only reruns.

## Main Results

| Condition | Top-1 | Top-3 | Value RMSE | Size MB | CPU ms |
|---|---:|---:|---:|---:|---:|
| FP32 teacher | 21.03% | 40.91% | 0.223 | 46.59 | 2.730 |
| INT8 teacher | 20.76% | 40.94% | 0.223 | 11.72 | 0.553 |
| Teacher prune 25% | 23.26% | 43.99% | 0.210 | 46.59 | 2.966 |
| Direct student | 19.83% | 39.21% | 0.209 | 11.33 | 0.461 |
| Distilled student | 18.92% | 40.20% | 0.241 | 11.33 | 0.360 |
| Distilled student INT8 | 18.72% | 39.88% | 0.242 | 2.90 | 0.533 |
| Distilled student prune 75% | 15.72% | 37.30% | 0.337 | 11.33 | 1.203 |

The complete 13-row table is in `outputs/comparison_v3/final_metrics.csv`.

## Interpretation

- **INT8 teacher quantization is the strongest practical compression result.** It reduced actual
  serialized size by 74.9%, improved one-thread CPU latency by about 4.9x, and lost only 0.27
  percentage points of top-1 agreement.
- **The direct student beat the distilled student on held-out top-1 and value RMSE.** Distillation
  did lower validation policy cross-entropy, but that did not convert into better top-1 agreement
  under this loss weighting and six-epoch budget.
- **Pruning needs the matched controls for honest interpretation.** An extra fine-tuning epoch
  improved the unpruned teacher from 21.03% to 22.65%. The 25%-pruned teacher reached 23.26%, only
  0.61 percentage points above that matched control.
- **Unstructured pruning did not produce practical storage or speed gains.** Dense checkpoint
  size stayed unchanged and CPU inference was not faster. The estimated sparse payload exceeded
  dense storage at 25% sparsity because index overhead dominated, matched dense size near 50%, and
  became smaller only at 75%.
- **The distilled student was less pruning-tolerant.** Its 25% condition slightly improved after
  matched fine-tuning, but top-1 fell sharply at 75% sparsity.

These are single-seed results from a surrogate model. They support comparisons within this
controlled benchmark, not claims about compressing Stockfish's shipped NNUE or engine-match Elo.

## Poster Assets

The poster-ready files are under `figures/comparison_v3/poster/` as 300-DPI PNG and PDF:

- `poster_1_accuracy_loss_vs_size_reduction`: compares real size reduction with held-out accuracy
  change and shows theoretical sparse payloads separately.
- `poster_2_accuracy_vs_cpu_latency`: shows every condition and the accuracy-latency Pareto
  frontier.
- `poster_3_accuracy_vs_pruning_sparsity`: isolates teacher and distilled-student pruning
  tolerance using matched 0% controls.
- `poster_compact_results_table`: supplies exact values for the six presentation conditions.

Report-only diagnostics are under `figures/comparison_v3/diagnostics/`, including training curves,
direct-versus-distilled validation policy loss, value RMSE tradeoffs, nonzero parameters, GPU
latency, and peak CUDA memory.

## W&B

Training runs:

- [FP32 teacher](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression/runs/wcakm1yc)
- [Direct student](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression/runs/3y2iqhw2)
- [Distilled student](https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression/runs/ci1alr64)

Evaluation links for all 13 conditions are recorded in
`outputs/comparison_v3/wandb_runs.json`.

## Artifacts and Resume Notes

- Implementation: `src/chess_student/comparison.py`
- Runner: `scripts/run_fair_compression_comparison.py`
- Tests: `tests/test_comparison.py`
- Metrics and metadata: `outputs/comparison_v3/`
- Poster and diagnostics: `figures/comparison_v3/`
- Local checkpoints: `checkpoints/comparison_v3/` (ignored by Git)

PyTorch warns that `torch.ao.quantization.quantize_dynamic` is deprecated in favor of `torchao`.
The current implementation is tested and reproducible, but a future maintenance pass should
migrate the quantization call. A stronger follow-up experiment should run multiple seeds and tune
distillation alpha/temperature before making a general claim that direct training is superior.
