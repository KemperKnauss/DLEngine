# DLEngine Corrected Experiment v2 Handoff

## Status

Completed on June 6, 2026 on branch `codex/experiment-v2-poster`.

This run preserves the earlier experiment and writes corrected data, checkpoints, metrics, plots,
and poster assets to new paths ending in `_v2` or `_v2_fair`.

## What Changed

- Streamed the large Lichess `.pgn.zst` file instead of extracting or scanning the full archive.
- Corrected policy targets to use the side-to-move perspective.
- Masked illegal moves before policy loss, top-k ranking, and compression evaluation.
- Propagated source game IDs through extraction and Stockfish labeling.
- Split labels by game rather than by position to prevent game-level leakage.
- Added a matched fine-tuning control to the pruning study.
- Shortened plot labels and filtered compression plots to matched models.
- Replaced the placeholder/broken-encoding poster with a results-filled poster.

## Experiment Configuration

| Setting | Value |
|---|---|
| Dataset | `C:\Users\danie\Downloads\lichess_db_standard_rated_2026-05.pgn.zst` |
| Teacher | Stockfish 18 AVX2 |
| Teacher depth | 10 |
| MultiPV | 5 |
| Ply stride | 2 |
| Usable labels | 74,731 |
| Split | 80/10/10, grouped by source game |
| Train | 59,760 positions / 1,780 games |
| Validation | 7,496 positions / 234 games |
| Test | 7,475 positions / 220 games |
| Epochs | 10 |
| Batch size | 128 |
| Architectures | 5 small CNN + 5 depthwise CNN configurations |
| Training objective | policy soft cross-entropy + value MSE |
| Model selection | lowest validation loss |
| Evaluation | one held-out game-level test split |

The train/validation/test game ID intersections were checked after the run and are all zero.

## Commands Run

Corrected smoke test:

```powershell
python scripts\run_overnight_experiment.py `
  --skip-smoke --skip-compression `
  --target-labels 500 --stockfish-depth 6 --multipv 5 `
  --epochs 1 --batch-size 64 --device cpu `
  --run-suffix _v2_smoke --grouped-split
```

Full corrected distillation and initial compression:

```powershell
python scripts\run_overnight_experiment.py `
  --skip-smoke `
  --target-labels 75000 --stockfish-depth 10 --multipv 5 `
  --epochs 10 --batch-size 128 --device cpu `
  --run-suffix _v2 --grouped-split
```

Fair compression rerun with an equally fine-tuned unpruned control:

```powershell
python -m scripts.compress_students `
  --checkpoints `
    checkpoints\overnight\final_v2\small_cnn_32x5_d10_75k_final_v2\best.pt `
    checkpoints\overnight\final_v2\small_cnn_64x5_d10_75k_final_v2\best.pt `
  --train-labels data\overnight\final_v2\splits\train.jsonl `
  --test-labels data\overnight\final_v2\splits\test.jsonl `
  --out-dir checkpoints\overnight\compression_v2_fair `
  --metrics-out outputs\overnight\compression_v2_fair_metrics.csv `
  --batch-size 128 --latency-repeats 100 `
  --prune-sparsities 0.25 0.5 0.75 `
  --prune-finetune-epochs 2 --prune-lr 0.0001 `
  --device cpu --wandb `
  --wandb-project dlengine-chess-compression
```

Plot regeneration:

```powershell
python -m scripts.plot_pareto `
  --metrics outputs\overnight\final_v2_metrics.csv `
  --out-dir figures\overnight\final_v2

python -m scripts.plot_compression `
  --final-metrics outputs\overnight\final_v2_metrics.csv `
  --compression-metrics outputs\overnight\compression_v2_fair_metrics.csv `
  --out-dir figures\overnight\compression_v2_fair
```

## Main Distillation Results

All metrics below are from the 7,475-position held-out test split.

| Model | Top-1 | Top-3 | Value RMSE | Pearson | Params | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| small CNN 32x5 | **21.32%** | 41.10% | 0.240 | 0.735 | 184,785 | 1.70 ms |
| small CNN 64x5 | 20.62% | **42.66%** | 0.239 | 0.735 | 311,633 | 2.01 ms |
| small CNN 64x3 | 20.40% | 40.78% | 0.231 | 0.755 | 237,649 | 1.21 ms |
| small CNN 32x3 | 20.23% | 38.97% | **0.227** | **0.773** | 166,225 | 1.18 ms |

The standard small CNN family outperformed every tested depthwise configuration in top-1
agreement. The best depthwise top-1 result was 19.38%.

## Fair Compression Results

### Dynamic INT8 quantization

- `small_cnn_32x5`: top-1 unchanged at 21.32%; checkpoint reduced from 0.717 MB to
  0.354 MB (50.6% smaller).
- `small_cnn_64x5`: top-1 unchanged at 20.62%; checkpoint reduced from 1.202 MB to
  0.839 MB (30.2% smaller).
- Dynamic quantization was slower in this measured CPU/PyTorch path. It only quantizes supported
  linear layers, so these convolution-heavy models do not receive a complete INT8 execution path.

### Magnitude pruning with matched training

Each pruned model and its unpruned control received two extra epochs at learning rate `1e-4`.

| Model | Control | 25% sparse | 50% sparse | 75% sparse |
|---|---:|---:|---:|---:|
| small CNN 32x5 top-1 | 22.57% | 21.43% | 21.73% | 18.90% |
| small CNN 64x5 top-1 | 23.71% | **24.19%** | 23.91% | 21.18% |

Interpretation:

- Much of the apparent improvement in the earlier pruning run came from two additional training
  epochs, not pruning itself.
- The 64x5 model tolerates 25-50% sparsity with no meaningful accuracy loss relative to its
  fine-tuned control.
- Both models degrade substantially at 75% sparsity.
- Dense PyTorch checkpoint files remain approximately the same size after unstructured pruning.
  Sparse storage or sparse kernels are required for actual file-size or latency benefits.

## Artifacts

Data and splits:

- `data/overnight/final_v2/`
- `data/overnight/final_v2/splits/train.jsonl`
- `data/overnight/final_v2/splits/val.jsonl`
- `data/overnight/final_v2/splits/test.jsonl`

Metrics:

- `outputs/overnight/final_v2_metrics.csv`
- `outputs/overnight/compression_v2_fair_metrics.csv`

Checkpoints:

- `checkpoints/overnight/final_v2/`
- `checkpoints/overnight/compression_v2_fair/`

Presentation plots:

- `figures/overnight/final_v2/`
- `figures/overnight/compression_v2_fair/`

Poster:

- `poster/poster_final_results.html`

The older `figures/final`, original overnight outputs, and original poster files were not deleted
or overwritten.

## Weights & Biases

Project:

<https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression>

Runs are grouped by distillation, quantization, pruning, and `pruning_control`. Local W&B run logs
are under `wandb/`.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Result: 5 tests passed. The tests cover side-to-move policy targets, illegal-move masking,
deterministic disjoint split indices, and grouped game isolation.

The final architecture CSV has 10 rows, the fair compression CSV has 10 rows, and each plotted
point comes from one of those CSV rows.

## Limits and Recommended Next Experiments

This is a strong single-run comparison, not a final statistical claim. The next work should be:

- [ ] Repeat the best 3-4 configurations across at least three random seeds and report mean plus
  confidence intervals.
- [ ] Add quantization-aware training or static convolution quantization; dynamic quantization
  alone primarily affects the linear heads.
- [ ] Test structured channel/filter pruning, which can reduce dense tensor dimensions and produce
  real latency and storage gains without specialized sparse kernels.
- [ ] Benchmark with fixed CPU thread counts and longer latency warmup/repetition.
- [ ] Measure chess playing strength with fixed-node or fixed-time engine matches.
- [ ] Compare against a simple supervised hard-label baseline to isolate the benefit of soft
  MultiPV distillation.

