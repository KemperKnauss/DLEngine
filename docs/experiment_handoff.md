# Experiment Handoff

## Goal

Complete the DLEngine distillation/compression roadmap: implement ZST dataset streaming, held-out splits, wandb online logging, corrected distillation targets, quantization, pruning, final metrics/plots, and presentation-ready handoff artifacts.

Status: completed for the requested 25k-label/depth-10 overnight run.

## Current Setup

- Branch: `codex/experiment-roadmap`
- Dataset primary: `C:\Users\danie\Downloads\lichess_db_standard_rated_2026-05.pgn.zst`
- Dataset fallback: `C:\Users\danie\Downloads\lichess_db_standard_rated_2026-05.pgn`
- Stockfish: `C:\Users\danie\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe`
- Stockfish version reported by UCI: Stockfish 18
- wandb project: `https://wandb.ai/danielqian6-university-of-washington/dlengine-chess-compression`

## Implemented

- Corrected policy distillation targets to use side-to-move perspective.
- Added `.pgn.zst` streaming support for FEN extraction.
- Added deterministic train/val/test split generation.
- Added explicit validation labels and wandb logging to student training.
- Added dynamic quantization and unstructured pruning experiment support.
- Added smoke/final/compression orchestration under overnight artifact paths.
- Added distillation and compression plots.
- Added sanity tests for policy targets and split disjointness.

## Completed Runs

Smoke run:

- Extracted 500 FEN positions from 18 games.
- Labeled 299 positions at Stockfish depth 6, MultiPV 5.
- Trained `small_cnn_16x3` for 3 epochs.
- Evaluated on held-out test split and generated smoke Pareto plots.

Final distillation run:

- Extracted 50,000 FEN positions from 1,468 games.
- Attempted 25,000 Stockfish labels at depth 10, MultiPV 5.
- Wrote 24,907 usable labels.
- Split into 19,927 train, 2,490 validation, and 2,490 test positions.
- Trained 10 student models for 10 epochs each on CPU.
- Evaluated every model only on held-out test labels.

Compression run:

- Selected the top two distillation checkpoints by Top-1 agreement:
  - `small_cnn_64x5_d10_25k_final`
  - `small_cnn_32x5_d10_25k_final`
- Applied dynamic quantization to both.
- Applied unstructured pruning at 25%, 50%, and 75% sparsity to both.
- Fine-tuned pruned models and evaluated all compressed variants on the held-out test split.

## Key Results

Best distillation model by Top-1:

| Model | Top-1 | Top-3 | Value RMSE | Value Pearson | Params | MB | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `small_cnn_64x5_d10_25k_final` | 0.108 | 0.270 | 0.224 | 0.770 | 311,633 | 1.202 | 1.854 |

Best distillation model by value RMSE:

| Model | Top-1 | Top-3 | Value RMSE | Value Pearson | Latency ms |
|---|---:|---:|---:|---:|---:|
| `small_cnn_64x3_d10_25k_final` | 0.081 | 0.184 | 0.210 | 0.803 | 1.182 |

Best compression Top-1 observed:

| Model | Compression | Top-1 | Top-3 | Value RMSE | Sparsity | MB | Latency ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `small_cnn_64x5_d10_25k_final_pruned_50` | unstructured pruning | 0.156 | 0.276 | 0.245 | 0.498 | 1.206 | 3.094 |
| `small_cnn_64x5_d10_25k_final_pruned_25` | unstructured pruning | 0.135 | 0.290 | 0.213 | 0.249 | 1.206 | 2.564 |

Quantization result note:

- Dynamic quantization preserved accuracy for both selected models.
- It reduced checkpoint size substantially for `small_cnn_32x5` and `small_cnn_64x5`.
- It did not improve measured CPU latency in this run, which is plausible because the model is convolution-heavy and dynamic quantization primarily affects linear layers.

Pruning result note:

- 25% and 50% pruning improved Top-1 on the two selected students after fine-tuning.
- Unstructured pruning did not reduce dense checkpoint size or latency as much as structured pruning would; report it as sparsity/compression pressure, not guaranteed hardware speedup.

## Artifact Paths

- Run log: `outputs/overnight/overnight_run.log`
- Smoke metrics: `outputs/overnight/smoke_metrics.csv`
- Final metrics: `outputs/overnight/final_metrics.csv`
- Compression metrics: `outputs/overnight/compression_metrics.csv`
- Final plots: `figures/overnight/final/`
- Compression plots: `figures/overnight/compression/`
- Checkpoints: `checkpoints/overnight/`
- Data/splits: `data/overnight/`

Existing `figures/final/` plots are preliminary/unverified because the corresponding metrics/checkpoints are not present in the repo.

## Reproduction Commands

Smoke only:

```powershell
python scripts/run_overnight_experiment.py --skip-final --skip-compression --device cpu
```

Full overnight default:

```powershell
python scripts/run_overnight_experiment.py --target-labels 25000 --stockfish-depth 10 --multipv 5 --epochs 10 --batch-size 128 --device cpu
```

Regenerate compression plots:

```powershell
python scripts/plot_compression.py --final-metrics outputs/overnight/final_metrics.csv --compression-metrics outputs/overnight/compression_metrics.csv --out-dir figures/overnight/compression
```

## Verification

- `python -m py_compile src\chess_student\data.py scripts\extract_fens.py scripts\label_with_stockfish.py scripts\split_labels.py scripts\train_student.py scripts\compress_students.py scripts\run_overnight_experiment.py scripts\plot_compression.py`
- `python -m unittest discover -s tests`
- Final metrics contain 10 rows, one per planned distillation model.
- Compression metrics contain 8 rows: 2 quantized models and 6 pruned models.
- Final and compression plots were generated from the local CSV metrics.

## Next Recommendations

- Use `small_cnn_64x5_d10_25k_final` as the main best-move imitation model.
- Use `small_cnn_64x3_d10_25k_final` when emphasizing value-head accuracy.
- Present dynamic quantization as a size-saving result, not a speed win.
- Present unstructured pruning as an accuracy/sparsity tradeoff, and mention that structured pruning would be the next step for real latency reduction.
