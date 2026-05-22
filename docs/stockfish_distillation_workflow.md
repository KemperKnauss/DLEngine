# Stockfish Distillation Workflow

This project now has a complete local workflow for training a lightweight neural student from Stockfish teacher labels.

## Setup

Create an environment and install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell, activation is:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Install or download Stockfish separately, then keep the executable path handy. On WSL/Ubuntu this is often `/usr/games/stockfish`; on Windows it may be a downloaded `.exe`.

## 1. Extract Positions From PGN

Download a small PGN sample first, then extract FEN positions:

```bash
python scripts/extract_fens.py \
  --pgn data/raw/sample_games.pgn \
  --out data/processed/fens.txt \
  --max-games 100 \
  --ply-stride 1
```

## 2. Generate Stockfish Labels

Use MultiPV so the student sees several reasonable teacher moves, not only one hard target:

```bash
python scripts/label_with_stockfish.py \
  --fens data/processed/fens.txt \
  --out data/labels/stockfish_labels.jsonl \
  --stockfish-path /usr/games/stockfish \
  --depth 10 \
  --multipv 5
```

For a first smoke test, add `--limit-positions 200`.

## 3. Train Student Models

Train a conventional CNN:

```bash
python scripts/train_student.py \
  --labels data/labels/stockfish_labels.jsonl \
  --model small_cnn \
  --channels 64 \
  --depth 4 \
  --out-dir checkpoints/small_cnn
```

Train a depthwise separable CNN:

```bash
python scripts/train_student.py \
  --labels data/labels/stockfish_labels.jsonl \
  --model depthwise_cnn \
  --channels 64 \
  --depth 4 \
  --out-dir checkpoints/depthwise_cnn
```

The model has a shared convolutional trunk, a policy head over UCI-style actions, and a value head trained against Stockfish centipawn evaluations mapped through `tanh(eval_cp / 1000)`.

## 4. Evaluate Models

Append each model's result to a single metrics file:

```bash
python scripts/evaluate_student.py \
  --checkpoint checkpoints/small_cnn/best.pt \
  --labels data/labels/stockfish_labels.jsonl \
  --metrics-out outputs/metrics.csv \
  --model-label small_cnn_64x4

python scripts/evaluate_student.py \
  --checkpoint checkpoints/depthwise_cnn/best.pt \
  --labels data/labels/stockfish_labels.jsonl \
  --metrics-out outputs/metrics.csv \
  --model-label depthwise_cnn_64x4
```

The metrics include top-1 and top-3 Stockfish move agreement, value RMSE, value correlation, parameter count, checkpoint size, and batch-1 latency.

## 5. Plot Pareto Frontiers

```bash
python scripts/plot_pareto.py \
  --metrics outputs/metrics.csv \
  --out-dir figures
```

This writes:

- `figures/pareto_top1_vs_latency.png`
- `figures/pareto_top1_vs_params.png`
- `figures/pareto_value_rmse_vs_latency.png`

The frontier marks models that are not dominated by another model with both lower cost and equal-or-better accuracy.

