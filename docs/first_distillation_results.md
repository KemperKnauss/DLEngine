# First Stockfish Distillation Run

This run verifies the full local workflow:

```text
Lichess PGN -> FEN positions -> Stockfish MultiPV labels -> student training -> evaluation -> Pareto plots
```

## Data and Teacher Settings

- PGN source: `data/raw_games/lichess_db_standard_rated_2014-07.pgn`
- Games sampled: 40
- FEN positions extracted: 1,399
- Stockfish labels written: 998
- Stockfish depth: 6
- MultiPV: 5

This is intentionally a small smoke-test run, not a final-quality training run.

## Student Training Settings

- Epochs: 3
- Batch size: 64
- Device: CPU
- Models: small CNN and depthwise separable CNN
- Channel sizes: 16 and 32

## Results

| Model | Top-1 | Top-3 | Value RMSE | Value Pearson | Params | Model MB | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_cnn_16x3 | 0.015 | 0.029 | 0.233 | 0.635 | 144,337 | 0.559 | 0.277 |
| small_cnn_32x3 | 0.027 | 0.044 | 0.227 | 0.700 | 166,225 | 0.642 | 0.318 |
| depthwise_cnn_16x3 | 0.017 | 0.025 | 0.231 | 0.672 | 140,529 | 0.544 | 0.470 |
| depthwise_cnn_32x3 | 0.002 | 0.016 | 0.235 | 0.667 | 150,417 | 0.583 | 0.587 |

The value head already learns a meaningful relationship with Stockfish evaluation. Move prediction remains weak, which is expected with only 998 shallow-depth labels and three epochs. The next research run should increase positions, train longer, and use a held-out evaluation split that is separate from the training labels.

## Generated Figures

- `figures/pareto_top1_vs_latency.png`
- `figures/pareto_top1_vs_params.png`
- `figures/pareto_value_rmse_vs_latency.png`

