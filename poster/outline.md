# Draft Poster Outline

## Title

Distilling Stockfish into a Lightweight Student Model

## Research Question

How small can a Stockfish-style CNN student model be before it loses useful chess judgment, measured by move agreement, evaluation error, model weights, memory use, and CPU inference latency?

## Motivation

Strong chess analysis is computationally expensive because high-quality results usually depend on deep search. This project studies whether a smaller-weight CNN student can approximate useful parts of Stockfish analysis well enough for constrained devices or lightweight local applications.

## Methodology

[Pipeline Diagram: PGN games -> FEN positions -> Stockfish MultiPV labels -> compact student model -> metrics/plots]

- Sample chess positions from the Lichess open database.
- Convert PGN games into FEN positions.
- Encode each board position as an 18-channel 8x8 tensor.
- Generate Stockfish MultiPV labels with top candidate moves and centipawn evaluations.
- Convert MultiPV scores into soft policy targets.
- Train smaller-weight CNN students with policy and value losses.

## Model Variants

[Architecture Comparison Table Here]

- Baseline compact CNN student.
- Depthwise separable CNN student.
- Width/depth sweeps that reduce model weights and latency.
- Optional final extension: quantized student.

The standard CNN uses regular convolutions that learn spatial patterns and cross-channel piece interactions together. The depthwise separable CNN splits this into a depthwise spatial filter followed by a 1x1 pointwise channel-mixing step, which should reduce weights and computation for edge-device inference.

## Training Loss

The student uses a combined loss:

```text
loss = soft policy cross-entropy + lambda * value MSE
```

The policy term trains the student to match a soft move distribution derived from Stockfish MultiPV scores. The value term trains the student to predict a bounded version of Stockfish's centipawn evaluation.

## Planned Experiments

- Scale the number of Stockfish-labeled Lichess positions.
- Sweep width and depth for standard CNN and depthwise separable CNN students.
- Compare shallow and deeper Stockfish teacher labels.
- Train longer and measure how move agreement and value error change.
- Plot edge-device tradeoffs using parameter count, checkpoint size, and CPU latency.

## Planned Evaluation

[Pareto Frontier: Top-1 agreement vs latency, Top-1 agreement vs parameters, Value RMSE vs latency]

We will compare Stockfish imitation quality against edge-device cost using Top-1/Top-3 move agreement, value RMSE, value correlation, parameter count, checkpoint size, and CPU latency.

## Expected Results

[Results Table Here: model, Top-1, Top-3, value RMSE, correlation, params, latency]

We expect larger compact CNNs to improve agreement with Stockfish at a modest cost increase, while depthwise separable models should reduce model weights and memory. The final analysis will identify which students lie on the Pareto frontier for edge-device use.
