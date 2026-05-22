# Draft Poster Outline

## Title

Compressing Chess Evaluation Networks for Edge-Device Inference

## Research Question

How much Stockfish-like move selection and board evaluation can compact neural student models preserve while reducing inference cost for low-memory devices?

## Motivation

Strong chess engines are computationally expensive because high-quality analysis usually depends on deep search and/or large learned evaluators. This project studies whether a small neural student can approximate useful parts of Stockfish analysis quickly enough for constrained devices or lightweight applications.

## Methodology

[Pipeline Diagram: PGN games -> FEN positions -> Stockfish MultiPV labels -> compact student model -> metrics/plots]

- Sample chess positions from the Lichess open database.
- Convert PGN games into FEN positions.
- Encode each board position as an 18-channel 8x8 tensor.
- Generate Stockfish MultiPV labels with top candidate moves and centipawn evaluations.
- Convert MultiPV scores into soft policy targets.
- Train compact CNN students with policy and value losses.

## Model Variants

[Architecture Comparison Table Here]

- Baseline compact CNN.
- Depthwise separable CNN.
- Width/depth sweeps for each architecture.
- Optional final extension: quantized student.

## Planned Evaluation

[Pareto Frontier: Top-1 agreement vs latency, Top-1 agreement vs parameters, Value RMSE vs latency]

We will compare Stockfish imitation quality against inference cost using Top-1/Top-3 move agreement, value RMSE, value correlation, parameter count, checkpoint size, and CPU latency.

## Expected Results

[Results Table Here: model, Top-1, Top-3, value RMSE, correlation, params, latency]

We expect larger compact CNNs to improve agreement with Stockfish at a modest cost increase, while depthwise separable models should reduce parameter count and memory. The final analysis will identify which models lie on the Pareto frontier.

## Related Work

[Related Works Summary: AlphaZero, Stockfish/NNUE, knowledge distillation and MobileNet-style efficient CNNs]
