# Draft Poster Outline

## Title

Compressing Chess Evaluation Networks for Edge-Device Inference

## Research Question

How do architectural modifications to a chess evaluation neural network affect the accuracy-efficiency tradeoff under edge-device compute constraints?

## Motivation

Modern neural chess engines can produce strong evaluations, but their residual-network architectures are often too expensive for low-memory or low-power devices. This project studies whether compact student models can preserve useful evaluation accuracy while reducing inference cost.

## Methodology

[Pipeline Diagram Here]

- Sample chess positions from Lichess and/or Libra-style game databases.
- Encode each board position into neural network input planes.
- Generate Stockfish evaluations as target labels.
- Optionally collect Leela Chess Zero teacher outputs for distillation.
- Train compact CNN students using supervised and distillation losses.

## Model Variants

[Architecture Comparison Here]

- Baseline compact CNN.
- Depthwise separable CNN.
- Distilled depthwise separable CNN.
- Optional: quantized or pruned student.

## Planned Evaluation

[Pareto Frontier Here]

We will compare evaluation accuracy against inference cost using MSE or WDL error, parameter count, FLOPs, latency, and memory footprint.

## Expected Results

[Results Table Here]

We expect smaller models to reduce inference cost substantially, with distillation helping recover part of the accuracy gap between compact students and larger teacher networks.

## Related Work

[Related Works Summary Here]
