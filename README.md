# Chess Evaluation Compression

This project studies how architectural changes to chess evaluation neural networks affect the accuracy-efficiency tradeoff under edge-device constraints.

The current goal is a midpoint deliverable: a draft poster, a short progress summary, and a related works section. Experimental code will grow gradually from this structure rather than being scaffolded all at once.

## Research Question

How do lightweight neural network architectures and compression strategies preserve chess evaluation accuracy while reducing inference cost for low-memory edge devices?

## Initial Direction

We plan to train compact student models for chess position evaluation using:

- Stockfish evaluations as objective target labels.
- Leela Chess Zero outputs as optional soft teacher targets.
- Compact CNN architectures, especially depthwise separable convolutions.
- Evaluation metrics that compare accuracy against latency, parameter count, memory use, and FLOPs.

## Repository Layout

```text
docs/       Written midpoint materials and planning notes.
poster/     Draft poster source and poster-specific assets.
src/        Small code modules as experiments begin.
figures/    Diagrams, plots, and result placeholders.
data/       Local datasets and generated labels; large files are ignored.
scripts/    Experiment commands for FEN extraction, Stockfish labeling, training, evaluation, and plots.
```

## Stockfish Distillation Pipeline

The repo includes a runnable teacher/student workflow:

1. Extract FEN positions from PGN files.
2. Run local Stockfish MultiPV analysis to create soft teacher labels.
3. Train a compact CNN or depthwise separable CNN student with policy and value heads.
4. Evaluate Stockfish move agreement, value error, parameter count, model size, and latency.
5. Plot Pareto-frontier graphs for accuracy versus efficiency.

See [docs/stockfish_distillation_workflow.md](docs/stockfish_distillation_workflow.md) for commands.

## Overnight Experiment Handoff

The current distillation + compression experiment implementation and results are summarized in
[docs/experiment_handoff.md](docs/experiment_handoff.md). It records the streamed Lichess dataset,
Stockfish settings, held-out split sizes, wandb project, final metrics, compression results, plots,
and resume commands for another agent or human.

The corrected 75k-label experiment, including legal-move masking, game-grouped splits, matched
pruning controls, final plots, and the results poster, is documented in
[docs/experiment_v2_handoff.md](docs/experiment_v2_handoff.md).

The controlled comparison of distillation, dynamic INT8 quantization, and magnitude pruning is
specified in [docs/compression_comparison_plan.md](docs/compression_comparison_plan.md), with final
metrics, W&B links, poster assets, interpretation, and resume instructions in
[docs/compression_comparison_handoff.md](docs/compression_comparison_handoff.md).

The final scaled eight-seed comparison, including validation-selected distillation settings,
confidence intervals, repeated latency measurements, and the revised three poster assets, is
documented in
[docs/compression_comparison_v4_handoff.md](docs/compression_comparison_v4_handoff.md).

## Midpoint Deliverables

- Draft poster with complete structure and placeholders for results.
- Progress summary describing architecture, data, preprocessing, planned experiments, and analysis.
- Related works paragraph covering three relevant prior works.
