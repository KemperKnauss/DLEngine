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
```

## Midpoint Deliverables

- Draft poster with complete structure and placeholders for results.
- Progress summary describing architecture, data, preprocessing, planned experiments, and analysis.
- Related works paragraph covering three relevant prior works.
