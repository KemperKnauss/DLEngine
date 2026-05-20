# Project Plan

## Midpoint Scope

For the first version, we will focus on a complete draft of the research poster and supporting text. The poster should look structurally finished, with placeholders where final experimental results will go.

## Working Hypothesis

Depthwise separable convolutional student models should reduce inference cost substantially compared with larger residual-network-style chess evaluators. Knowledge distillation from a stronger teacher may help recover some accuracy lost by shrinking the architecture.

## Planned Comparison

- Small conventional CNN baseline.
- Depthwise separable CNN student.
- Depthwise separable CNN student with teacher distillation.
- Optional later extension: quantized or pruned student.

## Planned Metrics

- Evaluation error against Stockfish labels.
- Agreement with teacher evaluation direction or value distribution.
- Parameter count.
- FLOPs or MACs.
- Batch-1 CPU latency.
- Peak memory or approximate model size.

## Immediate Tasks

1. Draft poster outline.
2. Write progress summary.
3. Write related works paragraph.
4. Add placeholder diagrams for the data/model/evaluation pipeline.
