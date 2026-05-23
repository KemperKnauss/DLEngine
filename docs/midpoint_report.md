# Mid-Point Milestone Report

## Draft Poster

The draft poster is located at `poster/poster (1).html`, with a text outline in `poster/outline.md`. It includes the refined research question, motivation, methodology, smaller-weight CNN student variants, planned evaluation metrics, and placeholder result visualizations. The poster layout is essentially complete; the remaining work for the final submission is to replace placeholders with the final larger-scale experimental results.

## Progress Summary

Our project investigates how much Stockfish-like chess evaluation behavior can be preserved by smaller-weight CNN student models designed for edge-device inference. We implemented a local teacher-student pipeline that samples games from the Lichess open database, converts PGN games into FEN board positions, and labels each position using a local Stockfish UCI engine. Each board is encoded as an 18-channel 8x8 tensor containing piece planes plus state information such as side to move, castling rights, and en passant availability. The student model uses a compact convolutional backbone with two heads: a policy head that predicts Stockfish-preferred moves and a value head that predicts Stockfish evaluation.

For the midpoint implementation, we trained small CNN and depthwise separable CNN variants using Stockfish MultiPV labels, where the top candidate moves are converted into soft policy targets and centipawn scores are converted into bounded value targets. Our first smoke test extracted 1,399 positions, labeled 998 with Stockfish at depth 6 and MultiPV 5, and trained four model variants. The final experiments will scale the number of positions, increase Stockfish depth, train for more epochs, and compare architectures by Top-1/Top-3 move agreement, value RMSE, value correlation, parameter count, model size, and CPU latency. We will interpret results using Pareto frontier plots to identify models with the best accuracy-efficiency tradeoff.

## Related Works

Silver et al.'s AlphaZero demonstrated that deep neural networks can guide chess play by combining policy/value prediction with search, replacing hand-crafted evaluation features with learned board representations. Our project is related because we also use a policy/value neural architecture for chess positions, but we focus on compressing a small student model rather than training a world-class engine through self-play.

Stockfish is a leading open-source UCI chess engine that combines highly optimized alpha-beta search with a strong NNUE-style evaluation function. It serves as our teacher because it can be run locally to generate reproducible best-move and centipawn labels for large numbers of FEN positions. Our work does not try to replace Stockfish search directly; instead, it studies how well a lightweight neural model can approximate Stockfish's analysis outputs.

Hinton et al.'s knowledge distillation work introduced the idea of training a smaller student model to imitate a stronger teacher's softened output distribution. This is central to our method: Stockfish MultiPV scores are converted into soft move targets so that the student learns not only the best move, but also which alternative moves remain reasonable. MobileNet-style depthwise separable convolutions also inform our architecture comparison because they reduce parameters and computation, making them suitable for edge-device chess evaluation.
