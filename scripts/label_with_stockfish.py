from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import chess.engine
from tqdm import tqdm


def score_to_centipawns(score: chess.engine.PovScore) -> int:
    white_score = score.white()
    if white_score.is_mate():
        mate = white_score.mate()
        return 100000 if mate and mate > 0 else -100000
    cp = white_score.score()
    return int(cp) if cp is not None else 0


def label_positions(
    fen_path: Path,
    output_path: Path,
    stockfish_path: str,
    depth: int,
    multipv: int,
    limit_positions: int | None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with fen_path.open("r", encoding="utf-8") as handle:
        fens = [line.strip() for line in handle if line.strip()]
    if limit_positions is not None:
        fens = fens[:limit_positions]

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    written = 0
    try:
        with output_path.open("w", encoding="utf-8") as out:
            for fen in tqdm(fens, desc="stockfish"):
                board = chess.Board(fen)
                infos = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
                if isinstance(infos, dict):
                    infos = [infos]

                labels = []
                for info in infos:
                    pv = info.get("pv")
                    if not pv:
                        continue
                    labels.append({"move": pv[0].uci(), "eval_cp": score_to_centipawns(info["score"])})

                if labels:
                    out.write(json.dumps({"fen": fen, "labels": labels}) + "\n")
                    written += 1
    finally:
        engine.quit()

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stockfish MultiPV labels for FEN positions.")
    parser.add_argument("--fens", type=Path, default=Path("data/processed/fens.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/labels/stockfish_labels.jsonl"))
    parser.add_argument("--stockfish-path", required=True, help="Path to the local Stockfish executable.")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--limit-positions", type=int, default=None)
    args = parser.parse_args()

    written = label_positions(args.fens, args.out, args.stockfish_path, args.depth, args.multipv, args.limit_positions)
    print(f"Wrote {written} labeled positions to {args.out}")


if __name__ == "__main__":
    main()
