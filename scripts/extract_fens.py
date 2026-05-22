from __future__ import annotations

import argparse
from pathlib import Path

import chess.pgn
from tqdm import tqdm


def extract_fens(pgn_path: Path, output_path: Path, max_games: int | None, ply_stride: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    games = 0

    with pgn_path.open("r", encoding="utf-8", errors="replace") as pgn, output_path.open("w", encoding="utf-8") as out:
        progress = tqdm(total=max_games, desc="games")
        while max_games is None or games < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games += 1
            board = game.board()
            out.write(board.fen() + "\n")
            count += 1
            for ply, move in enumerate(game.mainline_moves(), start=1):
                board.push(move)
                if ply % ply_stride == 0:
                    out.write(board.fen() + "\n")
                    count += 1
            progress.update(1)
        progress.close()

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FEN positions from a PGN file.")
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/fens.txt"))
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--ply-stride", type=int, default=1)
    args = parser.parse_args()

    count = extract_fens(args.pgn, args.out, args.max_games, args.ply_stride)
    print(f"Wrote {count} FEN positions to {args.out}")


if __name__ == "__main__":
    main()

