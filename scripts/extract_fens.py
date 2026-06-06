from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Iterator, TextIO

import chess.pgn
from tqdm import tqdm


@contextlib.contextmanager
def open_pgn_text(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("Reading .pgn.zst files requires `pip install zstandard`.") from exc

        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                yield text
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle


def extract_fens(
    pgn_path: Path,
    output_path: Path,
    game_ids_path: Path | None,
    max_games: int | None,
    ply_stride: int,
    max_positions: int | None,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    games = 0

    game_ids_handle = game_ids_path.open("w", encoding="utf-8") if game_ids_path else contextlib.nullcontext()
    with open_pgn_text(pgn_path) as pgn, output_path.open("w", encoding="utf-8") as out, game_ids_handle as game_ids:
        progress = tqdm(total=max_games, desc="games")
        while max_games is None or games < max_games:
            if max_positions is not None and count >= max_positions:
                break
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games += 1
            board = game.board()
            out.write(board.fen() + "\n")
            if game_ids:
                game_ids.write(f"{games - 1}\n")
            count += 1
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if max_positions is not None and count >= max_positions:
                    break
                board.push(move)
                if ply % ply_stride == 0:
                    out.write(board.fen() + "\n")
                    if game_ids:
                        game_ids.write(f"{games - 1}\n")
                    count += 1
            progress.update(1)
        progress.close()

    return count, games


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FEN positions from a PGN file.")
    parser.add_argument("--pgn", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/processed/fens.txt"))
    parser.add_argument("--game-ids-out", type=Path, default=None)
    parser.add_argument("--metadata-out", type=Path, default=None)
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--ply-stride", type=int, default=1)
    parser.add_argument("--max-positions", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.game_ids_out:
            args.game_ids_out.parent.mkdir(parents=True, exist_ok=True)
        count, games = extract_fens(
            args.pgn,
            args.out,
            args.game_ids_out,
            args.max_games,
            args.ply_stride,
            args.max_positions,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise
    print(f"Wrote {count} FEN positions from {games} games to {args.out}")
    metadata_path = args.metadata_out or args.out.with_suffix(".metadata.json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pgn": str(args.pgn),
                "out": str(args.out),
                "game_ids_out": str(args.game_ids_out) if args.game_ids_out else None,
                "max_games": args.max_games,
                "ply_stride": args.ply_stride,
                "max_positions": args.max_positions,
                "positions": count,
                "games": games,
            },
            handle,
            indent=2,
        )
    print(f"Wrote extraction metadata to {metadata_path}")


if __name__ == "__main__":
    main()

