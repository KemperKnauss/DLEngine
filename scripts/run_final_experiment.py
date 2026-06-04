from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PGN = ROOT / "data" / "raw_games" / "lichess_db_standard_rated_2014-07.pgn"
STOCKFISH = ROOT / "tools" / "stockfish-windows-x86-64-avx2.exe"
FENS = ROOT / "data" / "processed" / "final_fens.txt"
LABELS = ROOT / "data" / "labels" / "stockfish_final_labels.jsonl"
METRICS = ROOT / "outputs" / "final_metrics.csv"
FIGURES = ROOT / "figures" / "final"


MODEL_SWEEP = [
    ("small_cnn", 16, 3),
    ("small_cnn", 32, 3),
    ("small_cnn", 64, 3),
    ("small_cnn", 32, 5),
    ("small_cnn", 64, 5),
    ("depthwise_cnn", 16, 3),
    ("depthwise_cnn", 32, 3),
    ("depthwise_cnn", 64, 3),
    ("depthwise_cnn", 32, 5),
    ("depthwise_cnn", 64, 5),
]


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def run(command: list[str]) -> None:
    log("RUN " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT, check=True)


def labels_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def metrics_contains(model_label: str) -> bool:
    if not METRICS.exists():
        return False
    with METRICS.open("r", encoding="utf-8", newline="") as handle:
        return any(row.get("model") == model_label for row in csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the larger final Stockfish distillation experiment.")
    parser.add_argument("--max-games", type=int, default=1600)
    parser.add_argument("--limit-positions", type=int, default=50000)
    parser.add_argument("--stockfish-depth", type=int, default=12)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-relabel", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()

    log("Starting final experiment")
    log(f"Target labels={args.limit_positions}, Stockfish depth={args.stockfish_depth}, MultiPV={args.multipv}")

    if not FENS.exists() or args.force_relabel:
        run(
            [
                PYTHON,
                "scripts/extract_fens.py",
                "--pgn",
                str(PGN),
                "--out",
                str(FENS),
                "--max-games",
                str(args.max_games),
                "--ply-stride",
                "2",
            ]
        )
    else:
        log(f"Skipping FEN extraction; found {FENS}")

    if labels_count(LABELS) < args.limit_positions or args.force_relabel:
        run(
            [
                PYTHON,
                "scripts/label_with_stockfish.py",
                "--fens",
                str(FENS),
                "--out",
                str(LABELS),
                "--stockfish-path",
                str(STOCKFISH),
                "--depth",
                str(args.stockfish_depth),
                "--multipv",
                str(args.multipv),
                "--limit-positions",
                str(args.limit_positions),
            ]
        )
    else:
        log(f"Skipping Stockfish labeling; found {labels_count(LABELS)} labels")

    for model_name, channels, depth in MODEL_SWEEP:
        label = f"{model_name}_{channels}x{depth}_d{args.stockfish_depth}_{args.limit_positions // 1000}k"
        checkpoint_dir = ROOT / "checkpoints" / "final" / label
        best_checkpoint = checkpoint_dir / "best.pt"

        if not best_checkpoint.exists() or args.force_retrain:
            run(
                [
                    PYTHON,
                    "scripts/train_student.py",
                    "--labels",
                    str(LABELS),
                    "--model",
                    model_name,
                    "--channels",
                    str(channels),
                    "--depth",
                    str(depth),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--out-dir",
                    str(checkpoint_dir),
                    "--device",
                    args.device,
                ]
            )
        else:
            log(f"Skipping training; found {best_checkpoint}")

        if not metrics_contains(label):
            run(
                [
                    PYTHON,
                    "scripts/evaluate_student.py",
                    "--checkpoint",
                    str(best_checkpoint),
                    "--labels",
                    str(LABELS),
                    "--metrics-out",
                    str(METRICS),
                    "--model-label",
                    label,
                    "--batch-size",
                    str(args.batch_size),
                    "--latency-repeats",
                    "100",
                    "--device",
                    args.device,
                ]
            )
        else:
            log(f"Skipping evaluation; metrics already contain {label}")

    run([PYTHON, "scripts/plot_pareto.py", "--metrics", str(METRICS), "--out-dir", str(FIGURES)])
    log("Final experiment complete")


if __name__ == "__main__":
    main()
