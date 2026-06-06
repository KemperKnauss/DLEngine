from __future__ import annotations

import argparse
import csv
import os
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

DEFAULT_DATASET_ZST = Path(r"C:\Users\danie\Downloads\lichess_db_standard_rated_2026-05.pgn.zst")
DEFAULT_DATASET_PGN = Path(r"C:\Users\danie\Downloads\lichess_db_standard_rated_2026-05.pgn")
DEFAULT_STOCKFISH = Path(
    r"C:\Users\danie\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
)

DATA_ROOT = ROOT / "data" / "overnight"
OUTPUT_ROOT = ROOT / "outputs" / "overnight"
CHECKPOINT_ROOT = ROOT / "checkpoints" / "overnight"
FIGURE_ROOT = ROOT / "figures" / "overnight"

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


def run(command: list[str], log_path: Path) -> None:
    log("RUN " + " ".join(str(part) for part in command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("RUN " + " ".join(str(part) for part in command) + "\n")
        handle.flush()
        env = os.environ.copy()
        src_path = str(ROOT / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def select_dataset(primary: Path, fallback: Path) -> Path:
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Neither dataset path exists: {primary} or {fallback}")


def metrics_contains(path: Path, model_label: str) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as handle:
        return any(row.get("model") == model_label for row in csv.DictReader(handle))


def best_checkpoints(metrics_path: Path, top_n: int) -> list[Path]:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("checkpoint")]
    rows.sort(key=lambda row: (float(row.get("top1") or 0.0), -float(row.get("value_rmse") or 999.0)), reverse=True)
    checkpoints: list[Path] = []
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        if checkpoint.exists() and checkpoint not in checkpoints:
            checkpoints.append(checkpoint)
        if len(checkpoints) >= top_n:
            break
    return checkpoints


def write_run_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def run_pipeline(
    *,
    name: str,
    dataset: Path,
    stockfish: Path,
    max_games: int,
    max_positions: int,
    label_limit: int,
    stockfish_depth: int,
    multipv: int,
    ply_stride: int,
    epochs: int,
    batch_size: int,
    device: str,
    models: list[tuple[str, int, int]],
    wandb: bool,
    grouped_split: bool,
    log_path: Path,
) -> Path:
    run_root = DATA_ROOT / name
    fens = run_root / "fens.txt"
    game_ids = run_root / "game_ids.txt"
    labels = run_root / "stockfish_labels.jsonl"
    splits = run_root / "splits"
    metrics = OUTPUT_ROOT / f"{name}_metrics.csv"
    figures = FIGURE_ROOT / name

    write_run_metadata(
        OUTPUT_ROOT / f"{name}_run_metadata.json",
        {
            "dataset": dataset,
            "stockfish": stockfish,
            "max_games": max_games,
            "max_positions": max_positions,
            "label_limit": label_limit,
            "stockfish_depth": stockfish_depth,
            "multipv": multipv,
            "ply_stride": ply_stride,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "models": models,
        },
    )

    if count_rows(fens) < max_positions:
        run(
            [
                PYTHON,
                "scripts/extract_fens.py",
                "--pgn",
                str(dataset),
                "--out",
                str(fens),
                "--game-ids-out",
                str(game_ids),
                "--metadata-out",
                str(run_root / "fens.metadata.json"),
                "--max-games",
                str(max_games),
                "--ply-stride",
                str(ply_stride),
                "--max-positions",
                str(max_positions),
            ],
            log_path,
        )
    else:
        log(f"Skipping extraction for {name}; found {count_rows(fens)} FEN rows")

    existing_labels = count_rows(labels)
    if existing_labels < int(label_limit * 0.95):
        run(
            [
                PYTHON,
                "scripts/label_with_stockfish.py",
                "--fens",
                str(fens),
                "--game-ids",
                str(game_ids),
                "--out",
                str(labels),
                "--metadata-out",
                str(run_root / "stockfish_labels.metadata.json"),
                "--stockfish-path",
                str(stockfish),
                "--depth",
                str(stockfish_depth),
                "--multipv",
                str(multipv),
                "--limit-positions",
                str(label_limit),
            ],
            log_path,
        )
    else:
        log(f"Skipping labeling for {name}; found {existing_labels} labels")

    if count_rows(splits / "train.jsonl") == 0:
        split_command = [
                PYTHON,
                "scripts/split_labels.py",
                "--labels",
                str(labels),
                "--out-dir",
                str(splits),
                "--val-fraction",
                "0.1",
                "--test-fraction",
                "0.1",
                "--seed",
                "7",
            ]
        if grouped_split:
            split_command.extend(["--group-key", "game_id"])
        run(split_command, log_path)

    for model_name, channels, depth in models:
        label = f"{model_name}_{channels}x{depth}_d{stockfish_depth}_{label_limit // 1000}k_{name}"
        checkpoint_dir = CHECKPOINT_ROOT / name / label
        best_checkpoint = checkpoint_dir / "best.pt"
        if not best_checkpoint.exists():
            command = [
                PYTHON,
                "scripts/train_student.py",
                "--labels",
                str(splits / "train.jsonl"),
                "--val-labels",
                str(splits / "val.jsonl"),
                "--model",
                model_name,
                "--channels",
                str(channels),
                "--depth",
                str(depth),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(batch_size),
                "--out-dir",
                str(checkpoint_dir),
                "--device",
                device,
                "--wandb-project",
                "dlengine-chess-compression",
                "--wandb-group",
                "smoke" if name == "smoke" else "distillation",
                "--wandb-run-name",
                label,
            ]
            if wandb:
                command.append("--wandb")
            run(command, log_path)
        else:
            log(f"Skipping training; found {best_checkpoint}")

        if not metrics_contains(metrics, label):
            run(
                [
                    PYTHON,
                    "scripts/evaluate_student.py",
                    "--checkpoint",
                    str(best_checkpoint),
                    "--labels",
                    str(splits / "test.jsonl"),
                    "--metrics-out",
                    str(metrics),
                    "--model-label",
                    label,
                    "--batch-size",
                    str(batch_size),
                    "--latency-repeats",
                    "100",
                    "--device",
                    device,
                ],
                log_path,
            )
        else:
            log(f"Skipping evaluation; metrics already contain {label}")

    run([PYTHON, "scripts/plot_pareto.py", "--metrics", str(metrics), "--out-dir", str(figures)], log_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke, final distillation, and compression experiments.")
    parser.add_argument("--dataset-zst", type=Path, default=DEFAULT_DATASET_ZST)
    parser.add_argument("--dataset-pgn", type=Path, default=DEFAULT_DATASET_PGN)
    parser.add_argument("--stockfish-path", type=Path, default=DEFAULT_STOCKFISH)
    parser.add_argument("--target-labels", type=int, default=25000)
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--run-suffix", default="")
    parser.add_argument("--grouped-split", action="store_true")
    args = parser.parse_args()

    dataset = select_dataset(args.dataset_zst, args.dataset_pgn)
    if not args.stockfish_path.exists():
        raise FileNotFoundError(f"Stockfish executable not found: {args.stockfish_path}")

    log_path = OUTPUT_ROOT / "overnight_run.log"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke:
        run_pipeline(
            name="smoke",
            dataset=dataset,
            stockfish=args.stockfish_path,
            max_games=50,
            max_positions=500,
            label_limit=300,
            stockfish_depth=6,
            multipv=args.multipv,
            ply_stride=2,
            epochs=3,
            batch_size=64,
            device=args.device,
            models=[("small_cnn", 16, 3)],
            wandb=not args.no_wandb,
            grouped_split=args.grouped_split,
            log_path=log_path,
        )

    final_name = f"final{args.run_suffix}"
    final_metrics = OUTPUT_ROOT / f"{final_name}_metrics.csv"
    if not args.skip_final:
        final_metrics = run_pipeline(
            name=final_name,
            dataset=dataset,
            stockfish=args.stockfish_path,
            max_games=4000,
            max_positions=args.target_labels * 2,
            label_limit=args.target_labels,
            stockfish_depth=args.stockfish_depth,
            multipv=args.multipv,
            ply_stride=2,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            models=MODEL_SWEEP,
            wandb=not args.no_wandb,
            grouped_split=args.grouped_split,
            log_path=log_path,
        )

    if not args.skip_compression:
        checkpoints = best_checkpoints(final_metrics, top_n=2)
        if not checkpoints:
            raise RuntimeError(f"No best checkpoints found from {final_metrics}")
        compression_name = f"compression{args.run_suffix}"
        command = [
            PYTHON,
            "scripts/compress_students.py",
            "--checkpoints",
            *[str(path) for path in checkpoints],
            "--train-labels",
            str(DATA_ROOT / final_name / "splits" / "train.jsonl"),
            "--test-labels",
            str(DATA_ROOT / final_name / "splits" / "test.jsonl"),
            "--out-dir",
            str(CHECKPOINT_ROOT / compression_name),
            "--metrics-out",
            str(OUTPUT_ROOT / f"{compression_name}_metrics.csv"),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--wandb-project",
            "dlengine-chess-compression",
        ]
        if not args.no_wandb:
            command.append("--wandb")
        run(command, log_path)
        run(
            [
                PYTHON,
                "scripts/plot_compression.py",
                "--final-metrics",
                str(final_metrics),
                "--compression-metrics",
                str(OUTPUT_ROOT / f"{compression_name}_metrics.csv"),
                "--out-dir",
                str(FIGURE_ROOT / compression_name),
            ],
            log_path,
        )

    log("Overnight experiment workflow complete")


if __name__ == "__main__":
    main()
