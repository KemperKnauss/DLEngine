from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def split_indices(total: int, val_fraction: float, test_fraction: float, seed: int) -> dict[str, list[int]]:
    if total < 3:
        raise ValueError("Need at least 3 rows for train/val/test splits.")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than 1.")

    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    test_size = max(1, int(total * test_fraction))
    val_size = max(1, int(total * val_fraction))
    train_size = total - val_size - test_size
    if train_size <= 0:
        raise ValueError("Split fractions leave no training rows.")

    return {
        "train": indices[:train_size],
        "val": indices[train_size : train_size + val_size],
        "test": indices[train_size + val_size :],
    }


def write_rows(rows: list[str], indices: list[int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index in indices:
            handle.write(rows[index])


def grouped_split_indices(
    rows: list[str],
    group_key: str,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, line in enumerate(rows):
        row = json.loads(line)
        if group_key not in row:
            raise ValueError(f"Row {index} is missing group key {group_key!r}.")
        groups[str(row[group_key])].append(index)

    group_names = list(groups)
    random.Random(seed).shuffle(group_names)
    target_test = len(rows) * test_fraction
    target_val = len(rows) * val_fraction
    result = {"train": [], "val": [], "test": []}
    for group_name in group_names:
        if len(result["test"]) < target_test:
            destination = "test"
        elif len(result["val"]) < target_val:
            destination = "val"
        else:
            destination = "train"
        result[destination].extend(groups[group_name])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test label splits.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--group-key", default=None)
    args = parser.parse_args()

    with args.labels.open("r", encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]

    if args.group_key:
        splits = grouped_split_indices(rows, args.group_key, args.val_fraction, args.test_fraction, args.seed)
    else:
        splits = split_indices(len(rows), args.val_fraction, args.test_fraction, args.seed)
    paths = {
        "train": args.out_dir / "train.jsonl",
        "val": args.out_dir / "val.jsonl",
        "test": args.out_dir / "test.jsonl",
    }
    for name, indices in splits.items():
        write_rows(rows, indices, paths[name])

    metadata = {
        "source_labels": str(args.labels),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "total_rows": len(rows),
        "group_key": args.group_key,
        "splits": {
            name: {
                "path": str(paths[name]),
                "rows": len(indices),
                "indices": indices,
            }
            for name, indices in splits.items()
        },
    }
    metadata_path = args.out_dir / "split_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote splits to {args.out_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
