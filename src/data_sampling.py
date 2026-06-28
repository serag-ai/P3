"""Balance binary classification CSV files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Literal

SamplingStrategy = Literal["under", "over", "scale"]


def load_labeled_rows(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    """Return negative and positive rows from a CSV with text/label columns."""

    negative_rows: list[list[str]] = []
    positive_rows: list[list[str]] = []

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"text", "label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            target = negative_rows if row["label"] == "0" else positive_rows
            target.append([row["text"], row["label"]])

    return negative_rows, positive_rows


def balance_rows(
    negative_rows: list[list[str]],
    positive_rows: list[list[str]],
    *,
    strategy: SamplingStrategy,
    seed: int = 42,
    scale: int = 5,
) -> list[list[str]]:
    """Balance rows according to the selected strategy."""

    if not negative_rows or not positive_rows:
        raise ValueError("Both classes must contain at least one row.")

    rng = random.Random(seed)
    minority_count = min(len(negative_rows), len(positive_rows))
    majority_count = max(len(negative_rows), len(positive_rows))
    majority_rows = negative_rows if len(negative_rows) >= len(positive_rows) else positive_rows
    minority_rows = positive_rows if len(negative_rows) >= len(positive_rows) else negative_rows

    if strategy == "under":
        sampled_negative = rng.sample(negative_rows, minority_count)
        sampled_positive = rng.sample(positive_rows, minority_count)
        rows = sampled_negative + sampled_positive
    elif strategy == "over":
        rows = majority_rows + [rng.choice(minority_rows) for _ in range(majority_count)]
    elif strategy == "scale":
        target_negative_count = min(len(negative_rows), scale * len(positive_rows))
        rows = rng.sample(negative_rows, target_negative_count) + positive_rows
    else:
        raise ValueError(f"Unsupported sampling strategy: {strategy}")

    if len(rows) > majority_count * 2 and strategy != "over":
        raise RuntimeError("Unexpectedly produced too many sampled rows.")

    rng.shuffle(rows)
    return rows


def write_labeled_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def sample_file(
    input_file: Path,
    output_file: Path,
    *,
    strategy: SamplingStrategy,
    seed: int = 42,
    scale: int = 5,
) -> None:
    negative_rows, positive_rows = load_labeled_rows(input_file)
    rows = balance_rows(negative_rows, positive_rows, strategy=strategy, seed=seed, scale=scale)
    write_labeled_rows(output_file, rows)
    print(
        f"Wrote {len(rows)} rows to {output_file} "
        f"(positive={len(positive_rows)}, negative={len(negative_rows)})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=["under", "over", "scale"], default="under")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_file(
        args.input,
        args.output,
        strategy=args.strategy,
        seed=args.seed,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
