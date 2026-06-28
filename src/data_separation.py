"""Separate processed PPI rows into positive and negative pair files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_protein_pair(text: str) -> tuple[str, str]:
    parts = [part.strip() for part in text.split(",", maxsplit=1)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Could not split protein pair: {text!r}")
    return parts[0], parts[1]


def separate_data_file(input_file: Path, positive_output: Path, negative_output: Path) -> None:
    positive_rows: list[list[str]] = []
    negative_rows: list[list[str]] = []

    with input_file.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"text", "label"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_file} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            try:
                protein_a, protein_b = split_protein_pair(row["text"])
            except ValueError as exc:
                raise ValueError(f"Invalid protein pair in {input_file}:{row_number}") from exc

            target = negative_rows if row["label"] == "0" else positive_rows
            target.append([protein_a, protein_b, row["label"]])

    write_pair_rows(positive_output, positive_rows)
    write_pair_rows(negative_output, negative_rows)


def write_pair_rows(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["ProtA", "ProtB", "label"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--positive-output", type=Path, default=Path("output_positive.csv"))
    parser.add_argument("--negative-output", type=Path, default=Path("output_negative.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    separate_data_file(args.input, args.positive_output, args.negative_output)
    print(f"Wrote positive rows to {args.positive_output}")
    print(f"Wrote negative rows to {args.negative_output}")


if __name__ == "__main__":
    main()

