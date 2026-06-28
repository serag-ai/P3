"""Build train/eval/test CSV files for protein-protein interaction classification."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_FILES = (
    "HP11_L1_V1_20230625.csv",
    "HP11_L1_V1_20230625_5m.csv",
    "HP11_L1_V2_20230625_2m.csv",
    "HP11_L1_V2_20230625_5m.csv",
    "HP11_L1_V3_20230625_2m.csv",
    "HP11_L1_V3_20230625_5m.csv",
    "HP12_L1_V1_20230625_2m.csv",
    "HP12_L1_V1_20230625_5m.csv",
    "HP12_L1_V2_20230625_2m.csv",
    "HP12_L1_V2_20230625_5m.csv",
    "HP12_L1_V3_20230625_2m.csv",
    "HP12_L1_V3_20230625_5m.csv",
    "HP22_L1_V1_20230625_2m.csv",
    "HP22_L1_V1_20230625_5m.csv",
    "HP22_L1_V2_20230625_2m.csv",
    "HP22_L1_V2_20230625_5m.csv",
    "HP22_L1_V3_20230625_2m.csv",
    "HP22_L1_V3_20230625_5m.csv",
)

PROTEIN_ID_PREFIXES = ("NP", "YP", "LG", "Ga")


@dataclass(frozen=True)
class ProcessedInteraction:
    """One model-ready interaction row."""

    text: str
    fdr: float
    log_fc: float
    label: int

    def as_csv_row(self) -> list[str | float | int]:
        return [self.text, self.fdr, self.log_fc, self.label]


def spaced_sequence(sequence: str) -> str:
    """Format an amino-acid sequence as space-separated tokens."""

    return " ".join(sequence.strip())


def load_sequences(path: Path) -> dict[str, str]:
    """Load sequence lookup data keyed by fragment start."""

    sequences: dict[str, str] = {}
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"FragmentStart", "SequenceAA"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            fragment = row["FragmentStart"].strip()
            sequence = row["SequenceAA"].strip()
            if fragment and sequence:
                sequences[fragment] = sequence

    return sequences


def split_gene_pair(genes: str) -> tuple[str, str] | None:
    """Split a raw gene-pair value into two sequence lookup keys."""

    for prefix in PROTEIN_ID_PREFIXES:
        marker = f":{prefix}"
        if marker in genes:
            left, right_suffix = genes.split(marker, maxsplit=1)
            return left.strip(), f"{prefix}{right_suffix.strip()}"
    return None


def classify_interaction(fdr: float, log_fc: float, fdr_threshold: float, log_fc_threshold: float) -> int:
    """Return 1 for interacting and 0 for non-interacting."""

    return int(fdr < fdr_threshold and log_fc > log_fc_threshold)


def process_interaction_file(
    interaction_file: Path,
    sequences: dict[str, str],
    *,
    fdr_threshold: float = 0.05,
    log_fc_threshold: float = 1.5,
) -> list[ProcessedInteraction]:
    """Convert one raw interaction CSV into model-ready examples."""

    processed: list[ProcessedInteraction] = []
    with interaction_file.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"genes", "FDR", "logFC"}
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{interaction_file} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            gene_pair = split_gene_pair(row["genes"])
            if gene_pair is None:
                continue

            left_id, right_id = gene_pair
            left_sequence = sequences.get(left_id)
            right_sequence = sequences.get(right_id)
            if not left_sequence or not right_sequence:
                continue

            try:
                fdr = float(row["FDR"])
                log_fc = float(row["logFC"])
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value in {interaction_file}:{row_number}") from exc

            label = classify_interaction(fdr, log_fc, fdr_threshold, log_fc_threshold)
            text = f"{spaced_sequence(left_sequence)} , {spaced_sequence(right_sequence)}"
            processed.append(ProcessedInteraction(text=text, fdr=fdr, log_fc=log_fc, label=label))

    return processed


def split_rows(
    rows: list[ProcessedInteraction],
    *,
    train_ratio: float = 0.80,
    eval_ratio: float = 0.15,
) -> tuple[list[ProcessedInteraction], list[ProcessedInteraction], list[ProcessedInteraction]]:
    """Split rows into train, eval, and held-out test sets."""

    if train_ratio <= 0 or eval_ratio <= 0 or train_ratio + eval_ratio >= 1:
        raise ValueError("train_ratio and eval_ratio must be positive and sum to less than 1.")

    train_end = int(len(rows) * train_ratio)
    eval_end = int(len(rows) * (train_ratio + eval_ratio))
    return rows[:train_end], rows[train_end:eval_end], rows[eval_end:]


def write_rows(path: Path, rows: Iterable[ProcessedInteraction]) -> None:
    """Write model-ready rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["text", "FDR", "logFC", "label"])
        writer.writerows(row.as_csv_row() for row in rows)


def build_dataset(
    input_files: Iterable[Path],
    sequences_file: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    shuffle: bool = True,
    fdr_threshold: float = 0.05,
    log_fc_threshold: float = 1.5,
) -> tuple[Path, Path, Path]:
    """Process raw files and write train/eval/test outputs."""

    sequences = load_sequences(sequences_file)
    rows: list[ProcessedInteraction] = []
    for input_file in input_files:
        rows.extend(
            process_interaction_file(
                input_file,
                sequences,
                fdr_threshold=fdr_threshold,
                log_fc_threshold=log_fc_threshold,
            )
        )

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    train_rows, eval_rows, test_rows = split_rows(rows)
    train_path = output_dir / "output_seq_train.csv"
    eval_path = output_dir / "output_seq_test.csv"
    test_path = output_dir / "output_seq_t.csv"

    write_rows(train_path, train_rows)
    write_rows(eval_path, eval_rows)
    write_rows(test_path, test_rows)
    return train_path, eval_path, test_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=Path, default=Path("all_pools_sequences.csv"))
    parser.add_argument("--inputs", type=Path, nargs="+", default=[Path(p) for p in DEFAULT_INPUT_FILES])
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--log-fc-threshold", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_dataset(
        args.inputs,
        args.sequences,
        args.output_dir,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        fdr_threshold=args.fdr_threshold,
        log_fc_threshold=args.log_fc_threshold,
    )
    print("Wrote:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()

