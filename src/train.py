"""Fine-tune and evaluate a Transformer model for PPI classification."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np
from transformers import TrainingArguments
from trl import SFTTrainer

ID_TO_LABEL = {0: "Non-interacting", 1: "Interacting"}
LABEL_TO_ID = {"Non-interacting": 0, "Interacting": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--model-name", default="gpt2")#"distilbert-base-uncased"
    parser.add_argument("--output-dir", type=Path, default=Path("models/ppi-distilbert-lora"))
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.01)
    parser.add_argument("--plot-output", type=Path, default=None)
    return parser.parse_args()


def build_target_modules(model: Any) -> list[str]:
    """Detect linear module names that can be used as LoRA targets."""

    import torch.nn as nn

    names: set[str] = set()
    for full_name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names.add(full_name.split(".")[-1])

    substrings = ("q", "k", "v", "out", "proj", "lin", "dense", "fc", "ffn")
    filtered = sorted(name for name in names if any(s in name.lower() for s in substrings))
    return filtered or sorted(names)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    import evaluate

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)


def load_validation_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def evaluate_pipeline(classifier: Any, validation_file: Path) -> dict[str, Any]:
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    labels: list[int] = []
    predictions: list[int] = []

    for row in load_validation_rows(validation_file):
        result = classifier(row["text"])[0]
        predicted_label = LABEL_TO_ID[result["label"]]
        labels.append(int(row["label"]))
        predictions.append(predicted_label)

    return {
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(labels, predictions, output_dict=True),
        "roc_auc": roc_auc_score(labels, predictions) if len(set(labels)) > 1 else None,
    }


def save_prediction_plot(metrics: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(metrics["confusion_matrix"])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    x_labels = ["Interacting", "Non-interacting"]
    prediction_counts = [true_positive, true_negative]
    actual_counts = [true_positive + false_negative, true_negative + false_positive]

    x_axis = np.arange(len(x_labels))
    plt.figure(figsize=(7, 4))
    plt.bar(x_axis - 0.2, prediction_counts, 0.4, label="Prediction")
    plt.bar(x_axis + 0.2, actual_counts, 0.4, label="Actual")
    plt.xticks(x_axis, x_labels)
    plt.xlabel("Class")
    plt.ylabel("Samples")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        pipeline,
    )

    dataset = load_dataset(
        "csv",
        data_files={"train": str(args.train_file), "test": str(args.eval_file)},
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def preprocess(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(examples["text"], truncation=True)

    tokenized_dataset = dataset.map(preprocess, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=build_target_modules(model),
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[],
        hub_token=os.environ.get("HF_TOKEN"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    metrics = evaluate_pipeline(classifier, args.validation_file)
    if args.plot_output:
        save_prediction_plot(metrics, args.plot_output)
    return metrics


def main() -> None:
    args = parse_args()
    metrics = train(args)
    print(metrics)


if __name__ == "__main__":
    main()

