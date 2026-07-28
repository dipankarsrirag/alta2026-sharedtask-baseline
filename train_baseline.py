"""
Baseline for the ALTA 2026 shared task (BESSTIE sentiment + sarcasm
classification for en-AU / en-UK): fine-tunes a pretrained transformer
(default: distilbert-base-uncased) separately for the sentiment and
sarcasm tasks, with class-balanced loss weighting. Open-source
model/weights only.

Usage:
    python train_baseline.py --data_dir data --out answer.csv
    python train_baseline.py --model_name roberta-base --epochs 4
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_one_task(task, train_df, valid_df, tokenizer, args, device):
    train_enc = tokenizer(
        list(train_df["text"]), truncation=True, padding=True, max_length=args.max_length
    )
    valid_enc = tokenizer(
        list(valid_df["text"]), truncation=True, padding=True, max_length=args.max_length
    )
    train_labels = train_df[task].tolist()
    train_ds = TextDataset(train_enc, train_labels)
    valid_ds = TextDataset(valid_enc, valid_df[task].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=np.array(train_labels)
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.work_dir, task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()

    preds = trainer.predict(valid_ds)
    return np.argmax(preds.predictions, axis=-1)


def score(gold_df, run_df):
    scores = {}
    for task in ["sentiment", "sarcasm"]:
        for dialect in ["en-AU", "en-UK"]:
            gold = gold_df[gold_df["variety"] == dialect][task]
            pred = run_df[run_df["variety"] == dialect][task]
            scores[f"{task}-{dialect}"] = f1_score(gold, pred, average="macro")
    final = (
        min(scores["sentiment-en-AU"], scores["sentiment-en-UK"])
        + min(scores["sarcasm-en-AU"], scores["sarcasm-en-UK"])
    ) / 2
    return scores, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out", default="answer.csv")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--work_dir", default="./tf_runs")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    preds = {}
    for task in ["sentiment", "sarcasm"]:
        print(f"\n=== Fine-tuning for task: {task} ===")
        preds[task] = train_one_task(task, train_df, valid_df, tokenizer, args, device)

    answer_df = valid_df[["source", "variety", "text"]].copy()
    answer_df["sentiment"] = preds["sentiment"]
    answer_df["sarcasm"] = preds["sarcasm"]
    answer_df.to_csv(args.out, index=False)
    print(f"\nWrote predictions to {args.out}")

    # Local sanity check against the released gold labels for valid.csv.
    scores, final = score(valid_df, answer_df)
    print("\nBaseline scores on the public valid set (dev-only, not used for final ranking):")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"  final score: {final:.4f}")


if __name__ == "__main__":
    main()
