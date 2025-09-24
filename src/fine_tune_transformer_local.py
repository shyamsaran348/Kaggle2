#!/usr/bin/env python3
"""
Fine-tune a Hugging Face transformer for binary classification (rule_violation).
Usage (example):
python src/fine_tune_transformer_local.py \
  --train_path data/train.csv \
  --test_path data/test.csv \
  --output_dir outputs/transformer_roberta \
  --model_name distilroberta-base \
  --epochs 3 \
  --per_device_train_batch_size 8
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import torch
import torch.nn as nn
from scipy.special import expit  # sigmoid for numpy logits

# ----------------------------
# Custom Trainer to handle pos_weight
# ----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight  # store pos_weight for compute_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels").float()  # ensure float for BCE
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)  # flatten

        if self.pos_weight is not None:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fct(logits, labels)
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

# ----------------------------
# Compute ROC-AUC metric
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.array(logits).reshape(-1)
    probs = expit(logits)
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.5
    return {"roc_auc": float(auc)}

# ----------------------------
# Prepare datasets
# ----------------------------
def prepare_datasets(train_df, test_df, text_col="body", label_col="rule_violation", val_size=0.1, seed=42):
    train_df[text_col] = train_df[text_col].fillna("").astype(str)
    test_df[text_col] = test_df[text_col].fillna("").astype(str)
    train_df[label_col] = train_df[label_col].astype(int)

    train_split, val_split = train_test_split(
        train_df, test_size=val_size, random_state=seed, stratify=train_df[label_col]
    )

    ds_train = Dataset.from_pandas(
        train_split[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"})
    )
    ds_val = Dataset.from_pandas(
        val_split[[text_col, label_col]].rename(columns={text_col: "text", label_col: "labels"})
    )
    ds_test = Dataset.from_pandas(
        test_df[[text_col, "row_id"]].rename(columns={text_col: "text"})
    )

    return ds_train, ds_val, ds_test, train_split

# ----------------------------
# Tokenize datasets
# ----------------------------
def tokenize_datasets(ds_train, ds_val, ds_test, tokenizer, max_length=256):
    def fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    ds_train = ds_train.map(fn, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(fn, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(fn, batched=True, remove_columns=["text"])

    ds_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "row_id"])
    return ds_train, ds_val, ds_test

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--output_dir", type=str, default="outputs/transformer_model")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 1) Load CSVs
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

    # 2) Prepare datasets
    ds_train, ds_val, ds_test, train_split = prepare_datasets(train_df, test_df, val_size=args.val_size, seed=args.seed)

    # 3) Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ds_train, ds_val, ds_test = tokenize_datasets(ds_train, ds_val, ds_test, tokenizer, max_length=args.max_length)

    # 4) Compute pos_weight
    labels_arr = train_split["rule_violation"].values
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    pos_weight = torch.tensor(float(n_neg) / max(1.0, n_pos), dtype=torch.float32)
    print(f"pos_weight = {pos_weight.item()} (n_pos={n_pos}, n_neg={n_neg})")

    # 5) Load model
    print("Loading model:", args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="single_label_classification"  # <-- crucial for BCE instead of MSE
    )

    # 6) TrainingArguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        fp16=use_fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        save_total_limit=2,
    )

    # 7) Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        pos_weight=pos_weight
    )

    # 8) Train
    trainer.train()

    # 9) Save model & tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Model and tokenizer saved to:", args.output_dir)

    # 10) Predict
    print("Running inference on test set...")
    preds_output = trainer.predict(ds_test)
    logits = np.array(preds_output.predictions).reshape(-1)
    probs = expit(logits)

    test_row_ids = test_df["row_id"].values
    submission = pd.DataFrame({"row_id": test_row_ids, "rule_violation": probs})
    submission_file = os.path.join(args.output_dir, "submission_transformer.csv")
    submission.to_csv(submission_file, index=False)
    print("✅ Submission saved to:", submission_file)

if __name__ == "__main__":
    main()
