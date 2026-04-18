"""
Feature engineering for text.
Tokenizes text using the base model's tokenizer. The tokenizer+config
act as our 'feature definition' - guaranteed identical processing at
train and inference time (the point of a feature store).
"""
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    """Wraps tokenized text + labels for the Trainer / DataLoader."""
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def get_tokenizer(base_model: str):
    return AutoTokenizer.from_pretrained(base_model)


def tokenize_df(df: pd.DataFrame, tokenizer, max_length: int = 64) -> TextDataset:
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    labels = df["label"].values.astype(np.int64)
    return TextDataset(enc, labels)


def build_datasets(data_dir: str, base_model: str, max_length: int = 64):
    """Load parquet splits, tokenize with the shared tokenizer, return datasets."""
    tokenizer = get_tokenizer(base_model)
    d = Path(data_dir)
    train_df = pd.read_parquet(d / "train.parquet")
    eval_df = pd.read_parquet(d / "eval.parquet")
    drift_df = pd.read_parquet(d / "drift.parquet")

    train_ds = tokenize_df(train_df, tokenizer, max_length)
    eval_ds = tokenize_df(eval_df, tokenizer, max_length)
    drift_ds = tokenize_df(drift_df, tokenizer, max_length)

    print(f"[features] train={len(train_ds)}, eval={len(eval_ds)}, drift={len(drift_ds)}")
    return {
        "train": train_ds, "eval": eval_ds, "drift": drift_ds,
        "train_df": train_df, "eval_df": eval_df, "drift_df": drift_df,
        "tokenizer": tokenizer,
    }
