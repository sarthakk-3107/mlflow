"""
LoRA-wrapped DistilBERT for sentiment classification.

Why LoRA: full fine-tuning updates all ~66M DistilBERT params. On CPU,
that's impractical. LoRA freezes the base model and injects small trainable
rank-decomposition matrices into attention projections. With r=8 we train
~0.1% of params - fast enough for CPU, and results are competitive.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType


@dataclass
class LoRAHParams:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = None


def build_lora_model(base_model: str, num_labels: int, hparams: LoRAHParams):
    """Load base DistilBERT and wrap with LoRA adapters."""
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=num_labels
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=hparams.r,
        lora_alpha=hparams.alpha,
        lora_dropout=hparams.dropout,
        target_modules=hparams.target_modules or ["q_lin", "v_lin"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model


def count_trainable_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "pct": 100 * trainable / total}


@torch.no_grad()
def predict(model, dataset, batch_size: int = 32) -> tuple:
    """Run inference on a dataset. Returns (preds, probs, labels)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs, all_labels = [], []
    for batch in loader:
        labels = batch.pop("labels")
        out = model(**batch)
        probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    preds = probs.argmax(axis=-1)
    return preds, probs, labels
