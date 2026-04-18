"""
Drift detection for text.
For tabular data you can KS-test raw features. For text, raw tokens aren't
meaningful - we embed both reference and current batches with the base model's
[CLS] output, then KS-test each embedding dimension.
"""
from typing import Dict

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoModel


@torch.no_grad()
def embed(dataset, base_model: str, batch_size: int = 32) -> np.ndarray:
    """Compute [CLS] embeddings for a TextDataset."""
    model = AutoModel.from_pretrained(base_model)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_emb = []
    for batch in loader:
        batch.pop("labels", None)
        out = model(**batch)
        cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        all_emb.append(cls)
    return np.concatenate(all_emb, axis=0)


def detect_drift(
    reference_emb: np.ndarray,
    current_emb: np.ndarray,
    p_threshold: float = 0.05,
    drift_ratio_threshold: float = 0.15,
) -> Dict:
    """KS-test per embedding dimension. Returns overall drift decision."""
    assert reference_emb.shape[1] == current_emb.shape[1]
    n_dims = reference_emb.shape[1]

    # Embeddings are high-dim (768); sample dims to keep report short
    sample_size = min(50, n_dims)
    rng = np.random.default_rng(0)
    sampled_dims = rng.choice(n_dims, size=sample_size, replace=False)

    drifted = 0
    per_dim = []
    for i in sampled_dims:
        ks, p = stats.ks_2samp(reference_emb[:, i], current_emb[:, i])
        d = bool(p < p_threshold)
        if d:
            drifted += 1
        per_dim.append({"dim": int(i), "ks": float(ks), "p": float(p), "drifted": d})

    ratio = drifted / sample_size
    return {
        "drift_ratio": ratio,
        "drifted_dims": drifted,
        "sampled_dims": sample_size,
        "total_dims": n_dims,
        "overall_drift": ratio >= drift_ratio_threshold,
        "per_dim": per_dim,
    }


def print_drift_report(report: Dict):
    print(f"[drift] {report['drifted_dims']}/{report['sampled_dims']} sampled dims drifted "
          f"(ratio={report['drift_ratio']:.2f}) -> overall_drift={report['overall_drift']}")
