"""
Canary rollout for fine-tuned sentiment classifiers.
Routes a growing fraction of live traffic to the new candidate; rolls back
if candidate accuracy falls below threshold.
"""
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score

from .model import predict


def canary_rollout(
    champion, candidate, eval_ds,
    steps: List[float] = None,
    promotion_threshold: float = 0.75,
    seed: int = 0,
) -> Dict:
    steps = steps or [0.1, 0.25, 0.5, 1.0]

    # Precompute predictions for both models on the whole set (faster than
    # re-running per step)
    champ_preds, _, labels = predict(champion, eval_ds)
    cand_preds, _, _ = predict(candidate, eval_ds)

    rng = np.random.default_rng(seed)
    n = len(labels)
    history = []
    decision = "promote"

    for traffic_pct in steps:
        mask = rng.random(n) < traffic_pct
        if mask.sum() == 0:
            continue
        cand_acc = accuracy_score(labels[mask], cand_preds[mask])
        champ_acc = (accuracy_score(labels[~mask], champ_preds[~mask])
                     if (~mask).sum() > 0 else None)
        history.append({
            "traffic_pct": traffic_pct,
            "candidate_n": int(mask.sum()),
            "candidate_acc": float(cand_acc),
            "champion_acc": float(champ_acc) if champ_acc is not None else None,
        })
        print(f"[canary] traffic={traffic_pct:.0%}  candidate_acc={cand_acc:.3f}  "
              f"champion_acc={champ_acc if champ_acc is None else f'{champ_acc:.3f}'}")

        if cand_acc < promotion_threshold:
            decision = "rollback"
            print(f"[canary] candidate below {promotion_threshold} -> ROLLBACK")
            break

    return {"decision": decision, "history": history}
