"""
Data ingestion. Downloads SST-2 sentiment dataset from HuggingFace,
saves small splits to disk as parquet. Creates a 'drift' set by using
a different domain (IMDB-style longer movie reviews) to simulate
distribution shift in production.
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def ingest(output_dir: str = "data", train_size: int = 800, eval_size: int = 200,
           drift_size: int = 200, seed: int = 42):
    out = Path(output_dir)
    out.mkdir(exist_ok=True, parents=True)

    # SST-2: short sentence-level sentiment (training distribution)
    print("[ingest] downloading SST-2...")
    sst2 = load_dataset("stanfordnlp/sst2")
    train = sst2["train"].shuffle(seed=seed).select(range(train_size))
    val = sst2["validation"].shuffle(seed=seed).select(range(min(eval_size, len(sst2["validation"]))))

    train_df = pd.DataFrame({"text": train["sentence"], "label": train["label"]})
    val_df = pd.DataFrame({"text": val["sentence"], "label": val["label"]})
    train_df.to_parquet(out / "train.parquet", index=False)
    val_df.to_parquet(out / "eval.parquet", index=False)
    print(f"[ingest] train={len(train_df)}, eval={len(val_df)}")

    # IMDB: longer movie reviews - same task, different distribution (DRIFT)
    print("[ingest] downloading IMDB for drift set...")
    imdb = load_dataset("stanfordnlp/imdb", split="test")
    imdb = imdb.shuffle(seed=seed).select(range(drift_size))
    # Truncate long reviews to first 200 chars so lengths aren't absurd
    drift_df = pd.DataFrame({
        "text": [t[:200] for t in imdb["text"]],
        "label": imdb["label"],
    })
    drift_df.to_parquet(out / "drift.parquet", index=False)
    print(f"[ingest] drift={len(drift_df)}")

    return {"train": out / "train.parquet", "eval": out / "eval.parquet",
            "drift": out / "drift.parquet"}


if __name__ == "__main__":
    ingest()
