"""
End-to-end fine-tuning pipeline orchestrator.

  1. Ingest SST-2 + IMDB (as drift set)
  2. Tokenize (feature engineering)
  3. HPO - Optuna over LoRA hparams
  4. Fine-tune final model with best hparams -> log + register in MLflow
  5. Promote v1 to Production (champion)
  6. Drift detection: embed drift set vs training set, KS-test
  7. If drift detected -> retrain on combined data, register v2 as Staging
  8. Canary rollout: champion (v1) vs candidate (v2) on drift traffic
  9. Promote or rollback based on canary

Run: python -m src.pipeline
"""
import argparse
import gc
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch

from src import ingest, features, train, hpo, drift, registry, canary


ROOT = Path(__file__).parent.parent


def load_config(path: str = None) -> dict:
    path = path or str(ROOT / "configs" / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def run(config: dict):
    # 1. Ingest
    print("\n=== 1. Data Ingestion ===")
    ingest.ingest(
        output_dir=str(ROOT / "data"),
        train_size=config["data"]["train_size"],
        eval_size=config["data"]["eval_size"],
        drift_size=config["data"]["drift_size"],
        seed=config["data"]["random_state"],
    )

    # 2. Features (tokenize)
    print("\n=== 2. Feature Engineering (Tokenization) ===")
    ds = features.build_datasets(
        data_dir=str(ROOT / "data"),
        base_model=config["model"]["base_model"],
        max_length=config["data"]["max_length"],
    )

    # 3. HPO
    print("\n=== 3. Hyperparameter Optimization (Optuna) ===")
    best_params = hpo.run_hpo(config, ds["train"], ds["eval"])
    gc.collect()

    # 4. Final fine-tune + register
    print("\n=== 4. Final Fine-tune + Model Registry (MLflow) ===")
    hparams = {
        "lora_r": best_params["lora_r"],
        "lora_alpha": best_params["lora_alpha"],
        "lora_dropout": best_params["lora_dropout"],
        "lr": best_params["lr"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["max_epochs"],
        "target_modules": config["lora"]["target_modules"],
    }
    champ_result = train.train(config, ds["train"], ds["eval"], hparams=hparams)
    champion_model = champ_result["model"]

    client = registry.get_client(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["registered_model_name"]
    v1 = registry.get_latest_version(client, model_name, "None")
    if v1:
        registry.transition_stage(client, model_name, v1, "Production")

    # 5. Drift detection
    print("\n=== 5. Drift Detection (text embeddings) ===")
    ref_emb = drift.embed(ds["train"], config["model"]["base_model"])
    cur_emb = drift.embed(ds["drift"], config["model"]["base_model"])
    report = drift.detect_drift(
        ref_emb, cur_emb,
        drift_ratio_threshold=config["drift"]["threshold"],
    )
    drift.print_drift_report(report)
    del ref_emb, cur_emb
    gc.collect()

    # 6. Refresh on drift
    candidate_model = None
    if report["overall_drift"]:
        print("\n=== 6. Drift Detected -> Fine-tune Refresh ===")
        # Combine original train with drift data (with its labels) for retraining
        combined_df = pd.concat([ds["train_df"], ds["drift_df"]], ignore_index=True)
        combined_ds = features.tokenize_df(
            combined_df, ds["tokenizer"],
            max_length=config["data"]["max_length"],
        )
        cand_result = train.train(config, combined_ds, ds["eval"], hparams=hparams)
        candidate_model = cand_result["model"]

        v_new = registry.get_latest_version(client, model_name, "None")
        if v_new:
            registry.transition_stage(client, model_name, v_new, "Staging",
                                      archive_existing=False)
    else:
        print("\n=== 6. No drift -> skip refresh ===")

    # 7. Canary
    if candidate_model is not None:
        print("\n=== 7. Canary Rollout ===")
        # Use the drift set as simulated live production traffic
        cr = canary.canary_rollout(
            champion=champion_model,
            candidate=candidate_model,
            eval_ds=ds["drift"],
            steps=config["canary"]["steps"],
            promotion_threshold=config["canary"]["promotion_threshold"],
        )

        # 8. Act on decision
        print("\n=== 8. Acting on Canary Decision ===")
        staging_v = registry.get_latest_version(client, model_name, "Staging")
        if cr["decision"] == "promote" and staging_v:
            registry.transition_stage(client, model_name, staging_v, "Production",
                                      archive_existing=True)
            print("[pipeline] candidate promoted to Production")
        elif staging_v:
            registry.transition_stage(client, model_name, staging_v, "Archived",
                                      archive_existing=False)
            print("[pipeline] candidate archived (rollback)")

    print("\n=== Pipeline complete ===")
    print(f"View MLflow UI: mlflow ui --backend-store-uri {config['mlflow']['tracking_uri']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run(load_config(args.config))
