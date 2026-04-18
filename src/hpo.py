"""
HPO over LoRA rank, alpha, and learning rate using Optuna.
Each trial is a full mini fine-tune, so we keep n_trials small.
"""
from typing import Dict, Any

import mlflow
import optuna
from optuna.samplers import TPESampler

from .train import finetune_one


def run_hpo(
    config: dict,
    train_ds, eval_ds,
    n_trials: int = None,
    timeout: int = None,
) -> Dict[str, Any]:
    n_trials = n_trials or config["hpo"]["n_trials"]
    timeout = timeout or config["hpo"]["timeout_seconds"]

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"] + "_hpo")

    def objective(trial: optuna.Trial) -> float:
        hparams = {
            "lora_r": trial.suggest_categorical("lora_r", [4, 8, 16]),
            "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "batch_size": config["training"]["batch_size"],
            "epochs": 1,      # 1 epoch per trial to save time
            "target_modules": config["lora"]["target_modules"],
        }
        result = finetune_one(
            train_ds, eval_ds,
            base_model=config["model"]["base_model"],
            num_labels=config["model"]["num_labels"],
            hparams=hparams,
            run_name=f"hpo_trial_{trial.number}",
            log_model_artifact=False,
            output_dir=f"./ft_output/trial_{trial.number}",
        )
        return result["metrics"]["accuracy"]

    sampler = TPESampler(seed=config["data"]["random_state"])
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    print(f"[hpo] best params: {study.best_params}")
    print(f"[hpo] best accuracy: {study.best_value:.4f}")
    return study.best_params
