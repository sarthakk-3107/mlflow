"""
Fine-tuning loop using HuggingFace Trainer + MLflow tracking.
"""
from pathlib import Path
from typing import Dict, Any

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from .model import build_lora_model, LoRAHParams, count_trainable_params, predict


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }


def finetune_one(
    train_ds, eval_ds,
    base_model: str,
    num_labels: int,
    hparams: Dict[str, Any],
    output_dir: str = "./ft_output",
    log_to_mlflow: bool = True,
    run_name: str = "ft",
    registered_model_name: str = None,
    log_model_artifact: bool = True,
) -> Dict[str, Any]:
    """Fine-tune one LoRA model. Returns dict with model + metrics."""
    lora_hp = LoRAHParams(
        r=hparams.get("lora_r", 8),
        alpha=hparams.get("lora_alpha", 16),
        dropout=hparams.get("lora_dropout", 0.1),
        target_modules=hparams.get("target_modules", ["q_lin", "v_lin"]),
    )
    model = build_lora_model(base_model, num_labels, lora_hp)
    param_info = count_trainable_params(model)
    print(f"[train] trainable params: {param_info['trainable']:,} / "
          f"{param_info['total']:,} ({param_info['pct']:.2f}%)")

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=hparams.get("epochs", 2),
        per_device_train_batch_size=hparams.get("batch_size", 16),
        per_device_eval_batch_size=hparams.get("batch_size", 16),
        learning_rate=hparams.get("lr", 5e-4),
        eval_strategy="epoch",
        save_strategy="no",           # Skip intermediate checkpoints (disk + time)
        logging_steps=50,
        report_to="none",             # We handle MLflow ourselves
        disable_tqdm=True,
        dataloader_num_workers=0,
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
    )

    run_ctx = mlflow.start_run(run_name=run_name) if log_to_mlflow else None
    try:
        trainer.train()
        eval_metrics = trainer.evaluate()
        # Clean keys: 'eval_accuracy' -> 'accuracy'
        metrics = {
            "accuracy": float(eval_metrics["eval_accuracy"]),
            "f1": float(eval_metrics["eval_f1"]),
            "eval_loss": float(eval_metrics["eval_loss"]),
        }

        if log_to_mlflow:
            mlflow.log_params({k: v for k, v in hparams.items()
                              if isinstance(v, (int, float, str, bool))})
            mlflow.log_metric("trainable_params", param_info["trainable"])
            mlflow.log_metrics(metrics)
            if log_model_artifact:
                # Save LoRA adapters only - small (~few MB) vs full model (~250MB)
                adapter_dir = Path(output_dir) / "lora_adapter"
                model.save_pretrained(adapter_dir)
                mlflow.log_artifacts(str(adapter_dir), artifact_path="lora_adapter")
                if registered_model_name:
                    # Register a pointer to this run's artifact
                    client = mlflow.tracking.MlflowClient()
                    run_id = mlflow.active_run().info.run_id
                    model_uri = f"runs:/{run_id}/lora_adapter"
                    try:
                        client.create_registered_model(registered_model_name)
                    except Exception:
                        pass  # already exists
                    mv = client.create_model_version(
                        name=registered_model_name,
                        source=model_uri,
                        run_id=run_id,
                    )
                    print(f"[train] registered {registered_model_name} v{mv.version}")

        return {"model": model, "metrics": metrics, "trainer": trainer}
    finally:
        if run_ctx is not None:
            mlflow.end_run()


def train(config: dict, train_ds, eval_ds, hparams: Dict[str, Any] = None):
    """High-level entry point."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    if hparams is None:
        hparams = {
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["alpha"],
            "lora_dropout": config["lora"]["dropout"],
            "lr": config["training"]["learning_rate"],
            "batch_size": config["training"]["batch_size"],
            "epochs": config["training"]["max_epochs"],
            "target_modules": config["lora"]["target_modules"],
        }

    result = finetune_one(
        train_ds, eval_ds,
        base_model=config["model"]["base_model"],
        num_labels=config["model"]["num_labels"],
        hparams=hparams,
        run_name="final_finetune",
        registered_model_name=config["mlflow"]["registered_model_name"],
    )
    print(f"[train] final metrics: {result['metrics']}")
    return result
