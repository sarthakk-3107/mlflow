"""
Model registry helpers for the LoRA fine-tuning pipeline.
Uses MLflow's stage-based registry (Staging / Production / Archived).
"""
from typing import Optional

import mlflow
from mlflow.tracking import MlflowClient


def get_client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def get_latest_version(client: MlflowClient, name: str, stage: str = "None") -> Optional[str]:
    versions = client.get_latest_versions(name, stages=[stage])
    return versions[0].version if versions else None


def transition_stage(client: MlflowClient, name: str, version: str, stage: str,
                     archive_existing: bool = True):
    client.transition_model_version_stage(
        name=name, version=version, stage=stage,
        archive_existing_versions=archive_existing,
    )
    print(f"[registry] {name} v{version} -> {stage}")
