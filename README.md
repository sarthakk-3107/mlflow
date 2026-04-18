# LoRA Fine-tuning Pipeline with Feature Store + Model Registry

End-to-end fine-tuning pipeline for sentiment classification. Actually fine-tunes DistilBERT using LoRA adapters — trains ~0.1% of params so it runs on CPU.

## What it does

Fine-tunes `distilbert-base-uncased` on SST-2 sentiment, then simulates production drift using IMDB reviews (same task, different distribution) to trigger an automatic refresh and canary rollout.

## Stack

- **PyTorch + HuggingFace Transformers** — base model
- **PEFT (LoRA)** — parameter-efficient fine-tuning
- **MLflow** — experiment tracking + model registry
- **Optuna** — HPO over LoRA rank, alpha, dropout, learning rate
- **SciPy KS-test** — drift detection on [CLS] embeddings

## Setup

```bash
pip install -r requirements.txt
```

First run will download:
- DistilBERT base model (~250MB)
- SST-2 dataset (~7MB)
- IMDB dataset (~80MB, only uses 200 examples but downloads full split)

## Run

```bash
cd ml_pipeline_ft
python -m src.pipeline
```

## Expected runtime on CPU

| Stage | Time on weak CPU |
|-------|------------------|
| Data download (first run) | 1–3 min |
| Tokenization | seconds |
| HPO (3 trials × 1 epoch) | 3–8 min |
| Final fine-tune (2 epochs) | 2–4 min |
| Embedding computation (drift) | 30–60 sec |
| Refresh fine-tune | 2–5 min |
| Canary rollout | 10 sec |
| **Total** | **~10–20 min** |

## If it's too slow

Edit `configs/config.yaml`:

```yaml
data:
  train_size: 300       # from 800
  eval_size: 100        # from 200
  max_length: 32        # from 64 - huge speedup

training:
  max_epochs: 1         # from 2

hpo:
  n_trials: 2           # from 3
```

Or swap to a smaller base model:
```yaml
model:
  base_model: "prajjwal1/bert-tiny"   # 4M params, 10x faster
```

## What each stage does

1. **Ingest** — downloads SST-2 (training distribution) and IMDB (drift simulation).
2. **Features** — tokenizes text with the base model's tokenizer. The tokenizer acts as our feature store: identical processing at train and inference.
3. **HPO** — Optuna TPE over LoRA `r`, `alpha`, `dropout`, and `lr`. 1 epoch per trial.
4. **Fine-tune + register** — trains with best hparams on full epochs, saves LoRA adapter (few MB, not the whole 250MB model), logs to MLflow, registers v1, promotes to Production.
5. **Drift detection** — embeds both training and incoming batches using the base model's [CLS] token, KS-tests sampled dims. Flags overall drift if ≥15% of dims shifted.
6. **Refresh** — fine-tunes v2 on combined train + drift data, registers as Staging.
7. **Canary** — routes 10% → 25% → 50% → 100% of drift traffic to v2. Rolls back if accuracy drops below 0.75.
8. **Decision** — promotes v2 to Production if canary passes, else archives it.

## View results

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Open http://localhost:5000. You'll see:
- Two experiments: `sentiment_ft_hpo` (all HPO trials) and `sentiment_ft` (final + refresh)
- Model registry showing `sentiment_classifier` with versions in different stages
- Per-run params (LoRA config, LR), metrics (accuracy, F1, eval_loss, trainable_params), and artifacts (LoRA adapter files)

## Project layout

```
ml_pipeline_ft/
├── configs/config.yaml
├── src/
│   ├── ingest.py       # HF datasets download
│   ├── features.py     # tokenization
│   ├── model.py        # LoRA-wrapped DistilBERT
│   ├── train.py        # HF Trainer + MLflow
│   ├── hpo.py          # Optuna HPO
│   ├── drift.py        # embedding-based drift
│   ├── registry.py     # MLflow stage transitions
│   ├── canary.py       # rollout simulation
│   └── pipeline.py     # orchestrator
└── requirements.txt
```
