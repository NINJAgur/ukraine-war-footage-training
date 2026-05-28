# /train — Queue a YOLOv8 Training Job

## Description
Queues a YOLO model training job via the training-engine Celery worker (Q=training).
Supports Stage 1 (Baseline on Kaggle data) and Stage 2 (Fine-tune on scraped/labeled data).

## Usage
```
/train [stage] [--model MODEL] [--epochs N] [--batch N]
```

### Arguments
| Argument | Values | Description |
|----------|--------|-------------|
| `stage` | `baseline`, `finetune` | Training stage. Required. |
| `--model` | `AIRCRAFT`, `VEHICLE`, `PERSONNEL`, `GENERAL` | Model type. Required. |
| `--epochs` | Integer | Override default epoch count |
| `--batch` | Integer (max 8 local / 16 T4) | Override batch size |

### Examples
```bash
/train baseline --model AIRCRAFT            # Stage 1: train on Kaggle military datasets
/train finetune --model VEHICLE             # Stage 2: fine-tune on scraped merged dataset
/train baseline --model GENERAL --epochs 50
```

## What This Command Does

1. Verify GPU is available: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
2. Verify the training-engine Celery worker is running on Q=training
3. For finetune: verify a DONE baseline run with `weights_path` exists for the model
4. POST to `POST /api/admin/train` with `{stage, model_type}` → creates `TrainingRun(QUEUED)` in DB
5. Task dispatched to Q=training: `train_baseline` or `train_finetune`
6. Stream progress via WebSocket `/ws/training/{run_id}` until completion

## VRAM Budget (YOLOv8m)

| batch_size | VRAM | Safe on 8GB (RTX 3060 Ti)? | Safe on 16GB (T4)? |
|-----------|------|---------------------------|---------------------|
| 4 | ~4GB | Yes | Yes |
| 8 | ~6GB | Yes (default) | Yes |
| 16 | ~10GB | NO — OOM | Yes |

## Monitoring Progress

Training progress streams via WebSocket to Admin UI at `/admin`.
To monitor from CLI:

```python
from celery.result import AsyncResult
result = AsyncResult(task_id)
while result.state == 'PROGRESS':
    meta = result.info
    print(f"Epoch {meta['epoch']}/{meta['total_epochs']} — mAP50: {meta.get('mAP50', 'N/A')}")
    time.sleep(5)
```

## Troubleshooting

- **"CUDA out of memory"**: Reduce `--batch` to 4
- **"No training worker"**: Start with `celery -A celery_app worker -Q training --concurrency=1 --loglevel=info` from `training-engine/`
- **"No baseline for finetune"**: Run `/train baseline --model <MODEL>` first
- **"Stuck in QUEUED"**: Check training-engine worker logs; ensure `CELERY_BROKER_URL` points to the same Redis
