# /train — Queue a YOLOv8 Training Job

## Description
Queues a YOLO model training job via the ML Engine's GPU Celery worker.
Supports Stage 1 (Baseline on Kaggle data) and Stage 2 (Fine-tune on custom labeled data).

## Usage
```
/train [stage] [--datasets IDS] [--epochs N] [--batch N]
```

### Arguments
| Argument | Values | Description |
|----------|--------|-------------|
| `stage` | `baseline`, `finetune` | Training stage. Required. |
| `--datasets` | Comma-separated IDs | Dataset IDs to use (Stage 2 only). E.g., `--datasets 1,2,5` |
| `--epochs` | Integer | Override default epoch count |
| `--batch` | Integer (max 8) | Override batch size. Never exceed 8 on RTX 3060 Ti. |

### Examples
```bash
/train baseline                         # Stage 1: train on Kaggle military datasets
/train finetune --datasets 1,2,3        # Stage 2: fine-tune on custom labeled datasets
/train baseline --epochs 50             # Stage 1 with custom epoch count
/train finetune --datasets 4,5 --batch 4   # Stage 2 with smaller batch (more VRAM headroom)
```

## What This Command Does

1. Verify GPU is available: `python -c "import torch; print(torch.cuda.get_device_name(0))"`
2. Verify the ml-engine GPU Celery worker is running
3. For Stage 2: verify specified dataset IDs exist in the DB and have `status=LABELED`
4. Dispatch the appropriate Celery task to the `gpu` queue:
   - `train_baseline.apply_async(queue='gpu')` for Stage 1
   - `train_finetune.apply_async(args=[dataset_ids], queue='gpu')` for Stage 2
5. Return the `TrainingRun` ID and Celery task ID
6. Stream progress updates (epoch, loss, mAP50) until completion

## VRAM Safety

| batch_size | YOLOv8m VRAM | Safe on RTX 3060 Ti? |
|-----------|-------------|----------------------|
| 4 | ~4GB | Yes |
| 8 | ~6GB | Yes (default) |
| 16 | ~10GB | NO — will OOM |

The `gpu` Celery worker runs with `--concurrency=1`. Only one training job runs at a time.

## Monitoring Progress

Training progress is streamed via WebSocket to the Admin UI at `/admin/train`.
To monitor from CLI:

```python
from celery.result import AsyncResult
result = AsyncResult(task_id)
while result.state == 'PROGRESS':
    meta = result.info
    print(f"Epoch {meta['epoch']}/{meta['total_epochs']} — loss: {meta['loss']:.4f} — mAP50: {meta['mAP50']:.4f}")
    time.sleep(5)
```

## Troubleshooting

- **"CUDA out of memory"**: Reduce `--batch` to 4
- **"No GPU worker"**: Start with `celery -A ml_engine.celery_app worker -Q gpu --concurrency=1 --loglevel=info`
- **"Dataset not found"**: Check dataset status with `psql -c "SELECT id, status FROM datasets;"`
- **"weights not found for finetune"**: Run `/train baseline` first to generate `baseline.pt`
