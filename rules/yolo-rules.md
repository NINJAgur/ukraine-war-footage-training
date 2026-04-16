# YOLOv8 / ML Rules — Ukraine Combat Footage Project

These rules are enforced on all ML code in `ml-engine/`.
Violations must be corrected before merging.

---

## HARDWARE CONSTRAINT (NON-NEGOTIABLE)

**RTX 3060 Ti — 8GB VRAM**
- Maximum `batch_size = 8` for YOLOv8m
- Always use `amp=True` (mixed precision) to halve VRAM usage
- Always call `torch.cuda.empty_cache()` after GPU tasks complete
- Never run two GPU tasks concurrently (`concurrency=1` on `gpu` Celery queue)

---

## MANDATORY RULES

### 1. Always Use the Ultralytics Python API
```python
# CORRECT
from ultralytics import YOLO
model = YOLO('yolov8m.pt')
results = model.train(data='data.yaml', epochs=100, device='cuda:0')

# WRONG — subprocess is uncontrollable and doesn't integrate with Celery
import subprocess
subprocess.run(['yolo', 'train', 'data=data.yaml', 'model=yolov8m.pt'])
```

### 2. Always Specify `device='cuda:0'` Explicitly
```python
# CORRECT
model.train(data=data_yaml, epochs=100, device='cuda:0', batch=8, amp=True)
model.predict(source=video_path, device='cuda:0')

# WRONG — may default to CPU, causing extremely slow training
model.train(data=data_yaml, epochs=100)
```

### 3. VRAM-Safe Training Configuration
```python
# CORRECT defaults for RTX 3060 Ti (8GB)
model.train(
    data=data_yaml,
    epochs=100,
    device='cuda:0',
    batch=8,           # safe for YOLOv8m on 8GB
    imgsz=640,
    amp=True,          # mixed precision — halves VRAM
    cache='disk',      # disk cache, not RAM (Windows RAM limit)
    workers=4,         # Windows multiprocessing limit
    patience=20,       # early stopping
    project='runs',
    name=run_name,
)

# WRONG
model.train(batch=16)  # OOM on 8GB
model.train(cache=True)  # loads dataset into RAM — may cause OOM on large datasets
```

### 4. Validate Weights Path Before Loading
```python
# CORRECT
from pathlib import Path

weights_path = Path(config.BASELINE_WEIGHTS)
if not weights_path.exists():
    raise FileNotFoundError(f"Weights not found: {weights_path}")
model = YOLO(str(weights_path))

# WRONG — silent failure if file doesn't exist
model = YOLO('/path/to/missing_weights.pt')
```

### 5. YOLO Label Format (Auto-Labeling Output)
```
# Each label file: one detection per line
# Format: class_id center_x center_y width height
# All values normalized to [0.0, 1.0]

0 0.512 0.384 0.120 0.089   # class 0, centered at (51.2%, 38.4%), w=12%, h=8.9%
1 0.231 0.671 0.043 0.037   # class 1
```

Validation before writing:
```python
def write_yolo_label(path, detections):
    lines = []
    for cls_id, cx, cy, w, h in detections:
        assert 0 <= cx <= 1 and 0 <= cy <= 1
        assert 0 < w <= 1 and 0 < h <= 1
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(path).write_text('\n'.join(lines))
```

### 6. data.yaml Must Be Validated Before Training
```python
import yaml

def validate_data_yaml(yaml_path: str) -> dict:
    cfg = yaml.safe_load(open(yaml_path))
    required = ['path', 'train', 'val', 'nc', 'names']
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"data.yaml missing keys: {missing}")
    if cfg['nc'] != len(cfg['names']):
        raise ValueError(f"nc={cfg['nc']} but {len(cfg['names'])} names")
    return cfg
```

### 7. GPU Memory Cleanup After Tasks
```python
# At the end of every GPU Celery task:
import torch
import gc

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

### 8. Stage 2 Fine-Tuning: Load Baseline Weights
```python
# CORRECT — Stage 2 starts from baseline weights
baseline_path = Path(settings.BASELINE_WEIGHTS_PATH)
model = YOLO(str(baseline_path))  # loads Stage 1 best.pt

model.train(
    data=finetune_data_yaml,
    epochs=50,           # fewer epochs for fine-tuning
    device='cuda:0',
    batch=8,
    lr0=0.001,           # lower LR for fine-tuning (vs 0.01 from scratch)
    freeze=10,           # freeze first 10 layers
    amp=True,
)
```

---

## Class Names for Military Object Detection
```yaml
# Recommended class taxonomy for this project
names:
  0: military_vehicle
  1: tank
  2: armored_personnel_carrier
  3: soldier
  4: drone
  5: artillery
  6: truck
  7: helicopter
```
