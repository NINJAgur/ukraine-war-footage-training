# FastAPI Rules — Ukraine Combat Footage Project

These rules are enforced on all code in `web-app/backend/`.
Violations must be corrected before merging.

---

## MANDATORY RULES

### 1. All Route Handlers Must Be `async def`
```python
# CORRECT
@router.get("/feed")
async def get_feed(db: AsyncSession = Depends(get_db)):
    ...

# WRONG — blocks the event loop
@router.get("/feed")
def get_feed(db: Session = Depends(get_db)):
    ...
```

### 2. Use `AsyncSession` for All Database Access
```python
# CORRECT
from sqlalchemy.ext.asyncio import AsyncSession

async def get_db():
    async with AsyncSession(engine) as session:
        yield session

# WRONG — synchronous session in async route
from sqlalchemy.orm import Session
```

### 3. Pydantic v2 Schema Syntax
```python
# CORRECT — Pydantic v2
from pydantic import BaseModel, ConfigDict

class ClipResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    title: str

# WRONG — Pydantic v1 syntax
class ClipResponse(BaseModel):
    class Config:
        orm_mode = True
```

### 4. Dependency Injection via `Depends()`
```python
# CORRECT
@router.get("/admin/datasets")
async def get_datasets(
    db: AsyncSession = Depends(get_db),
    current_admin = Depends(get_current_admin)
):
    ...

# WRONG — no auth guard
@router.get("/admin/datasets")
async def get_datasets(db: AsyncSession = Depends(get_db)):
    ...  # unprotected admin route
```

### 5. All Admin Routes Protected with JWT
```python
# Every route in admin.py must include:
current_admin = Depends(get_current_admin)

# get_current_admin implementation:
async def get_current_admin(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return payload
```

### 6. Use `HTTPException` for Client Errors
```python
# CORRECT
if not clip:
    raise HTTPException(status_code=404, detail="Clip not found")

# WRONG — raw Python exceptions
if not clip:
    raise ValueError("Clip not found")  # becomes a 500
```

### 7. Environment Variables Only — No Hardcoded Secrets
```python
# CORRECT
import os
JWT_SECRET = os.environ["JWT_SECRET"]

# WRONG
JWT_SECRET = "my-secret-key-123"
```

### 8. Never Block the Event Loop
```python
# CORRECT
await asyncio.sleep(1)

# WRONG — blocks the entire event loop
import time
time.sleep(1)

# WRONG — synchronous file I/O in async route (use aiofiles or run_in_executor)
with open("file.txt") as f:
    data = f.read()
```

### 9. Celery Task Dispatch from Routes
```python
# CORRECT — dispatch and return task ID immediately
@router.post("/admin/train")
async def start_training(body: TrainRequest, _=Depends(get_current_admin)):
    task = train_baseline.delay(dataset_ids=body.dataset_ids)
    return {"task_id": task.id}

# WRONG — running ML training synchronously in the route
@router.post("/admin/train")
async def start_training(body: TrainRequest):
    result = model.train(...)  # blocks for hours
    return result
```

### 10. CORS Configuration from Environment
```python
# CORRECT
origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, ...)

# WRONG
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```

---

## Standard Response Shapes

```python
# List with pagination
{"items": [...], "total": 100, "page": 1, "per_page": 20}

# Single item
{"id": 1, "title": "...", ...}

# Task dispatch confirmation
{"task_id": "abc-123", "status": "QUEUED"}

# Error
{"detail": "Clip not found"}  # HTTPException auto-formats this
```
