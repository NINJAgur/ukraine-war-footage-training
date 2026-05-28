import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from db.session import AsyncSessionLocal
from shared.db.models import TrainingRun, TrainingStatus

router = APIRouter()


@router.websocket("/ws/training/{run_id}")
async def training_progress(websocket: WebSocket, run_id: int):
    await websocket.accept()
    try:
        while True:
            async with AsyncSessionLocal() as session:
                run = await session.get(TrainingRun, run_id)
            if run is None:
                await websocket.send_json({"error": "run not found"})
                break
            await websocket.send_json({
                "status":  run.status.value,
                "metrics": run.metrics or {},
            })
            if run.status in (TrainingStatus.DONE, TrainingStatus.ERROR):
                break
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
