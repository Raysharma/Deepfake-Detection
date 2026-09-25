from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from .config import UPLOAD_DIR
from celery.result import AsyncResult
from .celery_worker import celery_app, run_deepfake_analysis


app = FastAPI(
    title="Deepfake Detection Backend",
    description="Backend API for the Multimodal Deepfake Detection System",
    version="1.0.0",
)


ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".mp3",
}


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Deepfake Detection Backend is running."
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "backend_service"
    }


@app.post("/api/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}"
        )

    file_path = UPLOAD_DIR / file.filename

    with file_path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    task = run_deepfake_analysis.delay(str(file_path))
    return {
        "status": "processing",
        "filename": file.filename,
        "path": str(file_path),
        "task_id": task.id
    }


@app.get("/api/v1/status/{task_id}")
def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": task.status,
    }

    if task.successful():
        response["result"] = task.result

    elif task.failed():
        response["error"] = str(task.result)

    return response