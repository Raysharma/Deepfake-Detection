from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from .orchestrator import run_full_pipeline
from fastapi.middleware.cors import CORSMiddleware
from .config import UPLOAD_DIR
from celery.result import AsyncResult
from .celery_worker import celery_app, run_deepfake_analysis
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="Deepfake Detection Backend",
    description="Backend API for the Multimodal Deepfake Detection System",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    # Step 1: Save the uploaded file to disk first!
    with file_path.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

        # Step 2: Direct High-Speed Execution (Zero Redis timeout!)
    result = run_full_pipeline(str(file_path))
    return {
        "status": "completed",
        "filename": file.filename,
        "result": result
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

# Serve generated forensic heatmaps (Grad-CAM & Spectrograms)
import os
os.makedirs("temp_uploads", exist_ok=True)
app.mount("/temp_uploads", StaticFiles(directory="temp_uploads"), name="temp_uploads")

# Mount frontend at the very bottom so API routes take priority!
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
