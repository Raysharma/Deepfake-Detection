from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import time
from datetime import datetime

from backend.schemas import (
    ImageAnalysisResponse,
    VideoAnalysisResponse,
    AudioAnalysisResponse,
    FeedbackSubmission,
    SystemHealthResponse
)
from backend.database import (
    init_db,
    log_scan,
    log_user_feedback,
    get_recent_scans,
    get_scan_statistics
)
from backend.pipeline.image_pipeline import analyze_image_bytes
from backend.pipeline.video_pipeline import analyze_video_file
from backend.pipeline.audio_pipeline import analyze_audio_bytes
from backend.pipeline.forensics.api.forensic_routes import router as forensic_router
from backend.pipeline.forensics.services.model_service import get_or_load_model
from backend.pipeline.forensics.utils.file_utils import ensure_output_directories, OUTPUT_FORENSIC_DIR

# Initialize FastAPI App
app = FastAPI(
    title="AI Deepfake Detection Engine API",
    description="Multi-modal deepfake detection API with explainability layer and active feedback logging.",
    version="1.0.0"
)

# Enable CORS for local cross-origin development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

# Initialize DB immediately on module load and on startup
init_db()

@app.on_event("startup")
def startup_event():
    init_db()
    ensure_output_directories()
    get_or_load_model()

@app.get("/api/health", response_model=SystemHealthResponse, tags=["System"])
def check_health():
    stats = get_scan_statistics()
    return {
        "status": "ONLINE",
        "database_connected": True,
        "registered_modules": {
            "image_forensics": "Active (Independent ELA + 2D FFT + ResNet + Grad-CAM)",
            "image_detector": "Active (ELA + FFT + GradCAM Overlay)",
            "video_detector": "Active (Temporal Jitter & Frame Delta)",
            "audio_detector": "Active (Spectral Flux & MFCC)",
            "feedback_loop": "Active (SQLite Audit Log)"
        },
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_scans_performed": stats["total_scans"]
    }

@app.post("/api/analyze/image", response_model=ImageAnalysisResponse, tags=["Inference"])
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Image required.")
    
    contents = await file.read()
    result = analyze_image_bytes(contents, file.filename)
    
    scan_id = log_scan(
        media_type="image",
        filename=file.filename,
        is_fake=result["is_fake"],
        confidence_score=result["confidence_score"],
        verdict=result["verdict"],
        processing_time_ms=result["processing_time_ms"],
        details=result
    )
    
    return {
        "id": scan_id,
        "media_type": "image",
        "filename": file.filename,
        "is_fake": result["is_fake"],
        "confidence_score": result["confidence_score"],
        "verdict": result["verdict"],
        "facial_blending_score": result["facial_blending_score"],
        "frequency_artifact_score": result["frequency_artifact_score"],
        "compression_inconsistency_score": result["compression_inconsistency_score"],
        "heatmap_url": result["heatmap_url"],
        "detected_features": result["detected_features"],
        "explanation": result["explanation"],
        "processing_time_ms": result["processing_time_ms"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/analyze/video", response_model=VideoAnalysisResponse, tags=["Inference"])
async def analyze_video(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_video_file(contents, file.filename)
    
    scan_id = log_scan(
        media_type="video",
        filename=file.filename,
        is_fake=result["is_fake"],
        confidence_score=result["confidence_score"],
        verdict=result["verdict"],
        processing_time_ms=result["processing_time_ms"],
        details=result
    )
    
    return {
        "id": scan_id,
        "media_type": "video",
        "filename": file.filename,
        "is_fake": result["is_fake"],
        "confidence_score": result["confidence_score"],
        "verdict": result["verdict"],
        "frame_count_analyzed": result["frame_count_analyzed"],
        "temporal_flicker_score": result["temporal_flicker_score"],
        "optical_jitter_score": result["optical_jitter_score"],
        "lip_sync_consistency_score": result["lip_sync_consistency_score"],
        "suspicious_frames": result["suspicious_frames"],
        "explanation": result["explanation"],
        "processing_time_ms": result["processing_time_ms"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/analyze/audio", response_model=AudioAnalysisResponse, tags=["Inference"])
async def analyze_audio(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_audio_bytes(contents, file.filename)
    
    scan_id = log_scan(
        media_type="audio",
        filename=file.filename,
        is_fake=result["is_fake"],
        confidence_score=result["confidence_score"],
        verdict=result["verdict"],
        processing_time_ms=result["processing_time_ms"],
        details=result
    )
    
    return {
        "id": scan_id,
        "media_type": "audio",
        "filename": file.filename,
        "is_fake": result["is_fake"],
        "confidence_score": result["confidence_score"],
        "verdict": result["verdict"],
        "spectral_flux_score": result["spectral_flux_score"],
        "mfcc_anomaly_score": result["mfcc_anomaly_score"],
        "voice_cloning_probability": result["voice_cloning_probability"],
        "synthetic_silence_detected": result["synthetic_silence_detected"],
        "explanation": result["explanation"],
        "processing_time_ms": result["processing_time_ms"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/scans/recent", tags=["Audit & Feedback"])
def list_recent_scans():
    return get_recent_scans(limit=10)

@app.get("/api/scans/stats", tags=["Audit & Feedback"])
def scan_stats():
    return get_scan_statistics()

@app.post("/api/feedback", tags=["Audit & Feedback"])
def submit_feedback(feedback: FeedbackSubmission):
    feedback_id = log_user_feedback(
        scan_id=feedback.analysis_id,
        user_verdict=feedback.user_verdict,
        notes=feedback.notes or ""
    )
    return {"status": "SUCCESS", "feedback_id": feedback_id, "message": "Feedback recorded for active retraining loop."}

# Register Image Forensics Router
app.include_router(forensic_router)

# Mount forensic artifact outputs directory for web visualization
forensic_dir = ensure_output_directories()
app.mount("/outputs/forensic", StaticFiles(directory=forensic_dir), name="forensic_outputs")

# Mount static folder for web dashboard
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Deepfake Detection API is running. Visit /docs for Swagger API explorer."}
