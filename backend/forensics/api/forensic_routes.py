from fastapi import APIRouter, UploadFile, File, HTTPException, status
import logging
from typing import Dict, Any

from backend.forensics.schemas.forensic_schema import ForensicAnalysisResponse
from backend.forensics.services.forensic_service import run_forensic_pipeline
from backend.forensics.services.model_service import get_or_load_model, HAS_TORCH

try:
    from backend.database import log_scan
    HAS_DB = True
except ImportError:
    HAS_DB = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/forensics", tags=["Image Forensics"])

MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB

@router.get("/health", tags=["Image Forensics"])
def forensic_health():
    """Returns the operational status of all forensic sub-modules."""
    model = get_or_load_model()
    return {
        "status": "ONLINE",
        "submodules": {
            "ela_analyzer": "Ready (In-memory JPEG Compression Differential)",
            "fft_analyzer": "Ready (2D Fast Fourier Spectral Decomposition)",
            "deepfake_model": "Online" if model is not None else "Offline (PyTorch Required)",
            "gradcam_engine": "Ready" if model is not None else "Offline"
        },
        "torch_available": HAS_TORCH
    }

@router.post("/analyze", response_model=ForensicAnalysisResponse, tags=["Image Forensics"])
async def analyze_image_forensics(file: UploadFile = File(...)):
    """
    Primary Image Forensics Endpoint:
    Accepts multipart/form-data image (JPG, PNG, WEBP).
    Runs ELA, 2D FFT, Deepfake Model Inference, and Grad-CAM explainability.
    Returns structured forensic metrics and artifact URLs.
    """
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename missing."
        )

    # Read bytes safely
    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(file_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed limit of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
        )

    # Run complete forensic pipeline
    try:
        report = run_forensic_pipeline(file_bytes, file.filename)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Internal error processing image forensics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic pipeline processing error: {str(e)}"
        )

    # Log to SQLite audit database if available
    if HAS_DB:
        try:
            is_fake = (report["prediction"]["label"] == "deepfake")
            confidence = report["prediction"]["confidence"]
            verdict = "DEEPFAKE DETECTED" if is_fake else "AUTHENTIC MEDIA"
            log_scan(
                media_type="image_forensics",
                filename=file.filename,
                is_fake=is_fake,
                confidence_score=confidence,
                verdict=verdict,
                processing_time_ms=report["processing_time_ms"],
                details=report
            )
        except Exception as db_err:
            logger.warning(f"Could not log forensic scan to audit db: {db_err}")

    return report
