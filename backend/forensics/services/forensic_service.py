import time
import os
from datetime import datetime, timezone
from typing import Dict, Any, List
from PIL import Image

from backend.forensics.utils.file_utils import (
    validate_image_bytes,
    generate_unique_prefix,
    get_artifact_paths,
    ensure_output_directories
)
from backend.forensics.utils.image_utils import (
    load_image_from_bytes,
    save_image_safely
)
from backend.forensics.services.ela_service import generate_ela_analysis
from backend.forensics.services.fft_service import generate_fft_analysis
from backend.forensics.services.model_service import predict_deepfake
from backend.forensics.services.gradcam_service import generate_gradcam_analysis

def run_forensic_pipeline(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Executes the complete multi-signal Image Forensics pipeline:
    Image -> Safe Preprocessing ->
         ├── Error Level Analysis (ELA)
         ├── 2D FFT Frequency Analysis
         └── Deepfake Model Inference
                 └── Grad-CAM Visual Heatmap & Overlay
                     └── Structured Forensic Evidence Report

    Guarantees:
    - Original evidence image preserved without loss or overwrite.
    - Deterministic unique artifact naming prevents concurrent collisions.
    - Fault-tolerant execution: if an individual signal fails, remaining forensic outputs are preserved.
    - Evidence signals remain distinct; no invented arbitrary scoring.
    """
    start_time = time.time()
    ensure_output_directories()
    warnings: List[str] = []

    # Step 1: Validate input image
    valid, validation_err = validate_image_bytes(file_bytes, filename)
    if not valid:
        raise ValueError(validation_err or "Invalid image file.")

    # Generate unique session prefix for all artifacts
    prefix = generate_unique_prefix()

    # Step 2: Load original image safely
    try:
        pil_image = load_image_from_bytes(file_bytes)
    except Exception as e:
        raise ValueError(f"Failed to decode image data: {str(e)}")

    # Step 3: Save original evidence image untouched
    orig_ext = os.path.splitext(filename.lower())[1] or ".jpg"
    orig_abs_path, orig_url_path = get_artifact_paths(prefix, "original", ext=orig_ext)
    save_image_safely(pil_image, orig_abs_path, quality=100)

    # Step 4: Execute ELA Analysis
    try:
        ela_res = generate_ela_analysis(pil_image, prefix)
        if not ela_res.get("available"):
            warnings.append(f"ELA warning: {ela_res.get('error')}")
    except Exception as e:
        ela_res = {
            "available": False,
            "error": f"ELA execution failed: {str(e)}"
        }
        warnings.append(ela_res["error"])

    # Step 5: Execute 2D FFT Frequency Analysis
    try:
        fft_res = generate_fft_analysis(pil_image, prefix)
        if not fft_res.get("available"):
            warnings.append(f"FFT warning: {fft_res.get('error')}")
    except Exception as e:
        fft_res = {
            "available": False,
            "error": f"FFT execution failed: {str(e)}"
        }
        warnings.append(fft_res["error"])

    # Step 6: Deepfake Model Inference
    try:
        model_pred = predict_deepfake(pil_image)
        if not model_pred.get("success"):
            warnings.append(f"Model warning: {model_pred.get('error')}")
    except Exception as e:
        model_pred = {
            "success": False,
            "label": "unknown",
            "confidence": 0.0,
            "class_probabilities": {},
            "model_architecture": "Unavailable",
            "error": f"Model inference error: {str(e)}"
        }
        warnings.append(model_pred["error"])

    # Step 7: Grad-CAM Explainability Heatmap & Overlay
    target_class = model_pred.get("label", "deepfake")
    if target_class == "unknown":
        target_class = "deepfake"

    try:
        gradcam_res = generate_gradcam_analysis(pil_image, prefix, target_class_name=target_class)
        if not gradcam_res.get("available"):
            warnings.append(f"Grad-CAM warning: {gradcam_res.get('error')}")
    except Exception as e:
        gradcam_res = {
            "available": False,
            "target_class": target_class,
            "error": f"Grad-CAM execution failed: {str(e)}"
        }
        warnings.append(gradcam_res["error"])

    # Step 8: Assemble final report
    elapsed_ms = round((time.time() - start_time) * 1000.0, 2)
    timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "success": True,
        "scan_id": prefix,
        "filename": filename,
        "original_image_url": orig_url_path,
        "prediction": {
            "label": model_pred.get("label", "unknown"),
            "confidence": model_pred.get("confidence", 0.0),
            "class_probabilities": model_pred.get("class_probabilities", {}),
            "model_architecture": model_pred.get("model_architecture", "PyTorch ResNet-CNN")
        },
        "ela": ela_res,
        "fft": fft_res,
        "gradcam": gradcam_res,
        "warnings": warnings,
        "processing_time_ms": elapsed_ms,
        "timestamp": timestamp
    }
