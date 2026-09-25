"""
modules/backend_service/orchestrator_example.py
===============================================
Blueprint demonstrating how Person 4 (Backend Engineer) connects the 3 ML modules.
"""

import time
import uuid
import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.contracts import MultimodalFinalReport

# Import the 3 ML prediction contracts
from modules.image_modality.predict import predict as predict_image
from modules.audio_modality.predict import predict as predict_audio
from modules.video_sync_modality.predict import predict as predict_sync


def run_full_pipeline(file_path: str) -> dict:
    """
    Analyzes uploaded media by intelligently routing based on file type:
    - Images (.jpg, .png): Spatial Image Forensics (ELA + 2D FFT)
    - Audio (.wav, .mp3): Mel-Spectrogram CNN
    - Video (.mp4, .avi, .mov): Full Multimodal Tri-Fusion (Image + Audio + Lip-Sync)
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file_path)[1].lower()

    # Case 1: Still Image (.jpg, .png, etc.)
    if ext in [".jpg", ".jpeg", ".png"]:
        image_result = predict_image(file_path)
        overall_fake_prob = image_result.get("fake_probability", 0.0)
        verdict = "FAKE" if (image_result.get("is_fake", False) or overall_fake_prob >= 0.50) else "REAL"

        report = MultimodalFinalReport(
            task_id=task_id,
            video_filename=file_path,
            overall_verdict=verdict,
            overall_fake_probability=round(overall_fake_prob, 4),
            weights={"image": 1.0, "audio": 0.0, "sync": 0.0},
            image_analysis=image_result,
            audio_analysis={"modality": "audio_acoustic", "is_fake": False, "fake_probability": 0.0, "detected_anomalies": ["not_applicable_for_image"]},
            video_sync_analysis={"modality": "video_lip_sync", "is_fake": False, "fake_probability": 0.0, "detected_anomalies": ["not_applicable_for_image"]},
            processing_time_sec=round(time.time() - start_time, 2)
        )
        return report.to_dict()

    # Case 2: Audio Only (.wav, .mp3)
    elif ext in [".wav", ".mp3"]:
        audio_result = predict_audio(file_path)
        overall_fake_prob = audio_result.get("fake_probability", 0.0)
        verdict = "FAKE" if (audio_result.get("is_fake", False) or overall_fake_prob >= 0.50) else "REAL"

        report = MultimodalFinalReport(
            task_id=task_id,
            video_filename=file_path,
            overall_verdict=verdict,
            overall_fake_probability=round(overall_fake_prob, 4),
            weights={"image": 0.0, "audio": 1.0, "sync": 0.0},
            image_analysis={"modality": "image_spatial", "is_fake": False, "fake_probability": 0.0, "detected_artifacts": ["not_applicable_for_audio"]},
            audio_analysis=audio_result,
            video_sync_analysis={"modality": "video_lip_sync", "is_fake": False, "fake_probability": 0.0, "detected_anomalies": ["not_applicable_for_audio"]},
            processing_time_sec=round(time.time() - start_time, 2)
        )
        return report.to_dict()

    # Case 3: Video (.mp4, .avi, .mov) -> Full Multimodal Tri-Fusion
    else:
        image_result = predict_image(file_path)
        audio_result = predict_audio(file_path)
        sync_result = predict_sync(file_path)

        w_image = 0.35
        w_audio = 0.35
        w_sync = 0.30

        image_fake = image_result.get("fake_probability", 0.0)
        audio_fake = audio_result.get("fake_probability", 0.0)
        sync_fake = sync_result.get("fake_probability", 0.0)

        max_anomaly = max(image_fake, audio_fake, sync_fake)
        weighted_avg = (image_fake * w_image) + (audio_fake * w_audio) + (sync_fake * w_sync)
        overall_fake_prob = max(weighted_avg, max_anomaly)

        is_any_fake = (
            image_result.get("is_fake", False) or
            audio_result.get("is_fake", False) or
            sync_result.get("is_fake", False)
        )
        verdict = "FAKE" if (overall_fake_prob >= 0.50 or is_any_fake) else "REAL"

        report = MultimodalFinalReport(
            task_id=task_id,
            video_filename=file_path,
            overall_verdict=verdict,
            overall_fake_probability=round(overall_fake_prob, 4),
            weights={"image": w_image, "audio": w_audio, "sync": w_sync},
            image_analysis=image_result,
            audio_analysis=audio_result,
            video_sync_analysis=sync_result,
            processing_time_sec=round(time.time() - start_time, 2)
        )
        return report.to_dict()
