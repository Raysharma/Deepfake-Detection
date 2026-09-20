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


def run_full_pipeline(video_path: str) -> dict:
    """
    Called inside the Celery worker when a new video is uploaded.
    """
    start_time = time.time()
    task_id = str(uuid.uuid4())[:8]

    # 1. Execute the 3 modality modules
    image_result = predict_image(video_path)
    audio_result = predict_audio(video_path)
    sync_result = predict_sync(video_path)

    # 2. Weighted Aggregation (35% Spatial, 35% Audio, 30% Lip-Sync)
    w_image = 0.35
    w_audio = 0.35
    w_sync = 0.30

    overall_fake_prob = (
        (image_result.get("fake_probability", 0.0) * w_image) +
        (audio_result.get("fake_probability", 0.0) * w_audio) +
        (sync_result.get("fake_probability", 0.0) * w_sync)
    )

    verdict = "DEEPFAKE" if overall_fake_prob >= 0.50 else "AUTHENTIC"
    elapsed = round(time.time() - start_time, 2)

    # 3. Create standardized MongoDB report
    report = MultimodalFinalReport(
        task_id=task_id,
        video_filename=video_path,
        overall_verdict=verdict,
        overall_fake_probability=round(overall_fake_prob, 4),
        weights={"image": w_image, "audio": w_audio, "sync": w_sync},
        image_analysis=image_result,
        audio_analysis=audio_result,
        video_sync_analysis=sync_result,
        processing_time_sec=elapsed
    )

    return report.to_dict()


if __name__ == "__main__":
    print("Testing Orchestrator with dummy sample video:")
    result = run_full_pipeline("sample.mp4")
    import pprint
    pprint.pprint(result)
