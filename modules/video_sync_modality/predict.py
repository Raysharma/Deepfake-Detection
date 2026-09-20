"""
modules/video_sync_modality/predict.py
======================================
Inference Entrypoint for Person 3 (Multimodal & Lip-Sync AI - YOUR WORK)
------------------------------------------------------------------------
Conforms strictly to common/contracts.py so Person 4's backend orchestrator
can call predict(video_path) with zero conflicts.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.contracts import VideoSyncModalityOutput
from modules.video_sync_modality.inference import evaluate_video_lip_sync


def predict(video_path: str) -> dict:
    """
    Standardized inference function for the Lip-Sync detection module.
    
    Args:
        video_path (str): Local path to the target .mp4 video.
        
    Returns:
        dict: Standardized VideoSyncModalityOutput dictionary.
    """
    raw_result = evaluate_video_lip_sync(video_path)

    output = VideoSyncModalityOutput(
        modality="video_lip_sync",
        is_fake=raw_result.get("is_fake", False),
        fake_probability=raw_result.get("fake_probability", 0.0),
        sync_confidence=raw_result.get("sync_confidence", 0.0),
        offset_frames=raw_result.get("offset_frames", 0),
        min_distance=raw_result.get("min_distance", 0.0),
        frames_evaluated=raw_result.get("frames_evaluated", 0),
        lip_landmarks_found=raw_result.get("lip_landmarks_found", False),
        detected_anomalies=raw_result.get("detected_anomalies", []),
        error_message=raw_result.get("error_message", None)
    )

    return output.to_dict()


if __name__ == "__main__":
    import sys
    test_file = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample.mp4"
    print(f"[*] Running Lip-Sync prediction on: {test_file}")
    res = predict(test_file)
    import pprint
    pprint.pprint(res)
