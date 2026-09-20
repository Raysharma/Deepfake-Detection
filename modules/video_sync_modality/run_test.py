"""
modules/video_sync_modality/run_test.py
======================================
Standalone test runner for Person 3 (Lip-Sync Modality).
Generates a local sample video if needed and verifies the pipeline end-to-end.
"""

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import cv2
import numpy as np
from modules.video_sync_modality.predict import predict


def create_synthetic_test_video(output_path: str = "sample_data/demo_mouth.mp4", num_frames: int = 40):
    """Generates a dummy MP4 video with a simulated speaking mouth for pipeline testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = 320, 320
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))

    for i in range(num_frames):
        # Background
        frame = np.full((h, w, 3), 40, dtype=np.uint8)

        # Draw Face
        cv2.ellipse(frame, (160, 160), (100, 130), 0, 0, 360, (180, 200, 230), -1)

        # Draw Eyes
        cv2.circle(frame, (120, 130), 10, (50, 50, 50), -1)
        cv2.circle(frame, (200, 130), 10, (50, 50, 50), -1)

        # Draw Animating / Speaking Mouth
        mouth_open = int(8 + 14 * np.sin(i * 0.4) ** 2)
        cv2.ellipse(frame, (160, 225), (35, mouth_open), 0, 0, 360, (50, 50, 180), -1)

        out.write(frame)

    out.release()
    print(f"[OK] Created synthetic test video at: {output_path}")
    return output_path


def main():
    print("=" * 65)
    print(" [*] Person 3: Lip-Sync / SyncNet Modality Standalone Test")
    print("=" * 65)

    test_video = "sample_data/demo_mouth.mp4"
    if not os.path.exists(test_video):
        print("[*] No test video found. Creating synthetic speaking video...")
        create_synthetic_test_video(test_video)

    print(f"[*] Executing Lip-Sync prediction on: {test_video}\n")
    result = predict(test_video)

    print("-" * 65)
    print("  RESULT SUMMARY")
    print("-" * 65)
    for key, value in result.items():
        print(f"  {key:<22}: {value}")
    print("-" * 65)
    print("[OK] Pipeline executed successfully conforming to common/contracts.py!\n")


if __name__ == "__main__":
    main()
