"""
modules/video_sync_modality/data_preprocessing/lip_extractor.py
==============================================================
Extracts Mouth / Lip Region of Interest (ROI) from video frames.
Supports MediaPipe Face Mesh with OpenCV facial geometry fallback.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

# MediaPipe lip landmark indices (outer and inner lips)
LIP_LANDMARK_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
]


def _extract_lips_mediapipe(frame: np.ndarray, face_mesh) -> Optional[np.ndarray]:
    """Uses MediaPipe Face Mesh to isolate tightly cropped mouth bounding box."""
    h, w = frame.shape[:2]
    # Downscale frame to max 360p before passing to MediaPipe (5x faster!)
    scale = 1.0
    if max(h, w) > 480:
        scale = 480.0 / max(h, w)
        small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)


    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    xs = [int(landmarks[idx].x * w) for idx in LIP_LANDMARK_INDICES]
    ys = [int(landmarks[idx].y * h) for idx in LIP_LANDMARK_INDICES]

    x_min, x_max = max(0, min(xs)), min(w, max(xs))
    y_min, y_max = max(0, min(ys)), min(h, max(ys))

    # Add margin padding around lips
    pad_x = int((x_max - x_min) * 0.25)
    pad_y = int((y_max - y_min) * 0.25)

    x1 = max(0, x_min - pad_x)
    x2 = min(w, x_max + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(h, y_max + pad_y)

    mouth_crop = frame[y1:y2, x1:x2]
    if mouth_crop.size == 0:
        return None
    return mouth_crop


def _extract_lips_fallback(frame: np.ndarray) -> np.ndarray:
    """
    Fallback when MediaPipe is not loaded:
    Uses standard lower-facial geometric ROI heuristic (lower half of face).
    """
    h, w = frame.shape[:2]
    # Crop central lower region where mouth is anatomically located
    y1 = int(h * 0.60)
    y2 = int(h * 0.90)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)
    return frame[y1:y2, x1:x2]


def extract_lip_rois_from_video(
    video_path: str,
    target_size: Tuple[int, int] = (96, 96),
    max_frames: int = 75
) -> Tuple[List[np.ndarray], bool]:
    """
    Extracts standardized grayscale mouth ROI frames from video file.
    
    Args:
        video_path (str): Absolute or relative path to .mp4 video.
        target_size (int, int): Normalized output dimensions (width, height).
        max_frames (int): Maximum frames to sample for fast inference.
        
    Returns:
        Tuple[List[np.ndarray], bool]:
            - List of normalized (96, 96) grayscale numpy arrays.
            - Boolean indicating whether facial landmarks were successfully tracked.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], False

    lip_frames: List[np.ndarray] = []
    landmarks_found = False

    # Attempt to initialize MediaPipe FaceMesh
    face_mesh = None
    try:
        import mediapipe.python.solutions.face_mesh as mp_face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception:
        face_mesh = None

    frame_idx = 0
    while cap.isOpened() and len(lip_frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        mouth_crop = None
        if face_mesh is not None:
            mouth_crop = _extract_lips_mediapipe(frame, face_mesh)
            if mouth_crop is not None:
                landmarks_found = True

        if mouth_crop is None:
            mouth_crop = _extract_lips_fallback(frame)

        # Convert to single-channel Grayscale and resize to standard SyncNet shape (96, 96)
        if len(mouth_crop.shape) == 3:
            gray_mouth = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_mouth = mouth_crop

        resized = cv2.resize(gray_mouth, target_size)
        # Normalize pixel values to [0.0, 1.0]
        norm_mouth = resized.astype(np.float32) / 255.0
        lip_frames.append(norm_mouth)
        frame_idx += 1

    cap.release()
    if face_mesh:
        face_mesh.close()

    return lip_frames, landmarks_found
