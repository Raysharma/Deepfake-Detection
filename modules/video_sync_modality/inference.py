"""
modules/video_sync_modality/inference.py
========================================
Core Lip-Sync Inconsistency & Temporal Alignment Inference Engine
"""

import os
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

from modules.video_sync_modality.data_preprocessing.lip_extractor import extract_lip_rois_from_video
from modules.video_sync_modality.data_preprocessing.audio_extractor import extract_audio_and_mfcc
from modules.video_sync_modality.models.syncnet import SyncNet


# Global model cache to avoid re-instantiating every call
_SYNCNET_MODEL: SyncNet = None


def get_syncnet_model() -> SyncNet:
    """Loads and caches the SyncNet model in evaluation mode."""
    global _SYNCNET_MODEL
    if _SYNCNET_MODEL is None:
        model = SyncNet()
        model.eval()
        _SYNCNET_MODEL = model
    return _SYNCNET_MODEL


def evaluate_video_lip_sync(
    video_path: str,
    device: str = "cpu",
    max_search_offset: int = 10
) -> Dict[str, Any]:
    """
    Evaluates audio-visual lip synchronization of a video file.
    
    Args:
        video_path (str): Path to the target video.
        device (str): PyTorch device ('cpu' or 'cuda').
        max_search_offset (int): Number of frames to scan forward and backward for audio-lip delay.
        
    Returns:
        Dict[str, Any]: Detailed alignment metrics and deepfake probability.
    """
    if not os.path.exists(video_path):
        return {
            "is_fake": False,
            "fake_probability": 0.0,
            "sync_confidence": 0.0,
            "offset_frames": 0,
            "min_distance": 0.0,
            "frames_evaluated": 0,
            "lip_landmarks_found": False,
            "detected_anomalies": ["file_not_found"],
            "error_message": f"File not found: {video_path}"
        }

    # 1. Extract visual mouth frames (128, 128)
    lip_frames, landmarks_found = extract_lip_rois_from_video(video_path, target_size=(128, 128), max_frames=25)
    total_frames = len(lip_frames)

    if total_frames < 5:
        # Not enough frames to analyze
        return {
            "is_fake": False,
            "fake_probability": 0.50,
            "sync_confidence": 0.20,
            "offset_frames": 0,
            "min_distance": 1.2,
            "frames_evaluated": total_frames,
            "lip_landmarks_found": landmarks_found,
            "detected_anomalies": ["insufficient_video_frames"],
            "error_message": "Video too short for 5-frame sequence analysis."
        }

    # 2. Extract audio acoustic features (12 MFCC coefficients)
    mfcc_features, sr = extract_audio_and_mfcc(video_path, n_mfcc=12)

    # 3. Form 5-frame sliding visual windows: shape (N, 3, 5, 128, 128)
    num_sequences = min(total_frames - 5 + 1, 20)
    visual_batch = np.zeros((num_sequences, 3, 5, 128, 128), dtype=np.float32)
    for i in range(num_sequences):
        seq = np.stack(lip_frames[i : i + 5], axis=0)  # (5, 128, 128)
        # Duplicate to 3 channels (RGB)
        seq_3c = np.stack([seq, seq, seq], axis=0)      # (3, 5, 128, 128)
        visual_batch[i] = seq_3c

    model = get_syncnet_model()
    model.eval()

    # Pre-compute visual embeddings once (N, 1024)
    with torch.no_grad():
        v_tensor = torch.from_numpy(visual_batch).to(device)
        v_emb = model.forward_visual(v_tensor)  # (N, 1024)

    # 4. Search across temporal offsets (e.g. -10 to +10 frames)
    # 1 video frame = 4 acoustic hops (at 25 fps, 100Hz audio feature rate)
    offsets = list(range(-max_search_offset, max_search_offset + 1))
    offset_distances: List[float] = []

    for off in offsets:
        audio_batch = np.zeros((num_sequences, 1, 12, 20), dtype=np.float32)
        for i in range(num_sequences):
            audio_frame_start = (i + off) * 4
            audio_frame_end = audio_frame_start + 20

            if audio_frame_start >= 0 and audio_frame_end <= len(mfcc_features):
                chunk = mfcc_features[audio_frame_start:audio_frame_end, :12]
            elif audio_frame_start < 0:
                # Pad beginning
                pad_len = min(20, -audio_frame_start)
                avail = mfcc_features[:max(0, audio_frame_end), :12]
                chunk = np.pad(avail, ((pad_len, max(0, 20 - pad_len - len(avail))), (0, 0)))[:20]
            else:
                # Pad end
                avail = mfcc_features[max(0, audio_frame_start):, :12]
                chunk = np.pad(avail, ((0, max(0, 20 - len(avail))), (0, 0)))[:20]

            if chunk.shape[0] < 20:
                chunk = np.pad(chunk, ((0, 20 - chunk.shape[0]), (0, 0)))

            audio_batch[i, 0] = chunk[:20, :12].T  # (12, 20)

        with torch.no_grad():
            a_tensor = torch.from_numpy(audio_batch).to(device)
            a_emb = model.forward_audio(a_tensor)  # (N, 1024)
            dists = torch.norm(v_emb - a_emb, dim=1).cpu().numpy()
            offset_distances.append(float(np.mean(dists)))

    # Find the offset with minimal feature distance
    best_idx = int(np.argmin(offset_distances))
    best_offset = offsets[best_idx]
    min_distance = float(offset_distances[best_idx])
    zero_offset_idx = offsets.index(0)
    zero_dist = float(offset_distances[zero_offset_idx])

    # Classification Calibration:
    # Human speech and video encoders typically have standard broadcast tolerance of ±6 frames (±240ms).
    # Desynchronized deepfakes, re-dubbing, or manipulated lip synthesis show severe temporal delays (|offset| >= 8)
    # or elevated cross-modal distance (min_distance > 0.65).
    is_fake = bool(abs(best_offset) >= 8 or min_distance > 0.65)

    if is_fake:
        # High fake probability based on severity of lag
        severity = min(abs(best_offset) / 12.0, 1.0)
        fake_prob = float(np.clip(0.82 + 0.15 * severity, 0.75, 0.98))
        sync_conf = float(np.clip(1.0 - fake_prob, 0.02, 0.25))
    else:
        # Authentic: low fake probability, high synchrony confidence
        fake_prob = float(np.clip(min_distance * 0.28, 0.04, 0.18))
        sync_conf = float(np.clip(1.0 - fake_prob, 0.82, 0.98))

    detected_anomalies = []
    if is_fake:
        detected_anomalies.append("Lip-Sync Mismatch: Mouth shapes do not correspond with spoken phonemes")
        if abs(best_offset) >= 8:
            detected_anomalies.append(f"Significant temporal delay detected: {best_offset} frames offset")
        if min_distance > 0.65:
            detected_anomalies.append(f"High cross-modal feature divergence: distance={min_distance:.4f}")
    else:
        detected_anomalies.append("Audio-Visual Synchrony Confirmed (mouth phonemes and acoustic timing align)")

    return {
        "is_fake": is_fake,
        "fake_probability": round(fake_prob, 4),
        "sync_confidence": round(sync_conf, 4),
        "offset_frames": best_offset,
        "min_distance": round(min_distance, 4),
        "frames_evaluated": num_sequences,
        "lip_landmarks_found": landmarks_found,
        "detected_anomalies": detected_anomalies,
        "error_message": None
    }
