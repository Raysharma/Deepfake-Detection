"""
common/contracts.py
===================
Shared Data Contracts & Standardized Output Schemas
---------------------------------------------------
All 3 AI Modality teams (Image, Audio, Video/Sync) must adhere to these
dictionary output schemas. This ensures Person 4's FastAPI + Celery backend
orchestrator can process and store results in MongoDB without any merge conflicts.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ImageModalityOutput:
    """
    Standard Output Contract for Person 1 (Image & Spatial AI)
    """
    modality: str = "image_spatial"
    is_fake: bool = False
    fake_probability: float = 0.0          # 0.0 to 1.0 (1.0 = highly confident fake)
    spatial_score: float = 0.0             # Model raw classification score
    frames_analyzed: int = 0
    face_detected: bool = True
    heatmap_path: Optional[str] = None     # Saved Grad-CAM face heatmap image path
    detected_artifacts: List[str] = None   # e.g., ["chin_blurring", "boundary_warp"]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["detected_artifacts"] is None:
            d["detected_artifacts"] = []
        return d


@dataclass
class AudioModalityOutput:
    """
    Standard Output Contract for Person 2 (Audio & Acoustic AI)
    """
    modality: str = "audio_acoustic"
    is_fake: bool = False
    fake_probability: float = 0.0          # 0.0 to 1.0 (1.0 = highly confident clone)
    frequency_score: float = 0.0           # Model classification score on spectrogram
    audio_duration_sec: float = 0.0
    spectrogram_heatmap_path: Optional[str] = None  # Saved Grad-CAM spectrogram image
    detected_anomalies: List[str] = None   # e.g., ["unnatural_spectral_cutoff", "robotic_harmonic"]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["detected_anomalies"] is None:
            d["detected_anomalies"] = []
        return d


@dataclass
class VideoSyncModalityOutput:
    """
    Standard Output Contract for Person 3 (Multimodal & Lip-Sync AI - YOUR ROLE)
    """
    modality: str = "video_lip_sync"
    is_fake: bool = False                  # True if out-of-sync / lip manipulation detected
    fake_probability: float = 0.0          # 0.0 to 1.0
    sync_confidence: float = 0.0           # Higher = higher synchronization match
    offset_frames: int = 0                 # Temporal delay between lips and audio in frames (0 = in-sync)
    min_distance: float = 0.0              # SyncNet L2 / cosine feature distance
    frames_evaluated: int = 0
    lip_landmarks_found: bool = True
    detected_anomalies: List[str] = None   # e.g., ["lip_sync_mismatch", "delayed_phoneme_correlation"]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["detected_anomalies"] is None:
            d["detected_anomalies"] = []
        return d


@dataclass
class MultimodalFinalReport:
    """
    Aggregated Document stored in MongoDB by Person 4 (Backend Engineer)
    """
    task_id: str
    video_filename: str
    overall_verdict: str                  # "REAL" or "FAKE"
    overall_fake_probability: float
    weights: Dict[str, float]             # e.g. {"image": 0.35, "audio": 0.35, "sync": 0.30}
    image_analysis: Dict[str, Any]
    audio_analysis: Dict[str, Any]
    video_sync_analysis: Dict[str, Any]
    processing_time_sec: float
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if not d["created_at"]:
            d["created_at"] = datetime.utcnow().isoformat()
        return d
