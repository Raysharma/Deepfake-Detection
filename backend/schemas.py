from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class BaseAnalysisResponse(BaseModel):
    id: str
    media_type: str
    filename: str
    is_fake: bool
    confidence_score: float = Field(..., description="Confidence between 0.0 and 1.0 (1.0 = highly confident fake)")
    verdict: str
    processing_time_ms: float
    timestamp: str

class ImageAnalysisResponse(BaseAnalysisResponse):
    facial_blending_score: float
    frequency_artifact_score: float
    compression_inconsistency_score: float
    heatmap_url: Optional[str] = None
    detected_features: List[str] = []
    explanation: str

class VideoAnalysisResponse(BaseAnalysisResponse):
    frame_count_analyzed: int
    temporal_flicker_score: float
    optical_jitter_score: float
    lip_sync_consistency_score: float
    suspicious_frames: List[int] = []
    explanation: str

class AudioAnalysisResponse(BaseAnalysisResponse):
    spectral_flux_score: float
    mfcc_anomaly_score: float
    voice_cloning_probability: float
    synthetic_silence_detected: bool
    explanation: str

class FeedbackSubmission(BaseModel):
    analysis_id: str
    user_verdict: str = Field(..., description="'CONFIRMED_FAKE', 'CONFIRMED_REAL', or 'UNCERTAIN'")
    notes: Optional[str] = ""

class SystemHealthResponse(BaseModel):
    status: str
    database_connected: bool
    registered_modules: Dict[str, str]
    uptime_seconds: float
    total_scans_performed: int
