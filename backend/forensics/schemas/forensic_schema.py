from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class ELAResult(BaseModel):
    available: bool = True
    mean_error: Optional[float] = None
    max_error: Optional[float] = None
    std_error: Optional[float] = None
    high_error_ratio: Optional[float] = None
    image_url: Optional[str] = None
    quality: int = 90
    description: str = "Compression and editing inconsistency indicator (not standalone proof of deepfake)"
    error: Optional[str] = None

class FFTResult(BaseModel):
    available: bool = True
    mean_magnitude: Optional[float] = None
    max_magnitude: Optional[float] = None
    std_magnitude: Optional[float] = None
    high_frequency_ratio: Optional[float] = None
    image_url: Optional[str] = None
    description: str = "2D Fourier Transform frequency spectrum (analyzes high-frequency grid artifacts & decay)"
    error: Optional[str] = None

class GradCAMResult(BaseModel):
    available: bool = True
    target_class: Optional[str] = None
    target_layer: Optional[str] = None
    heatmap_url: Optional[str] = None
    overlay_url: Optional[str] = None
    description: str = "Visual explanation highlighting spatial regions contributing to deepfake classification"
    error: Optional[str] = None

class PredictionResult(BaseModel):
    label: str = Field(..., description="'deepfake' or 'real'")
    confidence: float = Field(..., description="Probability score between 0.0 and 1.0")
    class_probabilities: Dict[str, float] = Field(default_factory=dict)
    model_architecture: str = "PyTorch Deepfake ResNet-CNN"

class ForensicAnalysisResponse(BaseModel):
    success: bool = True
    scan_id: str
    filename: str
    original_image_url: str
    prediction: PredictionResult
    ela: ELAResult
    fft: FFTResult
    gradcam: GradCAMResult
    warnings: List[str] = Field(default_factory=list)
    processing_time_ms: float
    timestamp: str
