# Image Forensics Module for Deepfake Detection

An enterprise-ready, modular Image Forensics and Explainability pipeline built for integration into the **AI-Powered Deepfake Detection System**.

---

## 1. System Architecture

The Image Forensics module follows a decoupled multi-signal verification architecture:

```
Uploaded Image (JPG, PNG, WEBP)
       │
       ▼
Safe Ingest & Preprocessing (MIME/Magic Bytes Check, RGB Normalization)
       │
       ├─────────────────────────────────┬───────────────────────────────┐
       ▼                                 ▼                               ▼
Error Level Analysis (ELA)       2D FFT Spectral Analysis      Deepfake CNN Model
- Recompression Differential     - np.fft.fft2() + fftshift    - PyTorch ResNet Architecture
- Error Distribution Statistics  - Magnitude Spectrum (log)    - Eval Mode / No-Grad
- Artifact Localization Map      - High/Low Freq Ratio         - Class Probabilities
       │                                 │                               │
       │                                 │                               ▼
       │                                 │                     Grad-CAM Explainability
       │                                 │                     - Target Conv Layer (layer4)
       │                                 │                     - Gradient Backpropagation
       │                                 │                     - Heatmap & Overlay Maps
       │                                 │                               │
       └─────────────────────────────────┴───────────────────────────────┘
                                         │
                                         ▼
                     Structured Forensic Evidence Report
                         (JSON REST API & Frontend UI)
```

### Module Layout
```
Deepfake-Detection/
├── backend/
│   ├── app.py                             # FastAPI Application & Router Mounts
│   ├── database.py                        # SQLite Audit Store
│   └── forensics/                         # Dedicated Image Forensics Subsystem
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   └── forensic_routes.py         # POST /api/forensics/analyze
│       ├── schemas/
│       │   ├── __init__.py
│       │   └── forensic_schema.py         # Pydantic v2 Evidence Contracts
│       ├── services/
│       │   ├── __init__.py
│       │   ├── ela_service.py             # Error Level Analysis Engine
│       │   ├── fft_service.py             # 2D Fast Fourier Transform Engine
│       │   ├── model_service.py           # Deepfake Neural Model & Inference
│       │   ├── gradcam_service.py         # Grad-CAM Heatmap & Overlay Generator
│       │   └── forensic_service.py        # Evidence Fusion Orchestrator
│       └── utils/
│           ├── __init__.py
│           ├── file_utils.py              # Path Validation, Security & Temp Cleanup
│           └── image_utils.py             # Color Spaces, Resizing & Tensors
├── outputs/
│   └── forensic/                          # Deterministic Artifact Storage
├── tests/
│   ├── test_ela.py                        # ELA Unit & Edge Tests
│   ├── test_fft.py                        # 2D FFT Unit & Edge Tests
│   ├── test_model.py                      # Model Inference Tests
│   ├── test_gradcam.py                    # Grad-CAM Backprop Tests
│   └── test_forensic_api.py               # REST API Endpoint Tests
└── IMAGE_FORENSICS.md                     # Technical Documentation
```

---

## 2. Error Level Analysis (ELA) Methodology

Error Level Analysis detects variations in JPEG compression levels across an image:
1. **Recompression**: The original image is re-saved to an in-memory byte buffer at a configurable quality factor ($Q = 90$ by default).
2. **Pixel-wise Subtraction**: The absolute difference $|I_{\text{original}} - I_{\text{recompressed}}|$ is computed across all color channels.
3. **Statistical Quantification**:
   - $\text{Mean Error} = \frac{1}{HW}\sum_{x,y} |I_1(x,y) - I_2(x,y)|$
   - $\text{Max Error} = \max_{x,y} |I_1(x,y) - I_2(x,y)|$
   - $\text{High-Error Pixel Ratio} = \frac{\#\{p \mid \max_c |I_{1,c}(p) - I_{2,c}(p)| > \tau\}}{HW}$ (where $\tau = 20.0$ default).
4. **Contrast Amplification**: Differences are scaled by $255.0 / \max(\text{error})$ so subtle compression boundaries become visible to forensic examiners.

---

## 3. 2D Fast Fourier Transform (FFT) Methodology

Generative models (GANs and Diffusion architectures) frequently introduce subtle periodic grid artifacts or abnormal spectral decay due to upsampling and deconvolution layers.
1. **Grayscale Conversion**: Image converted via standard perceptual luminance weighting ($Y = 0.299R + 0.587G + 0.114B$).
2. **2D FFT**: Discrete 2D Fourier transform $F(u, v) = \sum_{x,y} f(x,y) e^{-j 2\pi (\frac{ux}{W} + \frac{vy}{H})}$.
3. **Quadrant Shift**: DC zero-frequency component shifted to the center matrix coordinates.
4. **Logarithmic Scaling**: $S(u, v) = \log(1 + |F(u, v)|)$.
5. **Radial Masking**: A circular mask of radius $r = \min(H, W) / 6$ separates low-frequency structural components from high-frequency fine details. The high-to-low frequency ratio $R_{HF/LF} = \frac{\text{Mean}(S_{\text{outer}})}{\text{Mean}(S_{\text{inner}}) + \epsilon}$ quantifies high-frequency energy persistence.

---

## 4. Deepfake Model Inference

- **Architecture**: PyTorch ResNet-CNN classifier (`DeepfakeCNNClassifier`) with residual blocks, global average pooling, and binary classification head (`[0: 'real', 1: 'deepfake']`).
- **Singleton Caching**: Model is initialized and warmed up once on startup (`backend/forensics/services/model_service.py`), avoiding repeated model loading overhead per request.
- **Evaluation Mode**: Executed strictly with `model.eval()` and `torch.no_grad()`.
- **Custom Weights Support**: The environment variable `DEEPFAKE_MODEL_PATH` can point to any PyTorch checkpoint (`.pth` or `.pt`).

---

## 5. Grad-CAM Explainability Methodology

Grad-CAM (Gradient-weighted Class Activation Mapping) produces visual explanations of convolutional network predictions:
1. Target convolutional layer: `layer4` (the final residual convolutional block before global average pooling).
2. Forward pass computes logits; the score $Y^c$ for class $c$ (e.g. `deepfake`) is extracted.
3. Backward pass propagates gradients $\frac{\partial Y^c}{\partial A^k_{i,j}}$ to feature map activations $A^k$.
4. Channel importance weights are calculated via Global Average Pooling:
   $$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial Y^c}{\partial A^k_{i,j}}$$
5. Positive contributions are aggregated via ReLU:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
6. Normalized activation map is resized to the original image dimensions using bicubic interpolation and colorized via the OpenCV JET colormap.
7. Both the raw heatmap and a 60/40 blended overlay with the original image are saved.

---

## 6. REST API Endpoints

### 1. Health Status
`GET /api/forensics/health`

Response:
```json
{
  "status": "ONLINE",
  "submodules": {
    "ela_analyzer": "Ready (In-memory JPEG Compression Differential)",
    "fft_analyzer": "Ready (2D Fast Fourier Spectral Decomposition)",
    "deepfake_model": "Online",
    "gradcam_engine": "Ready"
  },
  "torch_available": true
}
```

### 2. Forensic Analysis
`POST /api/forensics/analyze`

- **Content-Type**: `multipart/form-data`
- **Field**: `file` (JPG, PNG, or WEBP)

Response Structure:
```json
{
  "success": true,
  "scan_id": "4a71b2ef8901_1727145000",
  "filename": "sample_input.jpg",
  "original_image_url": "/outputs/forensic/4a71b2ef8901_1727145000_original.jpg",
  "prediction": {
    "label": "deepfake",
    "confidence": 0.9412,
    "class_probabilities": {
      "real": 0.0588,
      "deepfake": 0.9412
    },
    "model_architecture": "ResNet18-DeepfakeCNN"
  },
  "ela": {
    "available": true,
    "mean_error": 8.421,
    "max_error": 184.0,
    "std_error": 12.19,
    "high_error_ratio": 0.0612,
    "image_url": "/outputs/forensic/4a71b2ef8901_1727145000_ela.jpg",
    "quality": 90,
    "description": "..."
  },
  "fft": {
    "available": true,
    "mean_magnitude": 5.124,
    "max_magnitude": 14.88,
    "std_magnitude": 1.95,
    "high_frequency_ratio": 0.284,
    "image_url": "/outputs/forensic/4a71b2ef8901_1727145000_fft.jpg",
    "description": "..."
  },
  "gradcam": {
    "available": true,
    "target_class": "deepfake",
    "target_layer": "layer4",
    "heatmap_url": "/outputs/forensic/4a71b2ef8901_1727145000_gradcam.jpg",
    "overlay_url": "/outputs/forensic/4a71b2ef8901_1727145000_gradcam_overlay.jpg"
  },
  "warnings": [],
  "processing_time_ms": 142.5,
  "timestamp": "2026-09-24T02:30:00.000000"
}
```

HTTP Status Codes:
- `200 OK`: Successful analysis.
- `400 Bad Request`: Empty file, corrupted bytes, or unsupported file extension.
- `413 Request Entity Too Large`: File exceeds 15 MB upload ceiling.
- `500 Internal Server Error`: Processing pipeline failure.

---

## 7. Configuration

| Setting | Environment Variable | Default | Purpose |
|---|---|---|---|
| Model Weights | `DEEPFAKE_MODEL_PATH` | `""` | Optional path to custom `.pth` checkpoint |
| Target Conv Layer | `GRADCAM_TARGET_LAYER` | `"layer4"` | Layer hook for Grad-CAM activations |
| Max Upload Size | Built-in | 15 MB | Defense against denial-of-service |
| ELA JPEG Quality | Built-in | 90 | Quality baseline for compression delta |

---

## 8. Running Automated Tests

Run the full pytest suite:
```powershell
python -m pytest tests/ -v
```

---

## 9. Limitations & Ethical Forensics Grounding

1. **ELA is NOT Proof of Deepfake**: ELA detects differences in JPEG compression histories. A high ELA error may stem from benign image editing, cropping, or saving an image across multiple editing applications.
2. **FFT is NOT Proof of Deepfake**: Periodic frequency spikes indicate regular grid patterns or upsampling, which can also appear in digital textures, woven fabrics, or halftoned print scans.
3. **Grad-CAM is an Explainer, NOT a Detector**: Grad-CAM explains where a model looked during inference. It cannot independently determine truth or falsehood.
4. **Primary vs Supporting Signals**: The neural network classifier provides the primary statistical inference score; ELA and FFT provide supporting evidentiary artifacts for human review.
