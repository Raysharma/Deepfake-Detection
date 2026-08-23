# 🛡️ AI-Powered Multi-Modal Deepfake Detection System

An enterprise-grade multi-modal deepfake detection and explainability system supporting **Image**, **Video**, and **Audio/Voice** inputs with forensic artifact analysis, Grad-CAM visual heatmaps, audio-lip sync analysis, and an active learning feedback loop.

---

## 🌟 Key Features

- **Multi-Modal Detection Pipeline**:
  - 🖼️ **Image Forensics**: Error Level Analysis (ELA), 2D Fourier (FFT) Frequency Analysis, and Grad-CAM visual heatmap generator.
  - 🎬 **Video Forensics**: Frame sampling, inter-frame optical jitter variance, temporal flicker detection, and audio-to-lip synchronization scoring.
  - 🎙️ **Audio / Voice Forensics**: Zero-Crossing Rate (ZCR), Spectral Flux, and MFCC anomaly inspection for cloned speech detection.
- **Explainability Layer**: Visual Grad-CAM heatmaps highlighting suspicious facial/boundary regions.
- **Relational Audit Logging & Feedback**: SQLite database storing full scan metrics, inference latencies, and human reviewer feedback for model retraining.
- **RESTful API**: Fast, asynchronous endpoints built with FastAPI and strictly validated with Pydantic v2.
- **Interactive Dashboard**: Modern glassmorphic dark-mode web UI with real-time confidence metrics and preset demo modes.

---

## 🏗️ Architecture & Modules

```
DeepFake Detection System/
├── backend/
│   ├── app.py                     # FastAPI application & endpoint routing
│   ├── database.py                # SQLite database and audit feedback layer
│   ├── schemas.py                 # Pydantic data contracts for all 3 modalities
│   └── pipeline/
│       ├── image_pipeline.py      # ELA, 2D FFT, and Grad-CAM generator
│       ├── video_pipeline.py      # Temporal flicker & optical jitter analyzer
│       └── audio_pipeline.py      # Spectral flux & voice clone detector
├── static/
│   ├── index.html                 # Modern forensic dashboard
│   ├── css/styles.css             # Glassmorphic dark-mode styling
│   └── js/app.js                  # Async client-side application logic
├── requirements.txt               # Dependencies list
├── run_server.py                  # Entrypoint launcher script
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Raysharma/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Create and activate virtual environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
python run_server.py
```

### 5. Access the application
- 🌐 **Web Dashboard**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📖 **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 👥 Team Responsibilities

- **Module 1**: Core Pipeline, Database & API Architecture (FastAPI + SQLite)
- **Module 2**: Image Detection & Grad-CAM Explainability (OpenCV, ELA, 2D FFT)
- **Module 3**: Video Temporal & Audio-Lip Sync Analysis
- **Module 4**: Audio Spectral Forensics & Voice Cloned Speech Detection
