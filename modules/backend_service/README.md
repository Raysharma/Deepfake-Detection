# ⚙️ Person 4: Backend Infrastructure & Data Engineer

## 🎯 Your Responsibilities
1. Build the **FastAPI** application with upload endpoints (`/api/v1/upload`, `/api/v1/status/{task_id}`).
2. Configure **Celery + Redis** message broker to handle asynchronous background task processing so long video inference does not freeze the API.
3. Design and connect **MongoDB** collections to store:
   - Media metadata
   - Individual modality scores (Image, Audio, Sync)
   - Aggregated Multimodal Final Report
   - File paths to generated Grad-CAM heatmaps
4. Orchestrate the pipeline: import `predict()` from the 3 ML modules, execute them on the uploaded file path, compute the weighted final verdict, and persist the document to MongoDB.

## 📁 File Layout
```
backend_service/
├── README.md               # This file
├── app.py                  # FastAPI server entrypoint
├── celery_worker.py        # Celery task definitions
├── mongo_client.py         # MongoDB connection & PyMongo / Motor schemas
├── orchestrator.py         # Pipeline coordinator importing ML modules
└── config.py               # Redis & MongoDB connection URLs
```
