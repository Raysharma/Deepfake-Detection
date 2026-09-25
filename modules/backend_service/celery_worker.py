from celery import Celery

from .config import REDIS_URL
from .orchestrator import run_full_pipeline
from .mongo_client import get_analysis_collection


celery_app = Celery(
    "deepfake_backend",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
)


@celery_app.task
def health_check_task():
    return {
        "status": "healthy",
        "message": "Celery worker is running."
    }

@celery_app.task
def run_deepfake_analysis(video_path: str):
    result = run_full_pipeline(video_path)

    collection = get_analysis_collection()
    collection.insert_one(result)

    return result