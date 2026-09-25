from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from .config import MONGODB_URI, DATABASE_NAME, COLLECTION_NAME


# MongoDB client
client = None
db = None
analysis_collection = None


def connect_to_mongodb():
    """Create MongoDB connection."""
    global client, db, analysis_collection

    if not MONGODB_URI:
        raise ValueError("MONGODB_URI is not configured.")

    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000
    )

    # Check connection
    client.admin.command("ping")

    db = client[DATABASE_NAME]
    analysis_collection = db[COLLECTION_NAME]

    return db


def get_database():
    """Return the MongoDB database."""
    if db is None:
        connect_to_mongodb()

    return db


def get_analysis_collection():
    """Return the analysis results collection."""
    if analysis_collection is None:
        connect_to_mongodb()

    return analysis_collection


def close_mongodb():
    """Close MongoDB connection."""
    global client, db, analysis_collection

    if client is not None:
        client.close()

    client = None
    db = None
    analysis_collection = None