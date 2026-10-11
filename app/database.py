"""
MongoDB database configuration.

Provides:
- MongoDB client
- Database instance
- Collection access
- FastAPI database dependency
"""

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


# ===========================================================================
# MongoDB Configuration
# ===========================================================================

MONGODB_URL = os.getenv(
    "MONGODB_URL",
    "mongodb://localhost:27017",
)

MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "ct200",
)


# ===========================================================================
# MongoDB Client
# ===========================================================================

client = MongoClient(
    MONGODB_URL
)

db = client[
    MONGODB_DATABASE
]


# ===========================================================================
# Collections
# ===========================================================================

documents_collection = db["documents"]

requirements_collection = db["requirements"]

test_cases_collection = db["test_cases"]

validation_results_collection = db["validation_results"]

traceability_collection = db["traceability"]


# ===========================================================================
# Connection
# ===========================================================================

def create_tables():
    """
    MongoDB does not require table creation.

    Collections are created automatically when documents
    are inserted.
    """
    return db


def get_db():
    """
    FastAPI dependency.

    Returns the MongoDB database instance.
    """
    try:
        yield db
    finally:
        pass