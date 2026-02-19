import joblib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

def save_object(obj: Any, file_path: str) -> None:
    """Saves a Python object (model, encoder) to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)
    logger.info(f"Object saved to {file_path}")

def load_object(file_path: str) -> Any:
    """Loads a Python object."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info(f"Object loaded from {file_path}")
    return joblib.load(file_path)