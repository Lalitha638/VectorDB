import json
import os
import numpy as np

STORAGE_FILE = "vectordb_storage.json"

def save(metadata: dict, vectors: dict):
    data = {
        "metadata": metadata,
        "vectors": {id: vec.tolist() for id, vec in vectors.items()}
    }
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load() -> tuple:
    if not os.path.exists(STORAGE_FILE):
        return {}, {}
    with open(STORAGE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    vectors = {id: np.array(vec) for id, vec in data.get("vectors", {}).items()}
    return metadata, vectors