import numpy as np
from core.distance import METRICS

class BruteForce:
    def __init__(self):
        self.vectors = {}   # id -> np.ndarray

    def insert(self, id: str, vector: np.ndarray):
        self.vectors[id] = vector

    def delete(self, id: str):
        self.vectors.pop(id, None)

    def search(self, query: np.ndarray, k: int = 5, metric: str = "cosine"):
        dist_fn = METRICS[metric]
        results = [
            (dist_fn(query, vec), id)
            for id, vec in self.vectors.items()
        ]
        results.sort()
        return [(id, dist) for dist, id in results[:k]]