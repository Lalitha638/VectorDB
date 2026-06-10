import numpy as np
import time
from algorithms.brute_force import BruteForce
from algorithms.kd_tree import KDTree
from algorithms.hnsw import HNSW

class VectorDB:
    def __init__(self, dims=None):
        self.bf = BruteForce()
        self.kd = KDTree()
        self.hnsw = HNSW(M=16, ef_construction=200)
        self.metadata = {}
        self.dims = dims
        self._vectors = {}  # store raw vectors for persistence

    def insert(self, id: str, vector: list, meta: dict = None):
        v = np.array(vector, dtype=float)
        if self.dims is None:
            self.dims = len(v)
        if len(v) != self.dims:
            raise ValueError(f"Expected {self.dims}D vector, got {len(v)}D")
        self.bf.insert(id, v)
        self.kd.insert(id, v)
        self.hnsw.insert(id, v)
        self.metadata[id] = meta or {}
        self._vectors[id] = v

    def delete(self, id: str):
        self.bf.delete(id)
        self.kd.delete(id)
        self.hnsw.delete(id)
        self.metadata.pop(id, None)
        self._vectors.pop(id, None)

    def search(self, query: list, k: int = 5,
               metric: str = "cosine", algo: str = "hnsw"):
        q = np.array(query, dtype=float)
        algo_map = {"hnsw": self.hnsw, "kdtree": self.kd, "bruteforce": self.bf}
        engine = algo_map.get(algo, self.hnsw)
        results = engine.search(q, k=k, metric=metric)
        return [
            {"id": id, "distance": round(dist, 6), "meta": self.metadata.get(id, {})}
            for id, dist in results
        ]

    def benchmark(self, query: list, k: int = 5, metric: str = "cosine"):
        q = np.array(query, dtype=float)
        results = {}
        for name, engine in [("bruteforce", self.bf), ("kdtree", self.kd), ("hnsw", self.hnsw)]:
            start = time.perf_counter()
            hits = engine.search(q, k=k, metric=metric)
            elapsed = (time.perf_counter() - start) * 1000
            results[name] = {
                "time_ms": round(elapsed, 4),
                "results": [{"id": id, "distance": round(d, 6)} for id, d in hits]
            }
        return results


class MultiVectorDB:
    def __init__(self):
        self.demo_db = VectorDB(dims=16)
        self.doc_db = VectorDB(dims=768)

    def save_docs(self):
        """Save doc_db to disk."""
        from core.storage import save
        save(self.doc_db.metadata, self.doc_db._vectors)
        print(f"Saved {len(self.doc_db.metadata)} vectors to disk.")

    def load_docs(self):
        """Load doc_db from disk and rebuild indexes."""
        from core.storage import load
        metadata, vectors = load()
        if not vectors:
            print("No saved data found.")
            return
        for id, vec in vectors.items():
            meta = metadata.get(id, {})
            self.doc_db.insert(id, vec.tolist(), meta)
        print(f"Loaded {len(vectors)} vectors from disk.")