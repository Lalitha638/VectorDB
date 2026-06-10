import numpy as np
from core.distance import METRICS

class KDNode:
    def __init__(self, id, vector, axis, left=None, right=None):
        self.id = id
        self.vector = vector
        self.axis = axis        # which dimension we split on
        self.left = left
        self.right = right

class KDTree:
    def __init__(self):
        self.root = None
        self.all_vectors = {}

    def insert(self, id: str, vector: np.ndarray):
        self.all_vectors[id] = vector
        self.root = self._build(list(self.all_vectors.items()), depth=0)

    def delete(self, id: str):
        self.all_vectors.pop(id, None)
        self.root = self._build(list(self.all_vectors.items()), depth=0)

    def _build(self, items, depth):
        if not items:
            return None
        dims = len(items[0][1])
        axis = depth % dims
        items.sort(key=lambda x: x[1][axis])
        mid = len(items) // 2
        return KDNode(
            id=items[mid][0],
            vector=items[mid][1],
            axis=axis,
            left=self._build(items[:mid], depth + 1),
            right=self._build(items[mid + 1:], depth + 1)
        )

    def search(self, query: np.ndarray, k: int = 5, metric: str = "cosine"):
        dist_fn = METRICS[metric]
        best = []   # list of (dist, id)

        def _search(node):
            if node is None:
                return
            d = dist_fn(query, node.vector)
            best.append((d, node.id))
            best.sort()
            if len(best) > k:
                best.pop()

            # decide which subtree to explore first
            diff = query[node.axis] - node.vector[node.axis]
            close, away = (node.left, node.right) if diff <= 0 else (node.right, node.left)
            _search(close)

            # only explore the far side if it could contain a closer point
            if len(best) < k or abs(diff) < best[-1][0]:
                _search(away)

        _search(self.root)
        return [(id, dist) for dist, id in best]