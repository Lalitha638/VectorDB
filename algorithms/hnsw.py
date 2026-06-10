import numpy as np
import random
import heapq
from core.distance import METRICS

class HNSWNode:
    def __init__(self, id: str, vector: np.ndarray, max_layer: int):
        self.id = id
        self.vector = vector
        self.max_layer = max_layer
        # neighbors[layer] = list of neighbor node ids
        self.neighbors = [[] for _ in range(max_layer + 1)]

class HNSW:
    def __init__(self, M=16, ef_construction=200, mL=None):
        """
        M               : max neighbors per node per layer
        ef_construction : beam width during insert (higher = more accurate, slower build)
        mL              : level multiplier (controls layer height probability)
        """
        self.M = M
        self.ef_construction = ef_construction
        self.mL = mL or (1 / np.log(M))
        self.nodes = {}         # id -> HNSWNode
        self.entry_point = None
        self.max_layer = 0

    # ── helpers ──────────────────────────────────────────────────────────

    def _random_layer(self) -> int:
        """Randomly assign a max layer to a new node (exponential distribution)."""
        return int(-np.log(random.random()) * self.mL)

    def _distance(self, a: HNSWNode, b: HNSWNode, dist_fn) -> float:
        return dist_fn(a.vector, b.vector)

    def _search_layer(self, query_vec: np.ndarray, entry_ids: list,
                      ef: int, layer: int, dist_fn) -> list:
        """
        Beam search within a single layer.
        Returns ef closest node ids found.
        """
        visited = set(entry_ids)
        # min-heap of (dist, id) for candidates to expand
        candidates = []
        # max-heap of (dist, id) for current best results (negate dist for max-heap)
        results = []

        for eid in entry_ids:
            node = self.nodes[eid]
            d = dist_fn(query_vec, node.vector)
            heapq.heappush(candidates, (d, eid))
            heapq.heappush(results, (-d, eid))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)  # closest candidate
            worst_result_dist = -results[0][0]         # farthest in results

            # if closest candidate is farther than worst result, stop
            if c_dist > worst_result_dist and len(results) >= ef:
                break

            for neighbor_id in self.nodes[c_id].neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor = self.nodes[neighbor_id]
                    d = dist_fn(query_vec, neighbor.vector)
                    heapq.heappush(candidates, (d, neighbor_id))
                    heapq.heappush(results, (-d, neighbor_id))
                    if len(results) > ef:
                        heapq.heappop(results)  # remove farthest

        return [id for _, id in results]

    def _select_neighbors(self, node_id: str, candidates: list,
                          M: int, layer: int, dist_fn) -> list:
        """Pick the M closest candidates to connect to."""
        node = self.nodes[node_id]
        scored = [(dist_fn(node.vector, self.nodes[c].vector), c) for c in candidates]
        scored.sort()
        return [id for _, id in scored[:M]]

    # ── public API ───────────────────────────────────────────────────────

    def insert(self, id: str, vector: np.ndarray, metric: str = "cosine"):
        dist_fn = METRICS[metric]
        node_layer = self._random_layer()
        new_node = HNSWNode(id, np.array(vector, dtype=float), node_layer)
        self.nodes[id] = new_node

        if self.entry_point is None:
            self.entry_point = id
            self.max_layer = node_layer
            return

        entry_ids = [self.entry_point]

        # Phase 1: greedily descend from top layer to node_layer+1 (ef=1, just navigating)
        for layer in range(self.max_layer, node_layer, -1):
            entry_ids = self._search_layer(vector, entry_ids, ef=1, layer=layer, dist_fn=dist_fn)

        # Phase 2: from node_layer down to 0, do full beam search and connect
        for layer in range(min(node_layer, self.max_layer), -1, -1):
            neighbors_found = self._search_layer(
                vector, entry_ids, ef=self.ef_construction, layer=layer, dist_fn=dist_fn
            )
            chosen = self._select_neighbors(id, neighbors_found, self.M, layer, dist_fn)

            new_node.neighbors[layer] = chosen

            # connect back (bidirectional edges)
            for neighbor_id in chosen:
                neighbor = self.nodes[neighbor_id]
                if layer < len(neighbor.neighbors):
                    neighbor.neighbors[layer].append(id)
                    # prune if neighbor has too many connections
                    if len(neighbor.neighbors[layer]) > self.M * 2:
                        neighbor.neighbors[layer] = self._select_neighbors(
                            neighbor_id, neighbor.neighbors[layer],
                            self.M, layer, dist_fn
                        )

            entry_ids = neighbors_found

        if node_layer > self.max_layer:
            self.max_layer = node_layer
            self.entry_point = id

    def search(self, query: np.ndarray, k: int = 5,
               ef: int = 50, metric: str = "cosine") -> list:
        if not self.entry_point:
            return []

        dist_fn = METRICS[metric]
        entry_ids = [self.entry_point]

        # descend from top layer to layer 1 (greedy, ef=1)
        for layer in range(self.max_layer, 0, -1):
            entry_ids = self._search_layer(query, entry_ids, ef=1, layer=layer, dist_fn=dist_fn)

        # full beam search at layer 0
        candidates = self._search_layer(query, entry_ids, ef=max(ef, k), layer=0, dist_fn=dist_fn)

        # score and return top k
        scored = [(dist_fn(query, self.nodes[c].vector), c) for c in candidates]
        scored.sort()
        return [(id, dist) for dist, id in scored[:k]]

    def delete(self, id: str):
        """Remove a node (rebuilds affected neighbor lists)."""
        if id not in self.nodes:
            return
        node = self.nodes.pop(id)
        for layer, neighbors in enumerate(node.neighbors):
            for nb_id in neighbors:
                if nb_id in self.nodes and layer < len(self.nodes[nb_id].neighbors):
                    self.nodes[nb_id].neighbors[layer] = [
                        n for n in self.nodes[nb_id].neighbors[layer] if n != id
                    ]
        if self.entry_point == id:
            self.entry_point = next(iter(self.nodes), None)

    def get_info(self) -> dict:
        layer_counts = {}
        for node in self.nodes.values():
            layer_counts[node.max_layer] = layer_counts.get(node.max_layer, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "max_layer": self.max_layer,
            "entry_point": self.entry_point,
            "layer_distribution": layer_counts,
            "M": self.M,
            "ef_construction": self.ef_construction
        }