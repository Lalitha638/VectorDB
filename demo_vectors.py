import numpy as np

DEMO_VECTORS = [
    # Computer Science
    {"id": "binary_tree",      "label": "Binary Tree",      "category": "CS",     "vector": [0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "linked_list",      "label": "Linked List",      "category": "CS",     "vector": [0.8,0.9,0.6,0.7,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "hash_table",       "label": "Hash Table",       "category": "CS",     "vector": [0.7,0.6,0.9,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "neural_network",   "label": "Neural Network",   "category": "CS",     "vector": [0.6,0.7,0.8,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "graph_algorithm",  "label": "Graph Algorithm",  "category": "CS",     "vector": [0.85,0.75,0.65,0.55,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    # Math
    {"id": "calculus",         "label": "Calculus",         "category": "Math",   "vector": [0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "linear_algebra",   "label": "Linear Algebra",   "category": "Math",   "vector": [0.1,0.1,0.1,0.1,0.8,0.9,0.6,0.7,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "statistics",       "label": "Statistics",       "category": "Math",   "vector": [0.1,0.1,0.1,0.1,0.7,0.6,0.9,0.8,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "number_theory",    "label": "Number Theory",    "category": "Math",   "vector": [0.1,0.1,0.1,0.1,0.6,0.7,0.8,0.9,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    {"id": "topology",         "label": "Topology",         "category": "Math",   "vector": [0.1,0.1,0.1,0.1,0.85,0.75,0.65,0.55,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]},
    # Food
    {"id": "sushi",            "label": "Sushi",            "category": "Food",   "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6,0.1,0.1,0.1,0.1]},
    {"id": "pasta",            "label": "Pasta",            "category": "Food",   "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.9,0.6,0.7,0.1,0.1,0.1,0.1]},
    {"id": "tacos",            "label": "Tacos",            "category": "Food",   "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.7,0.6,0.9,0.8,0.1,0.1,0.1,0.1]},
    {"id": "curry",            "label": "Curry",            "category": "Food",   "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.6,0.7,0.8,0.9,0.1,0.1,0.1,0.1]},
    {"id": "biryani",          "label": "Biryani",          "category": "Food",   "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.75,0.65,0.85,0.95,0.1,0.1,0.1,0.1]},
    # Sports
    {"id": "basketball",       "label": "Basketball",       "category": "Sports", "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.9,0.8,0.7,0.6]},
    {"id": "football",         "label": "Football",         "category": "Sports", "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.8,0.9,0.6,0.7]},
    {"id": "cricket",          "label": "Cricket",          "category": "Sports", "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.7,0.6,0.9,0.8]},
    {"id": "tennis",           "label": "Tennis",           "category": "Sports", "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.6,0.7,0.8,0.9]},
    {"id": "kabaddi",          "label": "Kabaddi",          "category": "Sports", "vector": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.85,0.75,0.65,0.55]},
]

def get_demo_vectors():
    return [
        {**item, "vector": np.array(item["vector"], dtype=float)}
        for item in DEMO_VECTORS
    ]