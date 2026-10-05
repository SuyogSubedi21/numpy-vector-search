import json

import numpy as np


def normalize(vectors):
    """Make every vector length 1, so cosine similarity becomes a simple dot product."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def search(query_vector, vectors, texts, top_k=3):
    scores = vectors @ normalize(query_vector)    # cosine similarity with every chunk at once
    best = np.argsort(scores)[::-1][:top_k]       # indexes of the highest scores
    return [(float(scores[i]), texts[i]) for i in best]


def save_index(path, vectors, texts):
    np.save(f"{path}.npy", vectors)
    with open(f"{path}.json", "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)


def load_index(path):
    vectors = np.load(f"{path}.npy")
    with open(f"{path}.json", encoding="utf-8") as f:
        texts = json.load(f)
    return vectors, texts