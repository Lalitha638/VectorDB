content = """import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from core.vector_db import VectorDB
from core.ollama_client import embed, is_online
from demo_vectors import get_demo_vectors

app = FastAPI(title="VectorDB", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db = VectorDB()

for item in get_demo_vectors():
    db.insert(item["id"], item["vector"].tolist(), meta={"label": item["label"], "category": item["category"]})

class InsertRequest(BaseModel):
    id: str
    vector: list[float]
    meta: dict = {}

class EmbedRequest(BaseModel):
    id: str
    text: str
    meta: dict = {}

class SearchTextRequest(BaseModel):
    text: str
    k: int = 5

@app.get("/")
def root():
    return {"status": "VectorDB is running", "docs": "http://localhost:8080/docs"}

@app.get("/status")
def status():
    return {"ollama": "online" if is_online() else "offline", "embed_model": "nomic-embed-text", "vector_dims": 768}

@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        query = [float(x) for x in v.split(",")]
    except ValueError:
        raise HTTPException(400, "Invalid vector format.")
    return db.search(query, k=k, metric=metric, algo=algo)

@app.post("/insert")
def insert(req: InsertRequest):
    db.insert(req.id, req.vector, req.meta)
    return {"status": "ok", "id": req.id}

@app.delete("/delete/{id}")
def delete(id: str):
    db.delete(id)
    return {"status": "ok", "id": id}

@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    query = [float(x) for x in v.split(",")]
    return db.benchmark(query, k=k, metric=metric)

@app.get("/hnsw-info")
def hnsw_info():
    return db.hnsw.get_info()

@app.get("/items")
def items():
    return [{"id": id, **db.metadata[id]} for id in db.metadata]

@app.get("/stats")
def stats():
    return {"total_vectors": len(db.metadata), "algorithms": ["hnsw", "kdtree", "bruteforce"], "metrics": ["cosine", "euclidean", "manhattan"]}

@app.post("/embed-insert")
def embed_insert(req: EmbedRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")
    vector = embed(req.text)
    db.insert(req.id, vector, meta={"text": req.text, **req.meta})
    return {"status": "ok", "id": req.id, "dims": len(vector)}

@app.post("/embed-search")
def embed_search(req: SearchTextRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")
    vector = embed(req.text)
    results = db.search(vector, k=req.k, metric="cosine", algo="hnsw")
    return results
"""

with open("main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done! main.py written successfully.")

