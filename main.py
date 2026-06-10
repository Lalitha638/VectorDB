import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from core.vector_db import VectorDB, MultiVectorDB
from core.ollama_client import embed, generate, is_online
from demo_vectors import get_demo_vectors

app = FastAPI(title="VectorDB", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

multi = MultiVectorDB()

# Load saved documents from disk on startup
multi.load_docs()

for item in get_demo_vectors():
    multi.demo_db.insert(item["id"], item["vector"].tolist(), meta={"label": item["label"], "category": item["category"]})

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

class AskRequest(BaseModel):
    question: str
    k: int = 3

@app.get("/status")
def status():
    return {"ollama": "online" if is_online() else "offline", "embed_model": "nomic-embed-text", "vector_dims": 768}

@app.get("/search")
def search(v: str, k: int = 5, metric: str = "cosine", algo: str = "hnsw"):
    try:
        query = [float(x) for x in v.split(",")]
    except ValueError:
        raise HTTPException(400, "Invalid vector format.")
    return multi.demo_db.search(query, k=k, metric=metric, algo=algo)

@app.post("/insert")
def insert(req: InsertRequest):
    multi.demo_db.insert(req.id, req.vector, req.meta)
    return {"status": "ok", "id": req.id}

@app.delete("/delete/{id}")
def delete(id: str):
    multi.demo_db.delete(id)
    return {"status": "ok", "id": id}

@app.get("/benchmark")
def benchmark(v: str, k: int = 5, metric: str = "cosine"):
    query = [float(x) for x in v.split(",")]
    return multi.demo_db.benchmark(query, k=k, metric=metric)

@app.get("/hnsw-info")
def hnsw_info():
    return multi.demo_db.hnsw.get_info()

@app.get("/items")
def items():
    return [{"id": id, **multi.demo_db.metadata[id]} for id in multi.demo_db.metadata]

@app.get("/stats")
def stats():
    return {
        "demo_vectors": len(multi.demo_db.metadata),
        "doc_vectors": len(multi.doc_db.metadata),
        "algorithms": ["hnsw", "kdtree", "bruteforce"],
        "metrics": ["cosine", "euclidean", "manhattan"]
    }

@app.post("/embed-insert")
def embed_insert(req: EmbedRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")
    vector = embed(req.text)
    multi.doc_db.insert(req.id, vector, meta={"text": req.text, **req.meta})
    multi.save_docs()
    return {"status": "ok", "id": req.id, "dims": len(vector)}

@app.post("/embed-search")
def embed_search(req: SearchTextRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")
    vector = embed(req.text)
    results = multi.doc_db.search(vector, k=req.k, metric="cosine", algo="hnsw")
    return results

@app.post("/doc/insert")
def doc_insert(req: EmbedRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")

    words = req.text.split()
    chunk_size = 100
    overlap = 20
    chunks = []

    if len(words) <= chunk_size:
        chunks = [req.text]
    else:
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap

    inserted = []
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{req.id}_chunk{idx}"
        vector = embed(chunk)
        multi.doc_db.insert(chunk_id, vector, meta={
            "text": chunk,
            "source": req.id,
            "chunk": idx
        })
        inserted.append(chunk_id)

    # Save to disk after every insert
    multi.save_docs()

    return {
        "status": "ok",
        "source": req.id,
        "chunks_inserted": len(inserted),
        "chunk_ids": inserted
    }

@app.get("/doc/list")
def doc_list():
    docs = {}
    for id, meta in multi.doc_db.metadata.items():
        source = meta.get("source", id)
        if source not in docs:
            docs[source] = {"chunks": 0, "ids": []}
        docs[source]["chunks"] += 1
        docs[source]["ids"].append(id)
    return docs

@app.post("/doc/ask")
def doc_ask(req: AskRequest):
    if not is_online():
        raise HTTPException(503, "Ollama is offline. Run: ollama serve")

    if len(multi.doc_db.metadata) == 0:
        raise HTTPException(400, "No documents inserted yet. Use /doc/insert first.")

    q_vector = embed(req.question)
    results = multi.doc_db.search(q_vector, k=req.k, metric="cosine", algo="hnsw")

    context = "\n\n".join([
        f"[Chunk {i+1}]: {r['meta']['text']}"
        for i, r in enumerate(results)
    ])

    prompt = f"""You are a helpful study assistant. Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I don't have enough information in my notes to answer this."

Context from notes:
{context}

Question: {req.question}

Answer:"""

    answer = generate(prompt)

    return {
        "question": req.question,
        "answer": answer,
        "sources_used": [r["id"] for r in results],
        "context_chunks": [r["meta"]["text"] for r in results]
    }

@app.get("/")
def root():
    return FileResponse("index.html")