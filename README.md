# VectorDB 🧠

A Vector Database built from scratch in Python with a Web UI and RAG pipeline.

## Features
- 3 search algorithms — HNSW, KD-Tree, Brute Force
- Real text embeddings via Ollama (768D)
- Ask AI — RAG pipeline using llama3.2
- Colorful Web UI
- Persistent storage

## Tech Stack
- Python, FastAPI, Ollama, HTML/CSS/JS

## Setup
```bash
pip install fastapi uvicorn numpy requests aiofiles
ollama pull nomic-embed-text
ollama pull llama3.2
uvicorn main:app --reload --port 8080
```

Open http://localhost:8080

## Algorithms
| Algorithm | Complexity | Type |
|-----------|-----------|------|
| HNSW | O(log N) | Approximate |
| KD-Tree | O(log N) | Exact |
| Brute Force | O(N) | Exact |

<img width="860" height="511" alt="VectorDB1" src="https://github.com/user-attachments/assets/02ee415c-64a2-4604-9282-5d9a60ba6270" />

<img width="866" height="512" alt="ectorDB2" src="https://github.com/user-attachments/assets/1204b433-7f0d-419b-b96d-da3afd2f04d0" />
<img width="866" height="511" alt="VectorDB3PNG" src="https://github.com/user-attachments/assets/37cea279-c5ba-4728-a76a-fc1555ad0db5" />
