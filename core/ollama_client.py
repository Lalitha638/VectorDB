import requests

OLLAMA_URL = "http://localhost:11434"

def embed(text: str) -> list[float]:
    """Convert any text into a 768D vector using nomic-embed-text."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")
    return response.json()["embedding"]

def generate(prompt: str) -> str:
    """Generate an answer using llama3.2."""
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.text}")
    return response.json()["response"]

def is_online() -> bool:
    """Check if Ollama is running."""
    try:
        r = requests.get(OLLAMA_URL, timeout=2)
        return r.status_code == 200
    except:
        return False