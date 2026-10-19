"""Offline Knowledge Box API: retrieve from the local knowledge pack, answer with a local open-weight model."""
import json
import os
from pathlib import Path

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = (
    "You are an offline knowledge box. Answer using only the numbered sources provided. "
    "Cite sources like [1]. If the sources do not contain the answer, say so plainly. "
    "For medical or emergency questions, give general information and advise contacting "
    "emergency services or a professional when one can be reached."
)

app = FastAPI(title="Offline Knowledge Box")
state = {"vectors": None, "chunks": [], "embedder": None, "pack": {}}


@app.on_event("startup")
def load_pack():
    vec_path = DATA_DIR / "vectors.npy"
    if not vec_path.exists():
        print("No knowledge pack yet. Run: docker compose run --rm app python ingest.py")
        return
    from sentence_transformers import SentenceTransformer

    state["vectors"] = np.load(vec_path)
    with open(DATA_DIR / "chunks.jsonl", encoding="utf-8") as f:
        state["chunks"] = [json.loads(line) for line in f]
    pack_file = DATA_DIR / "pack.json"
    state["pack"] = json.loads(pack_file.read_text()) if pack_file.exists() else {}
    state["embedder"] = SentenceTransformer(state["pack"].get("embed_model", EMBED_MODEL))
    print(f"Loaded {len(state['chunks'])} chunks")


def retrieve(question: str, k: int):
    if state["vectors"] is None:
        return []
    q = state["embedder"].encode([question], normalize_embeddings=True)[0]
    scores = state["vectors"] @ q
    top = np.argsort(-scores)[:k]
    return [{**state["chunks"][i], "score": float(scores[i])} for i in top]


class Ask(BaseModel):
    question: str
    model: str | None = None
    top_k: int | None = None


@app.post("/api/ask")
async def ask(body: Ask):
    sources = retrieve(body.question, body.top_k or TOP_K)
    context = "\n\n".join(f"[{n}] {s['title']}: {s['text']}" for n, s in enumerate(sources, 1))
    prompt = f"Sources:\n{context or '(no knowledge pack loaded)'}\n\nQuestion: {body.question}"
    payload = {
        "model": body.model or CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Model call failed: {e}. Is the model pulled? See scripts/pull-models.sh")

    eval_count = data.get("eval_count", 0)
    eval_ns = data.get("eval_duration", 0) or 1
    return {
        "answer": data["message"]["content"],
        "model": payload["model"],
        "tokens_per_sec": round(eval_count / (eval_ns / 1e9), 1),
        "sources": [{"n": n, "title": s["title"], "url": s["url"], "score": round(s["score"], 3)}
                    for n, s in enumerate(sources, 1)],
    }


@app.get("/api/status")
async def status():
    models = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
    except httpx.HTTPError:
        pass
    return {"pack": state["pack"], "chunks": len(state["chunks"]),
            "default_model": CHAT_MODEL, "models": models}


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
