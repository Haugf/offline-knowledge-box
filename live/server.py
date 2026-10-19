"""Field Kit live v1: the Offline Knowledge Box running natively on Windows (no Docker, no Ollama).

Retrieves from the knowledge pack built by ../app/ingest.py and answers with a local
Hugging Face model through transformers, on the GPU when there is one.

Run:  .\\run.ps1     (or: python server.py)   then open http://localhost:8080
"""
import gc
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

HERE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", HERE / "data"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "4"))
OVERLAP_WORDS = 30  # must match ingest.py, used to stitch chunks back into articles

MODELS = {
    "smollm2-135m": {"repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "label": "SmolLM2 135M", "open": "Fully open"},
    "smollm2-1.7b": {"repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "label": "SmolLM2 1.7B", "open": "Fully open"},
    "llama3.1-8b": {"repo": "meta-llama/Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B", "open": "Open weights",
                    "quantize": True},
}
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "smollm2-1.7b")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are Field Kit, an offline reference for the outdoors. Answer using only the numbered "
    "sources provided. Keep it short and practical, and cite sources like [1]. If the sources "
    "do not contain the answer, say so plainly. For medical or emergency questions, give general "
    "information and advise contacting emergency services when one can be reached."
)

app = FastAPI(title="Field Kit live")
pack = {"vectors": None, "chunks": [], "embedder": None, "meta": {}, "titles": {}}
llm = {"key": None, "model": None, "tokenizer": None}
gen_lock = threading.Lock()


@app.on_event("startup")
def load_pack():
    vec_path = DATA_DIR / "vectors.npy"
    if not vec_path.exists():
        print("No knowledge pack yet. Run: .\\run.ps1 -Ingest")
        return
    from sentence_transformers import SentenceTransformer

    pack["vectors"] = np.load(vec_path)
    with open(DATA_DIR / "chunks.jsonl", encoding="utf-8") as f:
        pack["chunks"] = [json.loads(line) for line in f]
    meta_file = DATA_DIR / "pack.json"
    pack["meta"] = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    for i, c in enumerate(pack["chunks"]):
        pack["titles"].setdefault(c["title"], []).append(i)
    pack["embedder"] = SentenceTransformer(pack["meta"].get("embed_model", EMBED_MODEL), device=DEVICE)
    print(f"Loaded {len(pack['chunks'])} chunks from {len(pack['titles'])} articles")


def retrieve(question: str, k: int):
    if pack["vectors"] is None:
        return []
    q = pack["embedder"].encode([question], normalize_embeddings=True)[0]
    scores = pack["vectors"] @ q
    top = np.argsort(-scores)[:k]
    return [{**pack["chunks"][i], "score": float(scores[i])} for i in top]


def load_llm(key: str):
    if llm["key"] == key:
        return llm["model"], llm["tokenizer"]
    spec = MODELS[key]
    # One model on the GPU at a time so the 8B model has the whole card.
    llm.update(key=None, model=None, tokenizer=None)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
    if spec.get("quantize") and DEVICE == "cuda":
        # 8B in fp16 is ~16 GB; 4-bit is ~6 GB, which fits an 8 GB card.
        model = AutoModelForCausalLM.from_pretrained(
            spec["repo"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float16),
            device_map="auto",
        )
    else:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(spec["repo"], dtype=dtype).to(DEVICE)
    llm.update(key=key, model=model, tokenizer=tokenizer)
    return model, tokenizer


class Ask(BaseModel):
    question: str
    model: str | None = None
    top_k: int | None = None


@app.post("/api/ask")
def ask(body: Ask):
    key = body.model or DEFAULT_MODEL
    if key not in MODELS:
        raise HTTPException(400, f"Unknown model {key}")
    sources = retrieve(body.question, body.top_k or TOP_K)
    context = "\n\n".join(f"[{n}] {s['title']}: {s['text']}" for n, s in enumerate(sources, 1))
    prompt = f"Sources:\n{context or '(no knowledge pack loaded)'}\n\nQuestion: {body.question}"

    def stream():
        yield json.dumps({"sources": [{"n": n, "title": s["title"], "url": s["url"], "score": round(s["score"], 3)}
                                      for n, s in enumerate(sources, 1)]}) + "\n"
        with gen_lock:
            try:
                if llm["key"] != key:
                    yield json.dumps({"status": f"Loading {MODELS[key]['label']}…"}) + "\n"
                model, tokenizer = load_llm(key)
            except Exception as e:  # gated model without login, out of memory, no download, ...
                yield json.dumps({"error": f"Could not load {MODELS[key]['label']}: {e}"}) + "\n"
                return
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                   return_tensors="pt", return_dict=True).to(model.device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            thread = threading.Thread(target=model.generate, daemon=True, kwargs=dict(
                **inputs, streamer=streamer, max_new_tokens=400, do_sample=True, temperature=0.4, top_p=0.9,
                repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id))
            start = time.time()
            thread.start()
            text = ""
            for delta in streamer:
                text += delta
                yield json.dumps({"delta": delta}) + "\n"
            thread.join()
            n_tokens = len(tokenizer(text)["input_ids"])
            elapsed = max(time.time() - start, 1e-6)
            yield json.dumps({"done": True, "model": MODELS[key]["label"],
                              "tokens_per_sec": round(n_tokens / elapsed, 1)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/search")
def search(q: str, k: int = 12):
    """Library search: best-matching articles, one row per title."""
    hits, seen = [], set()
    for s in retrieve(q, k * 4):
        if s["title"] in seen:
            continue
        seen.add(s["title"])
        hits.append({"title": s["title"], "snippet": " ".join(s["text"].split()[:40]) + "…",
                     "score": round(s["score"], 3)})
        if len(hits) == k:
            break
    return {"results": hits}


@app.get("/api/article")
def article(title: str):
    idx = pack["titles"].get(title)
    if not idx:
        raise HTTPException(404, "Article not in the knowledge pack")
    parts = []
    for n, i in enumerate(idx):
        words = pack["chunks"][i]["text"].split()
        parts.extend(words if n == 0 else words[OVERLAP_WORDS:])
    return {"title": title, "url": pack["chunks"][idx[0]]["url"], "text": " ".join(parts)}


@app.get("/api/status")
def status():
    gpu = torch.cuda.get_device_name(0) if DEVICE == "cuda" else None
    return {
        "pack": pack["meta"], "chunks": len(pack["chunks"]), "articles": len(pack["titles"]),
        "device": DEVICE, "gpu": gpu, "loaded_model": llm["key"], "default_model": DEFAULT_MODEL,
        "models": [{"key": k, "label": v["label"], "open": v["open"]} for k, v in MODELS.items()],
    }


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8080")))
