"""Field Kit: an offline knowledge box running natively on Windows (no Docker, no Ollama).

Retrieves from the knowledge pack built by ingest.py and answers with a local
Hugging Face model through transformers, on the GPU when there is one.

Run:  .\\run.ps1     (or: python server.py)   then open http://localhost:8080
"""
import base64
import gc
import io
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
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer,
                          BitsAndBytesConfig, TextIteratorStreamer)

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
VISION_MODEL = os.getenv("VISION_MODEL", "HuggingFaceTB/SmolVLM2-2.2B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SURVEY_PROMPT = (
    "Identify the main plant, animal, insect or fungus in this photo. Answer in exactly this format:\n"
    "NAME: <common name>\nSCIENTIFIC: <scientific name, or unknown>\n"
    "FEATURES: <one sentence on the visible features you used>"
)

SYSTEM_PROMPT = (
    "You are Field Kit, an offline reference for the outdoors. Answer using only the numbered "
    "sources provided. Keep it short and practical, and cite sources like [1]. If the sources "
    "do not contain the answer, say so plainly. For medical or emergency questions, give general "
    "information and advise contacting emergency services when one can be reached."
)

app = FastAPI(title="Field Kit live")
pack = {"vectors": None, "chunks": [], "embedder": None, "meta": {}, "titles": {}, "lower_titles": {}}
# One model on the GPU at a time (chat or vision), so an 8 GB card never runs out of memory.
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
    pack["lower_titles"] = {t.lower(): t for t in pack["titles"]}
    pack["embedder"] = SentenceTransformer(pack["meta"].get("embed_model", EMBED_MODEL), device=DEVICE)
    print(f"Loaded {len(pack['chunks'])} chunks from {len(pack['titles'])} articles")


def retrieve(question: str, k: int):
    if pack["vectors"] is None:
        return []
    q = pack["embedder"].encode([question], normalize_embeddings=True)[0]
    scores = pack["vectors"] @ q
    top = np.argsort(-scores)[:k]
    return [{**pack["chunks"][i], "score": float(scores[i])} for i in top]


def free_gpu():
    llm.update(key=None, model=None, tokenizer=None)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def load_llm(key: str):
    if llm["key"] == key:
        return llm["model"], llm["tokenizer"]
    spec = MODELS[key]
    free_gpu()
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


def load_vision():
    if llm["key"] == "vision":
        return llm["model"], llm["tokenizer"]
    free_gpu()
    processor = AutoProcessor.from_pretrained(VISION_MODEL)
    dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(VISION_MODEL, dtype=dtype).to(DEVICE)
    llm.update(key="vision", model=model, tokenizer=processor)
    return model, processor


def find_articles(q: str, k: int):
    """Best-matching articles for a query, one row per title."""
    hits, seen = [], set()
    for s in retrieve(q, k * 4):
        if s["title"] in seen:
            continue
        seen.add(s["title"])
        hits.append({"title": s["title"], "snippet": " ".join(s["text"].split()[:40]) + "…",
                     "score": round(s["score"], 3)})
        if len(hits) == k:
            break
    return hits


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
    return {"results": find_articles(q, k)}


class Survey(BaseModel):
    image: str  # data URL or bare base64, already downscaled by the browser


@app.post("/api/survey")
def survey(body: Survey):
    """Identify what's in a photo with a small vision model, then look it up in the offline library."""
    try:
        raw = base64.b64decode(body.image.split(",", 1)[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "That doesn't look like an image")
    img.thumbnail((1024, 1024))

    with gen_lock:
        start = time.time()
        try:
            model, processor = load_vision()
        except Exception as e:
            raise HTTPException(500, f"Could not load the vision model: {e}")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": SURVEY_PROMPT}]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[img], return_tensors="pt").to(model.device, dtype=model.dtype)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        text = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    fields = {}
    for line in text.splitlines():
        label, _, value = line.partition(":")
        if value.strip():
            fields[label.strip().upper()] = value.strip().rstrip(".")
    name = fields.get("NAME") or text.split("\n")[0][:80]
    scientific = fields.get("SCIENTIFIC", "")
    if scientific.lower() in ("unknown", "n/a", "none"):
        scientific = ""
    features = fields.get("FEATURES", "")

    matches = find_articles(f"{name} {scientific} {features}", 4)
    # An article whose title is exactly the guessed name goes first.
    for guess in (scientific, name, name.rstrip("s")):
        title = pack["lower_titles"].get(guess.lower()) if guess else None
        if title:
            matches = [m for m in matches if m["title"] != title]
            matches.insert(0, {"title": title, "snippet": "", "score": 1.0, "exact": True})
            break
    return {"name": name, "scientific": scientific, "features": features, "raw": text,
            "matches": matches[:4], "seconds": round(time.time() - start, 1)}


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
