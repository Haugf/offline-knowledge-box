"""Build the offline knowledge pack from a Hugging Face dataset.

Run once while online:
    docker compose run --rm app python ingest.py

Streams articles from the dataset, splits them into chunks, embeds each chunk
with a Hugging Face sentence-transformers model, and saves everything to
DATA_DIR. After this, the box needs no network at all.
"""
import json
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DATASET = os.getenv("KNOWLEDGE_DATASET", "wikimedia/wikipedia")
CONFIG = os.getenv("KNOWLEDGE_CONFIG", "20231101.simple")
MAX_DOCS = int(os.getenv("KNOWLEDGE_MAX_DOCS", "20000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_WORDS = 180
OVERLAP_WORDS = 30


def chunk(text: str):
    words = text.split()
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, max(len(words), 1), step):
        piece = words[start : start + CHUNK_WORDS]
        if len(piece) >= 30:
            yield " ".join(piece)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {DATASET} ({CONFIG}), up to {MAX_DOCS} articles")
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)

    chunks = []
    for i, row in enumerate(ds):
        if i >= MAX_DOCS:
            break
        for piece in chunk(row["text"]):
            chunks.append({"title": row["title"], "url": row.get("url", ""), "text": piece})
        if i and i % 1000 == 0:
            print(f"  {i} articles, {len(chunks)} chunks")

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    np.save(DATA_DIR / "vectors.npy", vectors)
    with open(DATA_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (DATA_DIR / "pack.json").write_text(
        json.dumps(
            {"dataset": DATASET, "config": CONFIG, "docs": min(i + 1, MAX_DOCS),
             "chunks": len(chunks), "embed_model": EMBED_MODEL},
            indent=2,
        )
    )
    print(f"Done. Knowledge pack saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
