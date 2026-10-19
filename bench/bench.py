"""Benchmark open-weight models on this box: speed (tokens/sec) plus answers saved for quality review.

Usage (from repo root, with the stack running):
    python bench/bench.py --tier gpu --models hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M other/model:Q4_K_M
    python bench/bench.py --tier pi-like --push   # also uploads results to a Hugging Face dataset

Tiers are labels you choose: "pi-like", "mac", "gpu". Results append to bench/results.csv.
"""
import argparse
import csv
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

APP_URL = os.getenv("APP_URL", "http://localhost:8080")
QUESTIONS = [
    "How do I purify water if I have no filter?",
    "What are the signs of hypothermia?",
    "How do you find north without a compass?",
    "What foods are high in vitamin C?",
    "Who built the pyramids of Giza and when?",
]
OUT = Path(__file__).parent / "results.csv"
FIELDS = ["timestamp", "tier", "host", "model", "question", "tokens_per_sec", "latency_s", "answer", "sources"]


def run(models, tier):
    rows = []
    with httpx.Client(timeout=600) as client:
        for model in models:
            client.post(f"{APP_URL}/api/ask", json={"question": "Hello", "model": model})  # warm-up / load
            for q in QUESTIONS:
                t0 = time.time()
                r = client.post(f"{APP_URL}/api/ask", json={"question": q, "model": model})
                r.raise_for_status()
                d = r.json()
                rows.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "tier": tier, "host": platform.node(), "model": model, "question": q,
                    "tokens_per_sec": d["tokens_per_sec"], "latency_s": round(time.time() - t0, 2),
                    "answer": d["answer"], "sources": json.dumps([s["title"] for s in d["sources"]]),
                })
                print(f"{tier:8} {model.split('/')[-1]:40} {d['tokens_per_sec']:>6} tok/s  {q[:40]}")
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


def push():
    from huggingface_hub import HfApi

    user = os.environ["HF_USERNAME"]
    repo = f"{user}/offline-knowledge-box-bench"
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=str(OUT), path_in_repo="results.csv", repo_id=repo, repo_type="dataset")
    print(f"Pushed to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tier", required=True, help='label, e.g. "pi-like", "mac", "gpu"')
    p.add_argument("--models", nargs="+", default=[os.getenv("CHAT_MODEL", "hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M")])
    p.add_argument("--push", action="store_true", help="upload results.csv to a Hugging Face dataset")
    args = p.parse_args()
    run(args.models, args.tier)
    if args.push:
        push()
