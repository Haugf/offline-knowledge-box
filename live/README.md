# Field Kit — live v1

The version of the Offline Knowledge Box we're building live on stream. It runs **natively on
Windows** (no Docker, no WSL, no Ollama): Python + transformers on the GPU, the same knowledge
pack as [`app/ingest.py`](../app/ingest.py), and a phone-first web UI based on the
*Field Kit* mockups (US-map-style palette: map paper, public-land mint, forest ink, road red for
danger only).

## Run it

```powershell
cd live
.\run.ps1
```

First run creates a Python env (if needed), builds the knowledge pack from Simple English
Wikipedia (20k articles, a few minutes on a GPU), then serves http://localhost:8080.
The script also prints the address to use from a phone on the same Wi‑Fi.

| Flag | What it does |
|---|---|
| `-Ingest` | Rebuild the knowledge pack |
| `-MaxDocs 5000` | Smaller pack, faster to build |
| `-Model smollm2-135m` | Default model (`smollm2-135m`, `smollm2-1.7b`, `llama3.1-8b`) |

You can also switch models in the Field Notes header; the server keeps one on the GPU at a time.

## Models

| Key | Model | Open-ness | Notes |
|---|---|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct | Fully open | Instant, but weak |
| `smollm2-1.7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | Fully open | Default; good for a demo |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | Open weights (Llama license) | Gated: accept the license on Hugging Face and `huggingface-cli login` first. Loaded 4-bit (~6 GB VRAM) |

## What's in v1

- **Home** — status (library size, GPU, model), ask box, legend of tools.
- **Field Notes** — streamed answers grounded in the library, numbered citations, source chips, tok/s.
- **Reference** — semantic search over the pack and an offline article reader.

Not yet (shown as V2 on the home screen): Survey (camera ID), Quadrangle (offline maps),
Guides (first-aid packs), ruler/compass, Spanish & Chinese UI, night mode.

## API

`GET /api/status` · `POST /api/ask` (NDJSON stream: `sources`, `delta`…, `done`) ·
`GET /api/search?q=` · `GET /api/article?title=`
