# Field Kit — Offline Knowledge Box

An open-source, fully offline question-answering box, built live on stream for Hacktoberfest 2026.
A small **open-weight model** answers from a **local knowledge pack** built from Simple English
Wikipedia. It runs **natively on Windows** (no Docker, no WSL, no Ollama): Python + transformers
on the GPU, with a phone-first web UI in the *Field Kit* style (US-map palette: map paper,
public-land mint, forest ink, road red for danger only).

Your PC is the box: it runs the models and the search. Any phone on the same Wi‑Fi opens the UI in
its browser. Running the model on the phone itself is a goal for a future stream.

## Run it

```powershell
.\run.ps1
```

On the first run it creates a Python env (if needed), builds the knowledge pack (20k articles, a
few minutes on a GPU) and serves http://localhost:8080. It also prints the address to use from a
phone on the same Wi‑Fi. To let the phone in, allow port 8080 in Windows Firewall.

| Flag | What it does |
|---|---|
| `-Ingest` | Rebuild the knowledge pack |
| `-MaxDocs 5000` | Smaller pack, faster to build |
| `-Model smollm2-135m` | Default chat model (`smollm2-135m`, `smollm2-1.7b`, `llama3.1-8b`) |

You can also switch models in the Field Notes header. The server keeps one model on the GPU at a time.

## Models

| Key | Model | Open-ness | Notes |
|---|---|---|---|
| `smollm2-135m` | HuggingFaceTB/SmolLM2-135M-Instruct | Fully open | Instant, but weak |
| `smollm2-1.7b` | HuggingFaceTB/SmolLM2-1.7B-Instruct | Fully open | Default; good for a demo |
| `llama3.1-8b` | meta-llama/Llama-3.1-8B-Instruct | Open weights (Llama license) | Gated: accept the license on Hugging Face and `huggingface-cli login` first. Loaded 4-bit (~6 GB VRAM) |
| Survey | HuggingFaceTB/SmolVLM2-2.2B-Instruct | Fully open | Vision model for photo ID |

Embeddings: `sentence-transformers/all-MiniLM-L6-v2`. Knowledge: `wikimedia/wikipedia` (`20231101.simple`).

## What's in it

- **Home**: status (library size, GPU, model), ask box, legend of tools.
- **Field Notes**: streamed answers grounded in the library, numbered citations, source chips, tok/s.
- **Reference**: semantic search over the pack and an offline article reader.
- **Survey** *(work in progress)*: take a photo, a small vision model guesses what it is, then
  looks it up in the library.

Planned (shown as V2 on the home screen): Quadrangle (offline maps), Guides (first-aid packs),
on-phone mode, Spanish & Chinese UI, night mode.

## Files

| File | What it is |
|---|---|
| `run.ps1` | One-command setup and start |
| `server.py` | FastAPI server: retrieval, chat, survey, article API |
| `ingest.py` | Builds the knowledge pack into `data/` |
| `static/index.html` | The whole web UI |

API: `GET /api/status` · `POST /api/ask` (NDJSON stream: `sources`, `delta`…, `done`) ·
`GET /api/search?q=` · `GET /api/article?title=` · `POST /api/survey`

## Attribution

Knowledge pack content comes from Wikipedia under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Each answer links the articles it used. Models and datasets keep their own licenses; check each
Hugging Face model card.

## Safety

This is a reference tool, not medical, legal or emergency advice. Answers come from a small model
and a partial encyclopedia, so check anything important. Never eat or handle anything because an
app named it.

## License

MIT for code.
