# AkinatorEngine

An Akinator-style “guess who” game: the engine asks yes/no-style questions and narrows down a person from a Wikidata-backed knowledge base using Bayesian inference. Optional GPU acceleration via PyTorch.

## Features

- **Bayesian engine** – Mistake-tolerant likelihoods and top-candidate discrimination so questions stay informative.
- **Wikidata-backed** – People and attributes (gender, country, occupation, etc.) from Wikidata; labels via SPARQL + REST API or a preprocessed dataset.
- **Web UI** – Single-page interface at the root URL with Akinator-like styling and five answer options: Yes, No, Don’t know, Probably, Probably not.
- **REST API** – Endpoints for new game, answer, and guess; docs at `/docs`.

## Requirements

- Python 3.x
- For GPU support: PyTorch with CUDA (see [pytorch.org](https://pytorch.org)). CPU-only works fine for smaller knowledge bases.

## Quick start

**1. Clone and install**

```bash
git clone https://github.com/ns-1456/AkinatorEngine.git
cd AkinatorEngine
pip install -r requirements.txt
```

**2. Generate data** (run once)

The repo does **not** include large data files (they exceed GitHub’s size limit). You must generate them locally.

- **Option A – Preprocessed data (recommended)**  
  Downloads a ~94 MB preprocessed Wikidata person-bio CSV and writes up to 100k rows to `data/raw_wikidata.csv`.

  ```bash
  python src/fetch_preprocessed_data.py
  ```

- **Option B – Live Wikidata**  
  Fetches Q-IDs via SPARQL and labels via the Wikidata REST API. May hit timeouts on some networks; falls back to a small built-in list if SPARQL fails.

  ```bash
  python src/scraper.py
  ```

**3. Build the knowledge base** (run once)

Creates `data/*.pt` and `data/*.json` from `data/raw_wikidata.csv`:

```bash
python src/processor.py
```

**4. Run the server**

```bash
python main.py
```

Then open **http://localhost:8000** in a browser. API docs: **http://localhost:8000/docs**.

Alternative: `uvicorn src.server:app --reload` (run from project root so `data/` is found).

## Project structure

```
AkinatorEngine/
├── main.py              # Entry point; runs server on port 8000
├── requirements.txt
├── data/                # Generated (not in repo): raw_wikidata.csv, *.pt, *.json
├── static/
│   └── index.html       # Web UI served at /
├── src/
│   ├── server.py        # FastAPI app, game session, routes
│   ├── model.py         # Bayesian engine, question selection, beliefs
│   ├── processor.py     # Build knowledge base from CSV
│   ├── scraper.py       # Fetch from Wikidata (SPARQL + API)
│   └── fetch_preprocessed_data.py  # Download preprocessed CSV
└── UI/                  # Reference UI assets (e.g. flash-ui)
```

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serves the game UI |
| POST | `/api/game/start` | Start a new game; returns `game_id`, first question |
| POST | `/api/game/answer` | Submit answer (body: `game_id`, `answer` 0–1); returns next question or guess |
| GET | `/api/game/guess` | Get current top guess (e.g. for “Give up”) |

Answer encoding: **Yes** = 1, **No** = 0, **Don’t know** = 0.5, **Probably** = 0.75, **Probably not** = 0.25.

## License

See repository for license information.
