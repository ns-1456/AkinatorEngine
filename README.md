# Akinator Clone (GPU Bayesian Inference)

Akinator-style "guess who" game using GPU-accelerated Bayesian inference over a Wikidata knowledge base.

**Requirements:** Python 3.x. For GPU support, install PyTorch with CUDA 12.x (see [pytorch.org](https://pytorch.org)).

## How to run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Fetch data** (run once; creates `data/raw_wikidata.csv`)
   - **Option A – Preprocessed data (recommended):** Download ~94 MB entity-metadata CSV (~100k rows used). No SPARQL, works offline after download.
     ```bash
     pip install py7zr
     python src/fetch_preprocessed_data.py
     ```
   - **Option B – Live Wikidata:** Run the scraper (uses SPARQL + API; may time out on some networks).
     ```bash
     python src/scraper.py
     ```

3. **Build the knowledge base** (run once; creates `data/*.pt` and `data/*.json`)
   ```bash
   python src/processor.py
   ```

4. **Start the server**
   ```bash
   python main.py
   ```
   Or: `uvicorn src.server:app --reload`

   Open http://localhost:8000 (and check /docs for the API).
