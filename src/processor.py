"""
Convert data/raw_wikidata.csv into GPU-ready PyTorch tensors.
Saves knowledge_base.pt, candidates.json, features.json, popularity.pt under data/.
Run from repo root after scraper.
"""
import os
import json
import pandas as pd
import torch

DATA_DIR = "data"
INPUT_CSV = os.path.join(DATA_DIR, "raw_wikidata.csv")
TOP_K_FEATURES = 500


def main():
    # 1. Load CSV and clean
    if not os.path.exists(INPUT_CSV):
        print("Error: {} not found. Run src/scraper.py first.".format(INPUT_CSV))
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    df = df.dropna(subset=["itemLabel"])
    df = df.drop_duplicates(subset=["itemLabel"], keep="first")
    df = df.fillna("")

    # 2. One-hot encoding: build binary columns for Occupation, Gender, Country
    # Multi-value (e.g. occupation "actor, director") -> one column per value
    rows = []
    for _, r in df.iterrows():
        row = {}
        # Occupation: split by comma
        for part in str(r.get("occupationLabel", "")).split(","):
            key = "Occupation_" + part.strip() or "Unknown"
            if key != "Occupation_":
                row[key] = 1
        # Gender
        g = str(r.get("genderLabel", "")).strip() or "Unknown"
        row["Gender_" + g] = 1
        # Country
        c = str(r.get("countryLabel", "")).strip() or "Unknown"
        row["Country_" + c] = 1
        rows.append(row)

    # Build full one-hot matrix
    all_keys = []
    for row in rows:
        all_keys.extend(row.keys())
    unique_features = list(dict.fromkeys(all_keys))

    onehot = []
    for row in rows:
        onehot.append([1 if f in row else 0 for f in unique_features])

    frame = pd.DataFrame(onehot, columns=unique_features)
    candidates = df["itemLabel"].astype(str).tolist()

    # 3. Dimensionality reduction: keep top 500 most frequent feature columns
    col_sums = frame.sum(axis=0)
    top_cols = col_sums.nlargest(TOP_K_FEATURES).index.tolist()
    frame = frame[top_cols]

    # Question text for each column (e.g. "Occupation: Actor")
    raw_names = list(frame.columns)
    features = []
    for name in raw_names:
        if name.startswith("Occupation_"):
            features.append("Is occupation " + name.replace("Occupation_", "") + "?")
        elif name.startswith("Gender_"):
            features.append("Is gender " + name.replace("Gender_", "") + "?")
        elif name.startswith("Country_"):
            features.append("Is country " + name.replace("Country_", "") + "?")
        else:
            features.append(name)
    M = len(features)
    N = len(candidates)

    # 4. Outputs: tensors on CPU, JSON lists
    # knowledge_base: Boolean (N, M)
    kb = torch.tensor(frame.values, dtype=torch.float32)
    # popularity: normalized sitelinks
    sitelinks = df["sitelinks"].replace("", 0).astype(float)
    sitelinks = sitelinks.fillna(0).values
    popularity = torch.tensor(sitelinks, dtype=torch.float32)
    if popularity.sum() == 0:
        popularity = torch.ones(N, dtype=torch.float32) / N
    else:
        popularity = popularity / popularity.sum()

    os.makedirs(DATA_DIR, exist_ok=True)

    torch.save(kb, os.path.join(DATA_DIR, "knowledge_base.pt"))
    torch.save(popularity, os.path.join(DATA_DIR, "popularity.pt"))
    with open(os.path.join(DATA_DIR, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=0)
    with open(os.path.join(DATA_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=0)

    print("Saved knowledge_base.pt ({} x {}).".format(N, M))
    print("Saved candidates.json, features.json, popularity.pt.")


if __name__ == "__main__":
    main()
