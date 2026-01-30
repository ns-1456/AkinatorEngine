"""
Fetch top 10,000 most popular humans/characters from Wikidata.
Phase 1: Minimal SPARQL (item Q-id + sitelinks only, no label service).
Phase 2: Resolve labels via Wikidata REST API (wbgetentities).
Saves data/raw_wikidata.csv. Run from repo root.
"""
import os
import re
import csv
import time
import requests

SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_wikidata.csv")
TIMEOUT = 60
MAX_RETRIES = 2
LABEL_BATCH_SIZE = 50
API_DELAY_SEC = 0.2
# Smaller limit so SPARQL returns faster; increase later if needed
SPARQL_LIMIT = 5000

# Minimal query: no SERVICE wikibase:label, no OPTIONALs — avoids 504 on SPARQL endpoint
QUERY_TEMPLATE = """
SELECT ?item ?sitelinks WHERE {
  { ?item wdt:P31 wd:Q5 . } UNION { ?item wdt:P31 wd:Q95074 . }
  ?item wikibase:sitelinks ?sitelinks .
}
ORDER BY DESC(?sitelinks)
LIMIT %d
"""
QUERY = QUERY_TEMPLATE % SPARQL_LIMIT

QID_PATTERN = re.compile(r"^https?://www\.wikidata\.org/entity/(Q\d+)$")


def qid_from_uri(uri):
    m = QID_PATTERN.match(uri.strip())
    return m.group(1) if m else None


def _log(msg):
    print(msg, flush=True)


def fetch_sparql():
    """Fetch item + sitelinks from SPARQL (no labels)."""
    _log("Phase 1: Fetching item list from SPARQL (limit={})...".format(SPARQL_LIMIT))
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                _log("  Retry {} of {}...".format(attempt, MAX_RETRIES))
            r = requests.post(
                SPARQL_URL,
                data={"query": QUERY, "format": "json"},
                headers={"User-Agent": "AkinatorClone/1.0"},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            bindings = data.get("results", {}).get("bindings", [])
            rows = []
            for b in bindings:
                item_uri = b.get("item", {}).get("value", "")
                sitelinks = b.get("sitelinks", {}).get("value", "")
                qid = qid_from_uri(item_uri)
                if qid:
                    rows.append({"qid": qid, "sitelinks": sitelinks})
            return rows
        except requests.exceptions.Timeout:
            _log("  Timeout after {}s.".format(TIMEOUT))
        except requests.exceptions.RequestException as e:
            _log("  Request failed: {}".format(e))
    return None


def fetch_labels_batch(qids):
    """Fetch English labels for a list of Q-ids via wbgetentities."""
    if not qids:
        return {}
    ids = "|".join(qids)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": ids,
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": "AkinatorClone/1.0"},
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            entities = data.get("entities", {})
            result = {}
            for qid, ent in entities.items():
                if ent.get("type") == "item":
                    labels = ent.get("labels", {})
                    result[qid] = labels.get("en", {}).get("value", qid)
            return result
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES:
                time.sleep(API_DELAY_SEC * 2)
            else:
                _log("  Label batch failed: {}".format(e))
                return {q: q for q in qids}
    return {q: q for q in qids}


# Fallback Q-ids (well-known people/characters) when SPARQL fails — fetched via API only
FALLBACK_QIDS = [
    "Q76", "Q937", "Q5", "Q8441", "Q36834", "Q8023", "Q30", "Q6581097",
    "Q142", "Q48259", "Q123", "Q423", "Q8499", "Q881", "Q1339", "Q1744",
    "Q33986", "Q11575", "Q184", "Q383", "Q131", "Q1065", "Q6581072",
    "Q6581097", "Q95074", "Q16334295", "Q10833314", "Q22908", "Q22686",
]


def fetch_fallback_rows():
    """Build rows from fallback Q-ids using the REST API only (no SPARQL)."""
    _log("SPARQL failed. Using fallback list of {} entities via API...".format(len(FALLBACK_QIDS)))
    rows = []
    for qid in FALLBACK_QIDS:
        rows.append({"qid": qid, "sitelinks": ""})
    return rows


def main():
    _log("Starting scraper...")
    rows = fetch_sparql()
    if not rows:
        rows = fetch_fallback_rows()
    if not rows:
        _log("Error: no results from SPARQL and fallback failed.")
        raise SystemExit(1)

    _log("Phase 2: Resolving labels via REST API (batches of {})...".format(LABEL_BATCH_SIZE))
    all_qids = [r["qid"] for r in rows]
    qid_to_label = {}
    for i in range(0, len(all_qids), LABEL_BATCH_SIZE):
        batch = all_qids[i : i + LABEL_BATCH_SIZE]
        qid_to_label.update(fetch_labels_batch(batch))
        if (i + LABEL_BATCH_SIZE) % 500 == 0 or i + LABEL_BATCH_SIZE >= len(all_qids):
            _log("  Resolved {}/{} labels.".format(min(i + LABEL_BATCH_SIZE, len(all_qids)), len(all_qids)))
        time.sleep(API_DELAY_SEC)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cols = ["itemLabel", "occupationLabel", "genderLabel", "countryLabel", "sitelinks"]
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                "itemLabel": qid_to_label.get(r["qid"], r["qid"]),
                "occupationLabel": "",
                "genderLabel": "",
                "countryLabel": "",
                "sitelinks": r["sitelinks"],
            })

    _log("Saved to {} ({} rows).".format(OUTPUT_FILE, len(rows)))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: {}".format(e), flush=True)
        raise
