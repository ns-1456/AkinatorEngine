"""
Download preprocessed Wikidata person data (entity-metadata) and convert to
data/raw_wikidata.csv for the Akinator engine.

Source: https://sourceforge.net/projects/entity-metadata/files/wikidata/person/
- wikidata_person_bio-2024-01-combined.7z (~94 MB, single CSV, ~5M rows)
- Columns: person, personLabel, sex_or_genderLabel, country_of_citizenshipLabel, etc.

Run from repo root: python src/fetch_preprocessed_data.py
"""
import os
import sys
import csv
import tempfile
import urllib.request

# Optional: py7zr for extracting .7z (install with: pip install py7zr)
try:
    import py7zr
except ImportError:
    py7zr = None

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_wikidata.csv")
# Direct download (SourceForge)
ARCHIVE_URL = "https://downloads.sourceforge.net/project/entity-metadata/wikidata/person/wikidata_person_bio-2024-01-combined.7z"
# Use at most this many rows (None = use all); keeps pipeline fast
MAX_ROWS = 100_000


def log(msg):
    print(msg, flush=True)


def download_archive(dest_path):
    """Download the 7z archive with progress."""
    log("Downloading {} ...".format(ARCHIVE_URL))
    log("(This may take a few minutes; file is ~94 MB)")

    def reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100, 100.0 * downloaded / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write("\r  {:.1f} MB / {:.1f} MB ({:.1f}%)".format(mb, total_mb, pct))
        sys.stdout.flush()

    urllib.request.urlretrieve(ARCHIVE_URL, dest_path, reporthook)
    print("", flush=True)
    log("Downloaded to {}.".format(dest_path))


def extract_csv(archive_path, out_dir):
    """Extract the 7z archive; return path to the CSV inside."""
    if not py7zr:
        log("Error: py7zr is required to extract .7z. Install with: pip install py7zr")
        raise SystemExit(1)

    log("Extracting archive...")
    with py7zr.SevenZipFile(archive_path, "r") as z:
        z.extractall(out_dir)

    # Find the CSV (archive contains a single CSV)
    for name in os.listdir(out_dir):
        if name.endswith(".csv"):
            return os.path.join(out_dir, name)
    # Sometimes extracted with subdir
    for root, _, files in os.walk(out_dir):
        for name in files:
            if name.endswith(".csv"):
                return os.path.join(root, name)
    log("Error: no CSV found in archive.")
    raise SystemExit(1)


def normalize_value(v):
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return ""
    return str(v).strip()


def map_row(raw, headers):
    """Map entity-metadata columns to our schema: itemLabel, occupationLabel, genderLabel, countryLabel, sitelinks."""
    # Common column name variants in entity-metadata CSVs
    def get(name_candidates, default=""):
        for n in name_candidates:
            if n in headers:
                return normalize_value(raw.get(n, default))
        return default

    item_label = get(["personLabel", "person_label", "label"], "")
    gender = get(["sex_or_genderLabel", "sex_or_gender_label", "genderLabel"], "")
    country = get(["country_of_citizenshipLabel", "country_of_citizenship_label", "countryLabel"], "")
    occupation = get(["occupationLabel", "occupation_label"], "")
    # We don't have sitelinks in this dataset; use 1 so popularity is uniform
    sitelinks = "1"

    return {
        "itemLabel": item_label or get(["person"], "").replace("http://www.wikidata.org/entity/", ""),
        "occupationLabel": occupation,
        "genderLabel": gender or "Unknown",
        "countryLabel": country or "Unknown",
        "sitelinks": sitelinks,
    }


def convert_to_our_csv(csv_path, out_path, max_rows=MAX_ROWS):
    """Read entity-metadata CSV and write our schema CSV."""
    log("Reading {} (max_rows={})...".format(csv_path, max_rows or "all"))

    cols_out = ["itemLabel", "occupationLabel", "genderLabel", "countryLabel", "sitelinks"]
    rows_written = 0
    skipped_no_label = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        headers = reader.fieldnames or []
        with open(out_path, "w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=cols_out, extrasaction="ignore")
            w.writeheader()
            for raw in reader:
                if max_rows and rows_written >= max_rows:
                    break
                row = map_row(raw, headers)
                if not row["itemLabel"]:
                    skipped_no_label += 1
                    continue
                w.writerow(row)
                rows_written += 1
                if rows_written % 50_000 == 0:
                    log("  Written {} rows...".format(rows_written))

    if skipped_no_label:
        log("  Skipped {} rows with no itemLabel.".format(skipped_no_label))
    log("Wrote {} rows to {}.".format(rows_written, out_path))
    return rows_written


def main():
    log("Preprocessed Wikidata person data (entity-metadata)")
    log("Output: {}".format(OUTPUT_FILE))

    os.makedirs(DATA_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = os.path.join(tmp, "wikidata_person_bio.7z")
        download_archive(archive_path)
        csv_path = extract_csv(archive_path, tmp)
        convert_to_our_csv(csv_path, OUTPUT_FILE, max_rows=MAX_ROWS)

    log("Done. Next: python src/processor.py")


if __name__ == "__main__":
    main()
