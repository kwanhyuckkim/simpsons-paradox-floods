"""Download cached parquet bundle from Zenodo.

Verifies SHA256 hashes against ``src/floodbhm/data/manifests.py``. Skips
re-download for files already present with correct hashes.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

# Placeholder — populate with the actual Zenodo direct-download URLs once the
# DOI is minted. Each entry: (filename, url, sha256).
ZENODO_FILES: list[tuple[str, str, str]] = [
    # ("peaks_v2_category_ver2.parquet",
    #  "https://zenodo.org/record/XXXXXXX/files/peaks_v2_category_ver2.parquet?download=1",
    #  "abc123..."),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    target = Path(__file__).resolve().parent.parent / "data"
    target.mkdir(parents=True, exist_ok=True)

    if not ZENODO_FILES:
        print("Zenodo file manifest is empty. Once the dataset is published, populate ZENODO_FILES.")
        return 1

    for name, url, expected in ZENODO_FILES:
        dest = target / name
        if dest.exists() and sha256_of(dest) == expected:
            print(f"OK  {name} (hash verified)")
            continue
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url, dest)
        actual = sha256_of(dest)
        if actual != expected:
            print(f"HASH MISMATCH for {name}: expected {expected}, got {actual}")
            return 2
        print(f"OK  {name} (downloaded + hash verified)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
