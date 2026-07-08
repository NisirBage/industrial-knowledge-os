"""Seed the demo dataset (../demo_data/) into a running IKIOS backend.

Uploads each curated document via POST /api/v1/documents/upload — the exact
same code path a real user upload takes, so this script never bypasses the
real ingestion pipeline. Requires the backend to already be running (see
docs/getting-started/installation.md#seeding-demo-data).

Usage:
    python scripts/seed_demo_data.py
    IKIOS_API_BASE_URL=http://localhost:8000/api/v1 python scripts/seed_demo_data.py
"""

import os
import sys
from pathlib import Path

import httpx

_DEMO_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "demo_data"
_CONTENT_TYPES = {".txt": "text/plain", ".md": "text/markdown", ".pdf": "application/pdf"}
_EXCLUDED = {"README.md"}


def _demo_documents() -> list[Path]:
    return sorted(p for p in _DEMO_DATA_DIR.iterdir() if p.suffix in _CONTENT_TYPES and p.name not in _EXCLUDED)


def main() -> None:
    base_url = os.environ.get("IKIOS_API_BASE_URL", "http://localhost:8000/api/v1")
    documents = _demo_documents()

    if not documents:
        print(f"No demo documents found in {_DEMO_DATA_DIR}")
        sys.exit(1)

    print(f"Seeding {len(documents)} demo documents into {base_url} ...")

    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"Backend not reachable at {base_url}: {exc}")
            print("Start it first — see docs/getting-started/installation.md")
            sys.exit(1)

        for path in documents:
            with path.open("rb") as f:
                response = client.post(
                    "/documents/upload",
                    files={"file": (path.name, f, _CONTENT_TYPES[path.suffix])},
                )
            if response.status_code != 200:
                print(f"  FAILED  {path.name}: {response.status_code} {response.text}")
                continue
            body = response.json()
            print(f"  OK      {path.name} -> {body['num_chunks']} chunks, {body['num_entities']} entities, {body['num_relations']} relations")

    print("\nDone. Try the questions in demo_data/ground_truth.json against POST /api/v1/chat.")


if __name__ == "__main__":
    main()
