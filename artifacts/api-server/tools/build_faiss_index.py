"""Build the local FAISS demo index from Ollama embeddings.

This intentionally requires a reachable Ollama embedding model. It never
creates vectors from random or locally fabricated values.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.models import ModelUnavailableError, OllamaClient


async def build() -> None:
    seed_path = Path(__file__).resolve().parents[1] / "data" / "seed_scam_examples.json"
    seeds = json.loads(seed_path.read_text())
    client = OllamaClient()
    vectors = []
    for item in seeds:
        vectors.append(await client.embed(item["text"]))
    import faiss

    matrix = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)
    output = Path(settings.faiss_index_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output))
    metadata = output.with_suffix(".metadata.json")
    metadata.write_text(json.dumps(seeds, indent=2))
    print(f"Built FAISS index {output} with {index.ntotal} clearly labeled demo seed vectors.")


if __name__ == "__main__":
    try:
        asyncio.run(build())
    except ModelUnavailableError as exc:
        raise SystemExit(f"Cannot build FAISS index: {exc}") from exc