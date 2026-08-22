"""
Build the local FAISS similarity index from Ollama embeddings.

The index stores embeddings of clearly labeled demo scam examples.

IMPORTANT:
- No random/fabricated vectors are allowed.
- All vectors must come from the configured Ollama embedding model.
- The index uses FAISS IndexFlatL2.
- Metadata is stored next to the index.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow importing the app package when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.models import ModelUnavailableError, OllamaClient

logger = logging.getLogger("faiss_builder")


async def build() -> None:
    base_dir = Path(__file__).resolve().parents[1]

    seed_path = base_dir / "data" / "seed_scam_examples.json"

    if not seed_path.exists():
        raise SystemExit(
            f"Seed dataset not found: {seed_path}"
        )

    seeds: list[dict[str, Any]] = json.loads(
        seed_path.read_text(encoding="utf-8")
    )

    if not seeds:
        raise SystemExit("Seed dataset is empty.")

    # Validate seed records.
    valid_seeds: list[dict[str, Any]] = []

    for idx, item in enumerate(seeds):
        if not isinstance(item, dict):
            logger.warning("Skipping seed %s: not an object", idx)
            continue

        text = str(item.get("text", "")).strip()

        if not text:
            logger.warning(
                "Skipping seed %s: missing text",
                idx,
            )
            continue

        valid_seeds.append(item)

    if not valid_seeds:
        raise SystemExit("No valid seed examples found.")

    logger.info(
        "Building FAISS index from %d seed examples",
        len(valid_seeds),
    )

    client = OllamaClient()

    # Confirm the configured embedding model is available.
    embedding_ok, embedding_detail = await client.model_status(
        settings.ollama_embedding_model
    )

    if not embedding_ok:
        raise ModelUnavailableError(
            f"Embedding model unavailable: {embedding_detail}"
        )

    logger.info(
        "Using embedding model: %s",
        settings.ollama_embedding_model,
    )

    vectors: list[list[float]] = []

    for index_number, item in enumerate(valid_seeds, start=1):
        text = str(item["text"]).strip()

        logger.info(
            "Embedding seed %d/%d",
            index_number,
            len(valid_seeds),
        )

        vector = await client.embed(text)

        if not vector:
            raise ModelUnavailableError(
                f"Empty embedding returned for seed {index_number}"
            )

        vector_array = np.asarray(
            vector,
            dtype=np.float32,
        )

        if vector_array.ndim != 1:
            vector_array = vector_array.reshape(-1)

        if not np.all(np.isfinite(vector_array)):
            raise ModelUnavailableError(
                f"Invalid embedding values for seed {index_number}"
            )

        vectors.append(vector_array.tolist())

    # Ensure every embedding has the same dimension.
    dimensions = {len(vector) for vector in vectors}

    if len(dimensions) != 1:
        raise ModelUnavailableError(
            f"Embedding dimension mismatch across seeds: {sorted(dimensions)}"
        )

    dimension = dimensions.pop()

    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )

    matrix = np.ascontiguousarray(matrix)

    if matrix.shape != (len(valid_seeds), dimension):
        raise ModelUnavailableError(
            f"Unexpected embedding matrix shape: {matrix.shape}"
        )

    # ---------------------------------------------------------------
    # FAISS INDEX
    # ---------------------------------------------------------------
    #
    # Keep L2 because similarity.py is configured to understand L2.
    #
    import faiss  # type: ignore

    index = faiss.IndexFlatL2(dimension)

    index.add(matrix)

    if index.ntotal != len(valid_seeds):
        raise RuntimeError(
            f"FAISS index count mismatch: "
            f"expected={len(valid_seeds)}, actual={index.ntotal}"
        )

    output = Path(settings.faiss_index_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(output),
    )

    # ---------------------------------------------------------------
    # METADATA
    # ---------------------------------------------------------------

    metadata = output.with_suffix(".metadata.json")

    metadata_payload = {
        "embedding_model": settings.ollama_embedding_model,
        "metric": "L2",
        "dimension": dimension,
        "vector_count": int(index.ntotal),
        "source": "data/seed_scam_examples.json",
        "examples": valid_seeds,
    }

    metadata.write_text(
        json.dumps(
            metadata_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("========================================")
    print("FAISS INDEX BUILT SUCCESSFULLY")
    print("========================================")
    print(f"Index       : {output}")
    print(f"Metadata    : {metadata}")
    print(f"Model       : {settings.ollama_embedding_model}")
    print(f"Metric      : L2")
    print(f"Dimension   : {dimension}")
    print(f"Vectors     : {index.ntotal}")
    print("========================================")


if __name__ == "__main__":
    try:
        asyncio.run(build())
    except ModelUnavailableError as exc:
        raise SystemExit(
            f"Cannot build FAISS index: {exc}"
        ) from exc