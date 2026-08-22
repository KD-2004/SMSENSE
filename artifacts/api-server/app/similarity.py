from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .models import ModelUnavailableError, OllamaClient

logger = logging.getLogger(__name__)


class SimilarityStore:
    """
    FAISS-backed semantic similarity store.

    Responsibilities:
    - Load and validate the local FAISS index.
    - Embed the incoming message using Ollama.
    - Search nearest scam examples.
    - Return similarity evidence only.
    - Never convert similarity into a phishing probability by itself.
    """

    DEFAULT_TOP_K = 3

    def __init__(self) -> None:
        self.index_path = Path(settings.faiss_index_path)

        self.faiss = None
        self._index = None
        self._index_mtime_ns: int | None = None

        try:
            import faiss  # type: ignore

            self.faiss = faiss
        except ImportError:
            self.faiss = None
            logger.warning("FAISS is not installed.")

    # ------------------------------------------------------------------
    # INDEX MANAGEMENT
    # ------------------------------------------------------------------

    def _load_index(self):
        """Load and cache the FAISS index safely."""
        if self.faiss is None:
            raise ModelUnavailableError(
                "FAISS is not installed; similarity is unavailable"
            )

        if not self.index_path.exists():
            raise ModelUnavailableError(
                f"FAISS index is missing at {self.index_path}"
            )

        try:
            mtime_ns = self.index_path.stat().st_mtime_ns

            # Reuse cached index if the file has not changed.
            if (
                self._index is not None
                and self._index_mtime_ns == mtime_ns
            ):
                return self._index

            index = self.faiss.read_index(str(self.index_path))

            if index.ntotal <= 0:
                raise ModelUnavailableError(
                    f"FAISS index exists but is empty at {self.index_path}"
                )

            if getattr(index, "d", 0) <= 0:
                raise ModelUnavailableError(
                    "FAISS index has an invalid embedding dimension"
                )

            self._index = index
            self._index_mtime_ns = mtime_ns

            logger.info(
                "Loaded FAISS index: vectors=%s dimension=%s metric=%s",
                index.ntotal,
                index.d,
                getattr(index, "metric_type", "unknown"),
            )

            return index

        except ModelUnavailableError:
            raise
        except Exception as exc:
            logger.exception("Failed to load FAISS index")
            raise ModelUnavailableError(
                f"FAISS index could not be loaded: {type(exc).__name__}"
            ) from exc

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    def status(self) -> tuple[bool, str]:
        """Return whether the FAISS index is ready for similarity search."""
        if self.faiss is None:
            return False, "faiss-cpu is not installed; similarity is unavailable"

        if not self.index_path.exists():
            return (
                False,
                f"FAISS index is missing at {self.index_path}; "
                "similarity is unavailable",
            )

        try:
            index = self._load_index()

            return (
                True,
                (
                    f"FAISS index loaded with {index.ntotal} "
                    f"scam-example vectors "
                    f"(dimension={index.d})"
                ),
            )

        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # VECTOR VALIDATION
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_vector(
        embedding: list[float],
        expected_dimension: int,
        metric_type: int | None,
        faiss_module: Any,
    ) -> np.ndarray:
        """
        Convert an Ollama embedding into the exact shape FAISS expects.

        For inner-product indexes, normalize the vector so that cosine-like
        similarity is used consistently.

        For L2 indexes, leave the vector unchanged because the index builder
        determines the metric semantics.
        """
        if not embedding:
            raise ModelUnavailableError(
                "Embedding model returned an empty vector"
            )

        vector = np.asarray(embedding, dtype=np.float32)

        if vector.ndim != 1:
            vector = vector.reshape(-1)

        if vector.size != expected_dimension:
            raise ModelUnavailableError(
                "Embedding dimension mismatch: "
                f"query={vector.size}, index={expected_dimension}"
            )

        if not np.all(np.isfinite(vector)):
            raise ModelUnavailableError(
                "Embedding contains NaN or infinite values"
            )

        # Inner-product FAISS index:
        # normalize to unit length for cosine-like similarity.
        if (
            metric_type is not None
            and metric_type == faiss_module.METRIC_INNER_PRODUCT
        ):
            norm = float(np.linalg.norm(vector))

            if norm <= 0.0:
                raise ModelUnavailableError(
                    "Embedding vector has zero magnitude"
                )

            vector = vector / norm

        return np.ascontiguousarray(vector.reshape(1, -1))

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    async def search(
        self,
        raw: str,
        normalized: str,
        ollama: OllamaClient,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Search for semantically similar scam examples.

        The raw and normalized representations are embedded together.
        Similarity is evidence only; it is NOT treated as a phishing
        probability.
        """
        if not raw or not raw.strip():
            return {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": "Message is empty; similarity not performed",
            }

        try:
            index = self._load_index()
        except ModelUnavailableError as exc:
            return {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": str(exc),
            }

        # Keep both representations as evidence.
        embedding_input = (
            f"raw: {raw.strip()}\n"
            f"normalized: {normalized.strip()}"
        )

        try:
            embedding = await ollama.embed(embedding_input)

        except ModelUnavailableError:
            raise

        except Exception as exc:
            logger.exception("Embedding generation failed")
            raise ModelUnavailableError(
                f"Embedding generation failed: {type(exc).__name__}"
            ) from exc

        try:
            vector = self._prepare_vector(
                embedding=embedding,
                expected_dimension=index.d,
                metric_type=getattr(index, "metric_type", None),
                faiss_module=self.faiss,
            )
        except ModelUnavailableError:
            raise

        k = top_k if top_k is not None else self.DEFAULT_TOP_K
        k = max(1, min(int(k), int(index.ntotal)))

        try:
            distances, ids = index.search(vector, k)
        except Exception as exc:
            logger.exception("FAISS search failed")
            raise ModelUnavailableError(
                f"FAISS search failed: {type(exc).__name__}"
            ) from exc

        matches: list[dict[str, Any]] = []

        for raw_id, raw_distance in zip(ids[0], distances[0]):
            match_id = int(raw_id)
            distance = float(raw_distance)

            if match_id < 0:
                continue

            if not np.isfinite(distance):
                continue

            matches.append(
                {
                    "id": match_id,
                    "distance": distance,
                }
            )

        top_distance = (
            matches[0]["distance"]
            if matches
            else None
        )

        # IMPORTANT:
        # Do not turn FAISS distance into a fake probability.
        # Keep a bounded evidence value only if the backend's fusion
        # layer explicitly wants a simple similarity signal.
        #
        # This is a conservative signal:
        # - strong nearest-neighbor match -> up to 5
        # - no result -> 0
        #
        # The threshold is intentionally configurable.
        similarity_signal = 0.0

        if matches and top_distance is not None:
            similarity_signal = self._distance_to_signal(
                top_distance,
                metric_type=getattr(index, "metric_type", None),
            )

        return {
            "available": True,
            "matches": matches,
            "top_distance": top_distance,
            "similarity_signal": similarity_signal,
            "detail": (
                "Nearest-neighbor evidence only; "
                "similarity is not a phishing probability"
            ),
        }

    # ------------------------------------------------------------------
    # DISTANCE → EVIDENCE SIGNAL
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_to_signal(
        distance: float,
        metric_type: int | None,
    ) -> float:
        """
        Convert the nearest-neighbor distance into a SMALL bounded
        evidence contribution.

        This intentionally does NOT claim to be a probability.

        For L2:
            smaller distance = stronger match

        For inner product:
            larger score = stronger match
            because vectors are normalized before search
        """
        if not np.isfinite(distance):
            return 0.0

        # Inner product after L2 normalization is approximately cosine.
        if metric_type is not None:
            try:
                import faiss  # type: ignore

                if metric_type == faiss.METRIC_INNER_PRODUCT:
                    cosine = max(-1.0, min(1.0, distance))

                    # Conservative evidence mapping.
                    if cosine >= 0.85:
                        return 5.0
                    if cosine >= 0.75:
                        return 3.0
                    if cosine >= 0.65:
                        return 1.0

                    return 0.0
            except Exception:
                pass

        # L2 distance fallback.
        # These are NOT probabilities.
        if distance <= 0.60:
            return 5.0
        if distance <= 0.90:
            return 3.0
        if distance <= 1.20:
            return 1.0

        return 0.0