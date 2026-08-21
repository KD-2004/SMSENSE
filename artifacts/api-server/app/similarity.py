import json
import logging
from pathlib import Path
from typing import Any
from .config import settings
from .models import ModelUnavailableError, OllamaClient

logger = logging.getLogger(__name__)


class SimilarityStore:
    def __init__(self) -> None:
        self.index_path = Path(settings.faiss_index_path)
        self.available = False
        try:
            import faiss  # type: ignore
            self.faiss = faiss
            self.available = True
        except ImportError:
            self.faiss = None

    def status(self) -> tuple[bool, str]:
        return (True, "FAISS available") if self.available else (False, "faiss-cpu is not installed; similarity is unavailable")

    async def search(self, raw: str, normalized: str, ollama: OllamaClient) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "matches": [], "detail": self.status()[1]}
        embedding = await ollama.embed(f"raw: {raw}\nnormalized: {normalized}")
        if not self.index_path.exists():
            return {"available": True, "matches": [], "detail": "FAISS index has no seeded scam examples"}
        index = self.faiss.read_index(str(self.index_path))
        import numpy as np  # type: ignore
        vector = np.asarray([embedding], dtype="float32")
        distances, ids = index.search(vector, 3)
        return {"available": True, "matches": [{"id": int(i), "distance": float(d)} for i, d in zip(ids[0], distances[0]) if int(i) >= 0], "detail": "Nearest-neighbor evidence only"}