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
        if not self.available:
            return False, "faiss-cpu is not installed; similarity is unavailable"
        if not self.index_path.exists():
            return False, f"FAISS is installed but the index is missing at {self.index_path}; similarity is unavailable"
        try:
            index = self.faiss.read_index(str(self.index_path))
            if index.ntotal == 0:
                return False, f"FAISS index exists but is empty at {self.index_path}; similarity is unavailable"
            return True, f"FAISS index loaded with {index.ntotal} scam-example vectors"
        except Exception as exc:
            return False, f"FAISS index could not be loaded: {type(exc).__name__}"

    async def search(self, raw: str, normalized: str, ollama: OllamaClient) -> dict[str, Any]:
        index_status, index_detail = self.status()
        if not index_status:
            return {"available": False, "matches": [], "detail": index_detail}
        embedding = await ollama.embed(f"raw: {raw}\nnormalized: {normalized}")
        index = self.faiss.read_index(str(self.index_path))
        import numpy as np  # type: ignore
        vector = np.asarray([embedding], dtype="float32")
        distances, ids = index.search(vector, 3)
        return {"available": True, "matches": [{"id": int(i), "distance": float(d)} for i, d in zip(ids[0], distances[0]) if int(i) >= 0], "detail": "Nearest-neighbor evidence only"}