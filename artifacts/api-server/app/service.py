from typing import Any
from .fusion import fuse
from .models import HuggingFaceModels, ModelUnavailableError, OllamaClient
from .normalizer import normalize_message
from .parser import parse_message
from .rules import behavior_signals, language_hint, sender_signals, url_signals
from .similarity import SimilarityStore


class AnalysisService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.hf = HuggingFaceModels()
        self.similarity = SimilarityStore()

    async def analyze(self, message: str, sender: str | None, timestamp: Any = None, post_hoc: bool = False) -> dict[str, Any]:
        parsed = parse_message(message, sender, timestamp)
        normalized = normalize_message(message)
        behavior = behavior_signals(message)
        # Security decisions fail closed: model signals are required and are never
        # replaced with heuristic "fake" model scores.
        model_signals = await self.hf.signals(message, sender)
        content_signal = min(30, sum(signal.score for signal in model_signals))
        similarity = await self.similarity.search(parsed.raw, normalized.normalized, self.ollama)
        similarity_signal = min(5, max([0] + [max(0, 5 - match["distance"]) for match in similarity.get("matches", [])]))
        result = fuse(parsed, normalized, behavior, content_signal, similarity_signal)
        payload = {
            "risk_score": result.score, "label": result.label, "category": result.category, "language": language_hint(message),
            "raw_message": parsed.raw, "normalized_message": normalized.normalized, "sender": sender,
            "evidence": result.evidence, "suspicious_phrases": behavior.combination_hits,
            "url_analysis": url_signals(parsed), "obfuscation": normalized.__dict__,
            "explanation": {"top_contributing_evidence": result.top, "model_signals": [signal.__dict__ for signal in model_signals], "caps": {"content": 30, "url": 20, "behavior": 20, "sender": 15, "obfuscation": 10, "similarity": 5}},
        }
        if post_hoc:
            payload["post_hoc"] = await self.ollama.explain({"final_score": result.score, "final_label": result.label, "top_evidence": result.top, "raw_message": parsed.raw, "normalized_message": normalized.normalized, "suspicious_phrases": behavior.combination_hits})
        return payload