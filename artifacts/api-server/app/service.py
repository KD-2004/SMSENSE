from typing import Any
import logging

from .fusion import fuse
from .models import HuggingFaceModels, ModelUnavailableError, OllamaClient
from .normalizer import normalize_message
from .parser import parse_message
from .rules import behavior_signals, language_hint, url_signals
from .similarity import SimilarityStore

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.hf = HuggingFaceModels()
        self.similarity = SimilarityStore()

    async def analyze(
        self,
        message: str,
        sender: str | None,
        timestamp: Any = None,
        post_hoc: bool = False,
    ) -> dict[str, Any]:
        parsed = parse_message(message, sender, timestamp)
        normalized = normalize_message(message)
        behavior = behavior_signals(message)

        # Required content classifiers must be available; never invent fallback scores.
        model_signals = await self.hf.signals(message, sender)
        content_signal = min(30.0, sum(float(signal.score) for signal in model_signals))

        # Similarity is supporting evidence, not a probability. The store owns
        # metric-aware distance conversion. Keep core text analysis usable if the
        # optional embedding/similarity service is unavailable.
        similarity_result: dict[str, Any]
        similarity_signal = 0.0
        try:
            similarity_result = await self.similarity.search(
                parsed.raw, normalized.normalized, self.ollama
            )
            similarity_signal = float(
                similarity_result.get("similarity_signal", 0.0) or 0.0
            )
        except ModelUnavailableError as exc:
            similarity_result = {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": str(exc),
            }

        result = fuse(
            parsed=parsed,
            normalized=normalized,
            behavior=behavior,
            content_signal=content_signal,
            similarity_signal=similarity_signal,
        )
        payload: dict[str, Any] = {
            "risk_score": result.score,
            "label": result.label,
            "category": result.category,
            "language": language_hint(message),
            "raw_message": parsed.raw,
            "normalized_message": normalized.normalized,
            "sender": sender,
            "evidence": result.evidence,
            "suspicious_phrases": behavior.combination_hits,
            "url_analysis": url_signals(parsed),
            "similarity": similarity_result,
            "obfuscation": normalized.__dict__,
            "explanation": {
                "top_contributing_evidence": result.top,
                "model_signals": [signal.__dict__ for signal in model_signals],
                "caps": {
                    "content": 30,
                    "url": 20,
                    "behavior": 20,
                    "sender": 15,
                    "obfuscation": 10,
                    "similarity": 5,
                },
            },
            "explanation_status": {
                "requested": bool(post_hoc),
                "available": None if post_hoc else False,
                "detail": "pending" if post_hoc else "not requested",
            },
        }
        if post_hoc:
            try:
                payload["post_hoc"] = await self.ollama.explain(
                    {
                        "final_score": result.score,
                        "final_label": result.label,
                        "category": result.category,
                        "top_evidence": result.top,
                        "raw_message": parsed.raw,
                        "normalized_message": normalized.normalized,
                        "suspicious_phrases": behavior.combination_hits,
                    }
                )
                payload["explanation_status"] = {
                    "requested": True,
                    "available": True,
                    "detail": "Optional Qwen3 explanation generated",
                }
            except Exception as exc:
                # Explanations are deliberately post-hoc and must never change
                # a completed detection result or make the analysis unavailable.
                logger.warning(
                    "Optional post-hoc explanation unavailable (%s)",
                    type(exc).__name__,
                )
                payload["post_hoc"] = {
                    "summary": (
                        "The optional explanation service is unavailable. "
                        "The risk score and label were computed by the detection pipeline."
                    ),
                    "suspicious_phrases": behavior.combination_hits,
                    "recommended_action": (
                        "Do not click unexpected links or share OTPs, PINs, passwords, "
                        "or payment details until the message is verified independently."
                    ),
                }
                payload["explanation_status"] = {
                    "requested": True,
                    "available": False,
                    "detail": f"Optional explanation unavailable ({type(exc).__name__})",
                }
        return payload
