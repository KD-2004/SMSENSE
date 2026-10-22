from typing import Any
import logging

from .fusion import fuse
from .models import HuggingFaceModels, ModelSignal, ModelUnavailableError, OllamaClient, SmsShieldClient
from .normalizer import normalize_message
from .parser import parse_message
from .rules import behavior_signals, language_hint, url_signals
from .similarity import SimilarityStore

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.sms_shield = SmsShieldClient()
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

        # SMS-Shield is the primary Indian SMS model. Its 0–100 risk field is
        # model-provided evidence, not a calibrated probability. Keep its content
        # contribution bounded at 30/100 until our labelled benchmark justifies
        # changing the fusion weights.
        model_assessment = await self.sms_shield.classify(message, sender)
        content_signal = float(model_assessment["risk"]) * 0.30
        model_signals = [
            ModelSignal(
                "SMS-Shield",
                True,
                float(model_assessment["risk"]) * 0.15,
                (
                    f"Primary model risk={model_assessment['risk']}/100; "
                    f"verdict={model_assessment['verdict']}; "
                    f"segments={model_assessment['segments_analyzed']}"
                ),
            )
        ]
        # Optional IndicBERT remains a diagnostic challenger only. It is not
        # included in production fusion because its performance on our own
        # held-out SMS set has not yet been established.
        challenger_signal = await self.hf.challenger_signal(message)
        if challenger_signal is not None:
            model_signals.append(challenger_signal)

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
            "category": (
                result.category
                if result.category != "GENERAL_MESSAGE"
                else str(
                    model_assessment.get("scam_type")
                    if model_assessment.get("risk", 0) >= 50 and model_assessment.get("scam_type")
                    else model_assessment.get("category") or "GENERAL_MESSAGE"
                ).upper().replace(" ", "_")
            ),
            "model_assessment": model_assessment,
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
