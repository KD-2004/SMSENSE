import asyncio
import logging
from typing import Any

from .config import settings
from .fusion import fuse
from .models import HuggingFaceModels, ModelSignal, ModelUnavailableError, OllamaClient, SmsShieldClient
from .normalizer import normalize_message
from .parser import parse_message
from .rules import behavior_signals, language_hint, url_signals
from .similarity import SimilarityStore
from .url_reputation import UrlReputationClient

logger = logging.getLogger(__name__)


class AnalysisService:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.sms_shield = SmsShieldClient()
        self.hf = HuggingFaceModels()
        self.similarity = SimilarityStore()
        self.url_reputation = UrlReputationClient()

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

        # URL reputation is opt-in. Only extracted URL strings are sent to the
        # configured provider; SMS text, sender and other message data stay local.
        # Provider errors leave reputation as not_checked and never block analysis.
        url_analysis = url_signals(parsed)
        try:
            reputation_results = await asyncio.wait_for(
                self.url_reputation.check_urls(parsed.urls),
                timeout=settings.url_reputation_timeout_seconds,
            )
            reputation_by_url = {
                str(item.get("url")): item
                for item in reputation_results
                if isinstance(item, dict) and item.get("url")
            }
            url_items = url_analysis.get("urls", [])
            if isinstance(url_items, list):
                for url_item in url_items:
                    if not isinstance(url_item, dict):
                        continue
                    reputation = reputation_by_url.get(str(url_item.get("url", "")))
                    if reputation:
                        url_item["reputation"] = reputation.get("reputation", "not_checked")
                        url_item["reputation_provider"] = reputation.get("provider")
                        url_item["reputation_detail"] = reputation.get("detail")
                        url_item["threat_types"] = reputation.get("threat_types", [])
            url_analysis["evidence_basis"] = (
                "URL structure plus optional Google Safe Browsing reputation; "
                "a no_match result is not a guarantee that a URL is safe"
            )
        except Exception as exc:
            logger.warning("Optional URL reputation enrichment unavailable (%s)", type(exc).__name__)

        # SMS-Shield is the primary Indian SMS model. Its 0–100 risk field is
        # model-provided evidence, not a calibrated probability. Keep its content
        # contribution bounded at 30/100 until our labelled benchmark justifies
        # changing the fusion weights.
        try:
            model_assessment = await asyncio.wait_for(
                self.sms_shield.classify(message, sender),
                timeout=settings.classifier_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ModelUnavailableError(
                "SMS-Shield classification exceeded the configured "
                f"{settings.classifier_timeout_seconds:g}-second deadline. "
                "Try a shorter message, verify the local Ollama runtime is responsive, "
                "or increase SMSENSE_CLASSIFIER_TIMEOUT_SECONDS in .env."
            ) from exc

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
        if settings.indicbert_enabled:
            try:
                challenger_signal = await asyncio.wait_for(
                    self.hf.challenger_signal(message),
                    timeout=settings.challenger_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Optional IndicBERT challenger timed out after %.1f seconds; continuing without it",
                    settings.challenger_timeout_seconds,
                )
                challenger_signal = ModelSignal(
                    "IndicBERT",
                    False,
                    0.0,
                    "Optional IndicBERT challenger timed out; excluded from scoring",
                )
            except Exception as exc:
                logger.warning(
                    "Optional IndicBERT challenger failed (%s); continuing without it",
                    type(exc).__name__,
                )
                challenger_signal = ModelSignal(
                    "IndicBERT",
                    False,
                    0.0,
                    f"Optional IndicBERT challenger unavailable ({type(exc).__name__}); excluded from scoring",
                )
            if challenger_signal is not None:
                model_signals.append(challenger_signal)

        # Similarity is supporting evidence, not a probability. The store owns
        # metric-aware distance conversion. Keep core text analysis usable if the
        # optional embedding/similarity service is unavailable.
        similarity_result: dict[str, Any]
        similarity_signal = 0.0
        try:
            # Similarity is optional evidence. Bound it so a missing/slow
            # embedding model cannot hide a completed SMS-Shield prediction.
            similarity_result = await asyncio.wait_for(
                self.similarity.search(
                    parsed.raw, normalized.normalized, self.ollama
                ),
                timeout=settings.similarity_timeout_seconds,
            )
            similarity_signal = float(
                similarity_result.get("similarity_signal", 0.0) or 0.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Optional similarity enrichment timed out after %.1f seconds; continuing without it",
                settings.similarity_timeout_seconds,
            )
            similarity_result = {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": (
                    "Optional semantic similarity timed out after "
                    f"{settings.similarity_timeout_seconds:.1f} seconds; "
                    "the primary SMS analysis continued without it"
                ),
            }
        except ModelUnavailableError as exc:
            similarity_result = {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": str(exc),
            }
        except Exception as exc:
            logger.warning(
                "Optional similarity enrichment failed (%s); continuing without it",
                type(exc).__name__,
            )
            similarity_result = {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": (
                    "Optional semantic similarity unavailable "
                    f"({type(exc).__name__}); the primary SMS analysis continued"
                ),
            }

        # The bundled FAISS index is only a tiny demo set. Keep its nearest-
        # neighbour signal visible for diagnostics but out of risk scoring until
        # a group-split labelled evaluation demonstrates reliable incremental value.
        similarity_result["scoring_enabled"] = settings.similarity_scoring_enabled
        similarity_fusion_signal = (
            similarity_signal if settings.similarity_scoring_enabled else 0.0
        )
        if not settings.similarity_scoring_enabled:
            detail = str(similarity_result.get("detail", "")).strip()
            explanation = (
                "Semantic similarity is diagnostic-only and excluded from the "
                "risk score until it is validated on an independently labelled SMS set."
            )
            similarity_result["detail"] = f"{detail} {explanation}".strip()
        similarity_result["score_contribution"] = (
            similarity_fusion_signal if settings.similarity_scoring_enabled else 0.0
        )

        result = fuse(
            parsed=parsed,
            normalized=normalized,
            behavior=behavior,
            content_signal=content_signal,
            similarity_signal=similarity_fusion_signal,
            url_analysis=url_analysis,
        )
        # Record the evidence actually used by fusion, which can be lower than
        # the raw retrieval signal when independent evidence is absent.
        similarity_result["score_contribution"] = float(
            result.evidence.get("similarity", 0.0) or 0.0
        )
        model_risk = int(model_assessment.get("risk", 0) or 0)
        model_verdict = str(model_assessment.get("verdict", "unknown"))
        verdict_supported = bool(model_assessment.get("verdict_supported", True))
        verdict_consistent = bool(model_assessment.get("verdict_consistent", True))
        review_reasons: list[str] = []
        data_gaps: list[str] = []

        if not verdict_supported:
            review_reasons.append("model_returned_unsupported_verdict")
        elif not verdict_consistent:
            review_reasons.append("model_verdict_risk_band_mismatch")
        if model_risk >= 50 and result.score < settings.suspicious_threshold:
            review_reasons.append("model_risk_not_reflected_in_final_score")
        if model_risk >= 70 and result.score < settings.high_risk_threshold:
            review_reasons.append("high_model_risk_below_high_risk_threshold")

        url_review_items = url_analysis.get("urls", [])
        if not sender:
            data_gaps.append("sender_not_provided")
        if any(
            isinstance(item, dict) and item.get("reputation") == "not_checked"
            for item in url_review_items
        ):
            data_gaps.append("url_reputation_not_checked")
        if not similarity_result.get("available", False):
            data_gaps.append("semantic_similarity_unavailable")

        decision_diagnostics = {
            "model_risk": model_risk,
            "model_verdict": model_verdict,
            "model_verdict_supported": verdict_supported,
            "model_verdict_consistent": verdict_consistent,
            "final_risk_score": result.score,
            "final_label": result.label,
            "model_minus_final_score": model_risk - result.score,
            "requires_manual_review": bool(review_reasons),
            "review_reasons": list(dict.fromkeys(review_reasons)),
            "data_gaps": list(dict.fromkeys(data_gaps)),
        }

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
            "decision_diagnostics": decision_diagnostics,
            "language": language_hint(message),
            "raw_message": parsed.raw,
            "normalized_message": normalized.normalized,
            "sender": sender,
            "evidence": result.evidence,
            "suspicious_phrases": behavior.combination_hits,
            "url_analysis": url_analysis,
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
