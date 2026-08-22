"""
Validate BharatPhish Guard against a small fixed SMS test set.

This script:
- Runs the real pretrained models.
- Runs the real rule-based evidence fusion.
- Runs FAISS similarity when available.
- Runs Qwen3 as a post-hoc explanation layer.
- Prints complete evidence without fabricating unavailable signals.

IMPORTANT:
This is a validation utility, not a training script.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.fusion import fuse
from app.models import (
    HuggingFaceModels,
    ModelUnavailableError,
    OllamaClient,
)
from app.normalizer import normalize_message
from app.parser import parse_message
from app.rules import (
    behavior_signals,
    sender_signals,
    url_signals,
)
from app.similarity import SimilarityStore


# ---------------------------------------------------------------------
# VALIDATION MESSAGES
# ---------------------------------------------------------------------

MESSAGES: list[dict[str, str | None]] = [
    {
        "message": "KYC upd8 kro warna account blck hoga",
        "sender": "VM-SBIHELP",
    },
    {
        "message": "KYC update karo warna account block hoga",
        "sender": "VM-SBIHELP",
    },
    {
        "message": "Your OTP for login is 4821. Do not share it with anyone.",
        "sender": "SBI",
    },
    {
        "message": "Your account will be blocked. Verify KYC immediately.",
        "sender": "BANK",
    },
    {
        "message": "Kal class 10 baje hai bhai.",
        "sender": "Rahul",
    },
    {
        "message": "KYC upd8 kro abhi warna acc0unt blck hoga",
        "sender": "VM-SBIHELP",
    },
    {
        "message": "Congratulations! You are selected. Pay ₹999 registration fee to start work.",
        "sender": "+919876543210",
    },
]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def similarity_signal_from_result(
    similarity_result: dict[str, Any],
) -> float:
    """
    Use the similarity signal calculated by SimilarityStore.

    Do NOT manually convert raw FAISS distance into a probability.
    """
    if not similarity_result.get("available"):
        return 0.0

    value = similarity_result.get("similarity_signal", 0.0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(5.0, value))


def safe_model_result_dict(result: Any) -> dict[str, Any]:
    """Convert a dataclass/model result into a JSON-safe dictionary."""
    if hasattr(result, "__dict__"):
        return dict(result.__dict__)

    if isinstance(result, dict):
        return result

    return {"value": str(result)}


# ---------------------------------------------------------------------
# MAIN VALIDATION
# ---------------------------------------------------------------------

async def main() -> None:
    models = HuggingFaceModels()
    similarity = SimilarityStore()
    ollama = OllamaClient()

    rows: list[dict[str, Any]] = []

    for number, item in enumerate(MESSAGES, start=1):
        raw = str(item["message"])
        sender = item["sender"]

        print(
            f"\n{'=' * 80}\n"
            f"TEST {number}/{len(MESSAGES)}\n"
            f"{raw}\n"
            f"Sender: {sender}\n"
            f"{'=' * 80}"
        )

        # -------------------------------------------------------------
        # PARSE
        # -------------------------------------------------------------

        parsed = parse_message(
            raw,
            sender,
            None,
        )

        # -------------------------------------------------------------
        # NORMALIZATION / OBFUSCATION
        # -------------------------------------------------------------

        normalized = normalize_message(raw)

        # -------------------------------------------------------------
        # BEHAVIOR
        # -------------------------------------------------------------

        behavior = behavior_signals(raw)

        # -------------------------------------------------------------
        # PRETRAINED MODEL SIGNALS
        # -------------------------------------------------------------

        model_signals = await models.signals(
            raw,
            sender,
        )

        # Keep model outputs separate.
        hinspam_output = (
            safe_model_result_dict(model_signals[0])
            if len(model_signals) >= 1
            else {
                "available": False,
                "detail": "HinSpam result unavailable",
            }
        )

        silverguard_output = (
            safe_model_result_dict(model_signals[1])
            if len(model_signals) >= 2
            else {
                "available": False,
                "detail": "SilverGuard result unavailable",
            }
        )

        # -------------------------------------------------------------
        # CONTENT SIGNAL
        # -------------------------------------------------------------

        content_signal = 0.0

        for model_result in model_signals:
            score = getattr(model_result, "score", 0.0)

            try:
                score = float(score)
            except (TypeError, ValueError):
                score = 0.0

            content_signal += score

        # The fusion layer itself caps content at 30.
        content_signal = max(0.0, min(30.0, content_signal))

        # -------------------------------------------------------------
        # URL
        # -------------------------------------------------------------

        url_evidence = url_signals(parsed)

        # -------------------------------------------------------------
        # SENDER
        # -------------------------------------------------------------

        sender_evidence = sender_signals(sender)

        # -------------------------------------------------------------
        # SIMILARITY / FAISS
        # -------------------------------------------------------------

        try:
            similarity_result = await similarity.search(
                raw,
                normalized.normalized,
                ollama,
            )

            similarity_signal = similarity_signal_from_result(
                similarity_result
            )

        except ModelUnavailableError as exc:
            similarity_result = {
                "available": False,
                "matches": [],
                "top_distance": None,
                "similarity_signal": 0.0,
                "detail": str(exc),
            }
            similarity_signal = 0.0

        # -------------------------------------------------------------
        # RULE-BASED FUSION
        # -------------------------------------------------------------

        fusion = fuse(
            parsed=parsed,
            normalized=normalized,
            behavior=behavior,
            content_signal=content_signal,
            similarity_signal=similarity_signal,
        )

        # -------------------------------------------------------------
        # QWEN3 POST-HOC EXPLANATION
        # -------------------------------------------------------------

        try:
            explanation = await ollama.explain(
                {
                    "final_score": fusion.score,
                    "final_label": fusion.label,
                    "category": fusion.category,
                    "top_evidence": fusion.top,
                    "raw_message": raw,
                    "normalized_message": normalized.normalized,
                    "suspicious_phrases": behavior.combination_hits,
                }
            )

        except ModelUnavailableError as exc:
            explanation = {
                "status": "unavailable",
                "detail": str(exc),
            }

        # -------------------------------------------------------------
        # COMPLETE RESULT
        # -------------------------------------------------------------

        row = {
            "test_number": number,
            "message": raw,
            "sender": sender,

            "language_hint": (
                "not_available_here"
            ),

            "hinspam_output": hinspam_output,
            "silverguard_output": silverguard_output,

            "url_evidence": url_evidence,
            "sender_evidence": sender_evidence,

            "behavior_evidence": (
                safe_model_result_dict(behavior)
            ),

            "obfuscation_evidence": (
                safe_model_result_dict(normalized)
            ),

            "similarity_result": similarity_result,

            "final_rule_based_score": fusion.score,
            "final_label": fusion.label,
            "category": fusion.category,
            "evidence": fusion.evidence,
            "top_evidence": fusion.top,

            "qwen_explanation": explanation,
        }

        rows.append(row)

        print(
            json.dumps(
                row,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------

    summary = []

    for row in rows:
        summary.append(
            {
                "test_number": row["test_number"],
                "message": row["message"],
                "risk_score": row["final_rule_based_score"],
                "label": row["final_label"],
                "category": row["category"],
                "content": row["evidence"].get("content", 0),
                "sender": row["evidence"].get("sender", 0),
                "url": row["evidence"].get("url", 0),
                "behavior": row["evidence"].get("behavior", 0),
                "obfuscation": row["evidence"].get("obfuscation", 0),
                "similarity": row["evidence"].get("similarity", 0),
            }
        )

    print("\n")
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())