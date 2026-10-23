from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import ModelUnavailableError
from .schemas import (
    AdversarialRequest,
    AdversarialResponse,
    AnalysisResponse,
    AnalyzeRequest,
    ExplainRequest,
    ExplanationResponse,
    FeedbackRequest,
    HealthResponse,
)
from .service import AnalysisService
from .storage import Storage


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)

logger = logging.getLogger("smsense")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Run startup diagnostics using FastAPI's lifespan interface."""
    await startup_diagnostics()
    yield


app = FastAPI(
    title="SMSense API",
    description=(
        "AI-assisted SMS threat intelligence and "
        "vernacular phishing detection API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS AND RESPONSE SECURITY HEADERS
# ============================================================
#
# Only local development frontends are allowed by default. Override
# CORS_ALLOWED_ORIGINS with comma-separated exact origins when hosting
# the frontend elsewhere. CORS is not authentication; keep the API bound
# to loopback unless it is protected by an authenticated deployment layer.
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    """Avoid caching SMS/history responses and apply basic browser protections."""
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Cache-Control", "no-store")
    return response


# ============================================================
# SERVICES
# ============================================================

service = AnalysisService()
storage = Storage()


# ============================================================
# STARTUP DIAGNOSTICS
# ============================================================

async def startup_diagnostics() -> None:
    """
    Log the configured detection stack at startup.

    This does not fabricate availability.
    Each component reports its actual runtime state.
    """

    try:
        package_ok, package_detail = (
            service.hf.package_status()
        )
    except Exception as exc:
        package_ok = False
        package_detail = (
            f"HF status check failed: {type(exc).__name__}"
        )

    try:
        faiss_ok, faiss_detail = (
            service.similarity.status()
        )
    except Exception as exc:
        faiss_ok = False
        faiss_detail = (
            f"FAISS status check failed: {type(exc).__name__}"
        )

    logger.info(
        "SMSense API starting"
    )

    logger.info(
        "Configured Ollama chat model: %s",
        settings.ollama_chat_model,
    )

    logger.info(
        "Configured Ollama embedding model: %s",
        settings.ollama_embedding_model,
    )

    logger.info(
        "Configured primary SMS-Shield classifier: %s",
        settings.sms_shield_model,
    )

    logger.info(
        "Optional IndicBERT challenger enabled: %s",
        settings.indicbert_enabled,
    )

    logger.info(
        "Optional Hugging Face challenger runtime: available=%s detail=%s",
        package_ok,
        package_detail,
    )

    logger.info(
        "FAISS runtime: available=%s detail=%s",
        faiss_ok,
        faiss_detail,
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/api/health",
    response_model=HealthResponse,
)
async def health() -> HealthResponse:
    """
    Detailed component health.

    Used by:
        - SMSense System Health page
        - debugging
        - deployment verification
    """

    # --------------------------------------------------------
    # Ollama
    # --------------------------------------------------------

    try:
        ollama_ok, ollama_detail = (
            await service.ollama.status()
        )
    except Exception as exc:
        ollama_ok = False
        ollama_detail = (
            f"Ollama status check failed: "
            f"{type(exc).__name__}"
        )

    # --------------------------------------------------------
    # Qwen3 8B
    # --------------------------------------------------------

    try:
        (
            ollama_chat_ok,
            ollama_chat_detail,
        ) = await service.ollama.model_status(
            settings.ollama_chat_model
        )
    except Exception as exc:
        ollama_chat_ok = False
        ollama_chat_detail = (
            f"Chat model status check failed: "
            f"{type(exc).__name__}"
        )

    # --------------------------------------------------------
    # Qwen3 Embedding
    # --------------------------------------------------------

    try:
        (
            ollama_embedding_ok,
            ollama_embedding_detail,
        ) = await service.ollama.model_status(
            settings.ollama_embedding_model
        )
    except Exception as exc:
        ollama_embedding_ok = False
        ollama_embedding_detail = (
            f"Embedding model status check failed: "
            f"{type(exc).__name__}"
        )

    # --------------------------------------------------------
    # Primary SMS-Shield classifier and optional HF challenger
    # --------------------------------------------------------

    try:
        sms_shield_ok, sms_shield_detail = await service.sms_shield.status()
    except Exception as exc:
        sms_shield_ok = False
        sms_shield_detail = f"SMS-Shield status check failed: {type(exc).__name__}"

    try:
        hf_status = await service.hf.detailed_status()
    except Exception as exc:
        hf_status = {
            "indicbert": (False, f"Status check failed: {type(exc).__name__}"),
        }

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    try:
        faiss_ok, faiss_detail = (
            service.similarity.status()
        )
    except Exception as exc:
        faiss_ok = False
        faiss_detail = (
            f"FAISS status check failed: "
            f"{type(exc).__name__}"
        )

    # --------------------------------------------------------
    # SQLite
    # --------------------------------------------------------

    try:
        storage.stats()

        sqlite_ok = True
        sqlite_detail = "SQLite is writable"

    except Exception as exc:
        sqlite_ok = False
        sqlite_detail = (
            f"SQLite unavailable: "
            f"{type(exc).__name__}"
        )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    components = {
        "ollama": {
            "available": ollama_ok,
            "detail": ollama_detail,
        },

        "ollama_chat_qwen3_8b": {
            "available": ollama_chat_ok,
            "detail": ollama_chat_detail,
        },

        "ollama_embedding_qwen3_0_6b": {
            "available": ollama_embedding_ok,
            "detail": ollama_embedding_detail,
        },

        "sms_shield_classifier": {
            "available": sms_shield_ok,
            "detail": sms_shield_detail,
        },

        "faiss": {
            "available": faiss_ok,
            "detail": faiss_detail,
        },

        "google_safe_browsing": {
            "available": bool(settings.google_safe_browsing_api_key),
            "detail": (
                "API key configured; provider is checked during URL lookups"
                if settings.google_safe_browsing_api_key
                else "Not configured; URL analysis uses local structural evidence only"
            ),
        },

        "sqlite": {
            "available": sqlite_ok,
            "detail": sqlite_detail,
        },
    }

    if settings.indicbert_enabled:
        indicbert_ok, indicbert_detail = hf_status.get(
            "indicbert",
            (False, "Optional IndicBERT challenger status unavailable"),
        )
        components["hugging_face_indicbert"] = {
            "available": indicbert_ok,
            "detail": indicbert_detail,
        }

    # The core service requires the primary SMS-Shield model and writable
    # SQLite history. Qwen3 explanation/embedding, FAISS similarity, and the
    # optional IndicBERT challenger are diagnostic or enhancement components;
    # their absence must not report primary detection as unavailable.
    core_ok = bool(components["sms_shield_classifier"]["available"]) and bool(
        components["sqlite"]["available"]
    )

    return HealthResponse(
        status=(
            "ok"
            if core_ok
            else "degraded"
        ),
        components=components,
    )


# ============================================================
# LIGHTWEIGHT HEALTH CHECK
# ============================================================

@app.get("/api/healthz")
async def healthz() -> dict[str, str]:
    """
    Fast endpoint used by the frontend to determine whether
    the API server itself is reachable.
    """

    return {
        "status": "ok"
    }


# ============================================================
# ANALYZE MESSAGE
# ============================================================

@app.post(
    "/api/analyze",
    response_model=AnalysisResponse,
)
async def analyze(
    request: AnalyzeRequest,
) -> AnalysisResponse:
    """
    Analyze a single SMS.

    Important:
    - The backend is authoritative for the risk score.
    - The frontend does not calculate detection logic.
    """

    try:

        # Return the primary classification first. The UI requests the optional
        # Qwen explanation separately so slow/empty explainer output cannot hide
        # a completed SMS-Shield result.
        result = await service.analyze(
            request.message,
            request.sender,
            request.timestamp,
            post_hoc=False,
        )

    except ModelUnavailableError as exc:

        logger.error(
            "Model unavailable during analysis: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": str(exc),
                "hint": (
                    "Verify that the configured pretrained "
                    "models and Ollama services are available. "
                    "No fabricated security result was returned."
                ),
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected analysis failure"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "ANALYSIS_ERROR",
                "message": (
                    "The SMS analysis could not be completed."
                ),
            },
        ) from exc

    # --------------------------------------------------------
    # Store completed analysis
    # --------------------------------------------------------

    try:

        analysis_id = storage.save_analysis(
            result
        )

    except Exception as exc:

        logger.exception(
            "Failed to save analysis"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "STORAGE_ERROR",
                "message": (
                    "Analysis completed but could not "
                    "be stored in history."
                ),
            },
        ) from exc

    return AnalysisResponse(
        id=analysis_id,
        **result,
    )



@app.post("/api/explain", response_model=ExplanationResponse)
async def explain(request: ExplainRequest) -> ExplanationResponse:
    """Generate an optional explanation after the primary result is already visible."""
    payload = {
        "final_score": request.risk_score,
        "final_label": request.label,
        "category": request.category,
        "top_evidence": request.top_evidence,
        "raw_message": request.raw_message,
        "normalized_message": request.normalized_message,
        "suspicious_phrases": request.suspicious_phrases,
    }
    try:
        explanation = await service.ollama.explain(payload)
        return ExplanationResponse(
            summary=explanation["summary"],
            suspicious_phrases=explanation.get(
                "suspicious_phrases", request.suspicious_phrases
            ),
            recommended_action=explanation["recommended_action"],
            available=True,
            detail="Optional Qwen explanation generated",
        )
    except Exception as exc:
        # Do not turn an explanation-provider outage into an analysis outage.
        logger.warning(
            "Optional explanation endpoint unavailable (%s)",
            type(exc).__name__,
        )
        return ExplanationResponse(
            summary=(
                "The optional AI explanation is unavailable. "
                "The risk score and verdict above were already produced by "
                "the primary detection pipeline."
            ),
            suspicious_phrases=request.suspicious_phrases[:10],
            recommended_action=(
                "Verify the sender independently. Do not open unexpected links "
                "or share OTPs, PINs, passwords, or payment details until the "
                "message is verified through an official channel."
            ),
            available=False,
            detail=f"Optional explanation unavailable ({type(exc).__name__})",
        )


# ============================================================
# ADVERSARIAL TEST
# ============================================================

@app.post(
    "/api/adversarial-test",
    response_model=AdversarialResponse,
)
async def adversarial_test(
    request: AdversarialRequest,
) -> AdversarialResponse:
    """
    Analyze multiple message variants.

    Used by the SMSense Adversarial Test page.

    The endpoint deliberately does not save every adversarial
    test as normal history unless the backend service itself
    changes that behavior.
    """

    results = []

    try:

        for item in request.messages:

            result = await service.analyze(
                item.message,
                item.sender,
                item.timestamp,
                post_hoc=False,
            )

            results.append(
                AnalysisResponse(
                    **result
                )
            )

    except ModelUnavailableError as exc:

        logger.error(
            "Model unavailable during adversarial test: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected adversarial test failure"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "ADVERSARIAL_TEST_ERROR",
                "message": (
                    "The adversarial test could not be completed."
                ),
            },
        ) from exc

    return AdversarialResponse(
        results=results,
        invariants=[
            "raw_message is preserved exactly",
            "normalization does not replace raw message",
            "post-hoc explanation cannot alter score or label",
        ],
    )


# ============================================================
# HISTORY
# ============================================================

@app.get("/api/history")
async def history(
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
    ),
) -> list[dict]:
    """
    Return recent stored analyses.

    Frontend can request:
        /api/history?limit=500
    """

    try:
        return storage.history(limit)

    except Exception as exc:

        logger.exception(
            "Failed to load history"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "HISTORY_ERROR",
                "message": (
                    "Analysis history could not be loaded."
                ),
            },
        ) from exc


# ============================================================
# HISTORY DETAIL
# ============================================================

@app.get(
    "/api/history/{analysis_id}"
)
async def history_detail(
    analysis_id: int,
) -> dict:
    """
    Return one complete analysis.
    """

    try:

        result = storage.get(
            analysis_id
        )

    except Exception as exc:

        logger.exception(
            "Failed to load analysis %s",
            analysis_id,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "HISTORY_DETAIL_ERROR",
                "message": (
                    "The requested analysis "
                    "could not be loaded."
                ),
            },
        ) from exc

    if result is None:

        raise HTTPException(
            status_code=404,
            detail={
                "code": "ANALYSIS_NOT_FOUND",
                "message": "Analysis not found.",
            },
        )

    return result


# ============================================================
# STATS
# ============================================================

@app.get("/api/stats")
async def stats() -> dict:
    """
    Return aggregate statistics.

    This endpoint should eventually be the primary source for
    dashboard aggregate numbers.

    The frontend can also calculate temporary prototype
    statistics directly from /api/history.
    """

    try:

        result = storage.stats()

        # Guarantee a dictionary response.
        if not isinstance(result, dict):
            raise ValueError(
                "storage.stats() must return a dictionary"
            )

        return result

    except Exception as exc:

        logger.exception(
            "Failed to load statistics"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "STATS_ERROR",
                "message": (
                    "Dashboard statistics could not "
                    "be loaded."
                ),
            },
        ) from exc


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/api/feedback")
async def feedback(
    request: FeedbackRequest,
) -> dict[str, str]:
    """
    Store user feedback for an analysis.
    """

    try:

        analysis = storage.get(
            request.analysis_id
        )

    except Exception as exc:

        logger.exception(
            "Failed to locate analysis for feedback"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "FEEDBACK_LOOKUP_ERROR",
                "message": (
                    "The analysis could not be checked."
                ),
            },
        ) from exc

    if analysis is None:

        raise HTTPException(
            status_code=404,
            detail={
                "code": "ANALYSIS_NOT_FOUND",
                "message": "Analysis not found.",
            },
        )

    try:

        storage.feedback(
            request.analysis_id,
            request.helpful,
            request.note,
        )

    except Exception as exc:

        logger.exception(
            "Failed to save feedback"
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "FEEDBACK_STORAGE_ERROR",
                "message": (
                    "Feedback could not be saved."
                ),
            },
        ) from exc

    return {
        "status": "recorded"
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=False,
    )