import logging
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import ModelUnavailableError
from .schemas import AdversarialRequest, AdversarialResponse, AnalysisResponse, AnalyzeRequest, FeedbackRequest, HealthResponse
from .service import AnalysisService
from .storage import Storage

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("bharatphish_guard")
app = FastAPI(title="BharatPhish Guard API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
service = AnalysisService()
storage = Storage()


@app.on_event("startup")
async def startup_diagnostics() -> None:
    package_ok, package_detail = service.hf.package_status()
    faiss_ok, faiss_detail = service.similarity.status()
    logger.info("Configured Ollama chat model: %s", settings.ollama_chat_model)
    logger.info("Configured Ollama embedding model: %s", settings.ollama_embedding_model)
    logger.info("Configured HinSpam model: %s", settings.hinspam_model)
    logger.info("Configured SilverGuard model: %s", settings.silverguard_model)
    logger.info("Hugging Face runtime package: available=%s detail=%s", package_ok, package_detail)
    logger.info("FAISS runtime: available=%s detail=%s", faiss_ok, faiss_detail)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_ok, ollama_detail = await service.ollama.status()
    ollama_chat_ok, ollama_chat_detail = await service.ollama.model_status(settings.ollama_chat_model)
    ollama_embedding_ok, ollama_embedding_detail = await service.ollama.model_status(settings.ollama_embedding_model)
    hf_status = await service.hf.detailed_status()
    faiss_ok, faiss_detail = service.similarity.status()
    try:
        storage.stats()
        sqlite_ok, sqlite_detail = True, "SQLite is writable"
    except Exception as exc:
        sqlite_ok, sqlite_detail = False, f"SQLite unavailable: {type(exc).__name__}"
    components = {
        "ollama": {"available": ollama_ok, "detail": ollama_detail},
        "ollama_chat_qwen3_8b": {"available": ollama_chat_ok, "detail": ollama_chat_detail},
        "ollama_embedding_qwen3_0_6b": {"available": ollama_embedding_ok, "detail": ollama_embedding_detail},
        "hugging_face_hinspam": {"available": hf_status["hinspam"][0], "detail": hf_status["hinspam"][1]},
        "hugging_face_silverguard": {"available": hf_status["silverguard"][0], "detail": hf_status["silverguard"][1]},
        "faiss": {"available": faiss_ok, "detail": faiss_detail},
        "sqlite": {"available": sqlite_ok, "detail": sqlite_detail},
    }
    return HealthResponse(status="ok" if all(item["available"] for item in components.values()) else "degraded", components=components)


@app.get("/api/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalyzeRequest) -> AnalysisResponse:
    try:
        result = await service.analyze(request.message, request.sender, request.timestamp, post_hoc=True)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": str(exc), "hint": "Install/load the configured pretrained models and retry; no security result was fabricated."}) from exc
    analysis_id = storage.save_analysis(result)
    return AnalysisResponse(id=analysis_id, **result)


@app.post("/api/adversarial-test", response_model=AdversarialResponse)
async def adversarial_test(request: AdversarialRequest) -> AdversarialResponse:
    results = []
    try:
        for item in request.messages:
            result = await service.analyze(item.message, item.sender, item.timestamp, post_hoc=False)
            results.append(AnalysisResponse(**result))
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail={"code": "MODEL_UNAVAILABLE", "message": str(exc)}) from exc
    return AdversarialResponse(results=results, invariants=["raw_message is preserved exactly", "post-hoc explanation cannot alter score or label"])


@app.get("/api/history")
async def history(limit: int = Query(default=50, ge=1, le=500)) -> list[dict]:
    return storage.history(limit)


@app.get("/api/history/{analysis_id}")
async def history_detail(analysis_id: int) -> dict:
    result = storage.get(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/api/stats")
async def stats() -> dict:
    return storage.stats()


@app.post("/api/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, str]:
    if storage.get(request.analysis_id) is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    storage.feedback(request.analysis_id, request.helpful, request.note)
    return {"status": "recorded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.port, reload=False)