import os
import pytest

from app.models import HuggingFaceModels, ModelUnavailableError, OllamaClient
from app.similarity import SimilarityStore


REAL_RUNTIME = os.getenv("RUN_REAL_MODEL_TESTS", "").lower() == "true"


@pytest.mark.integration
@pytest.mark.skipif(not REAL_RUNTIME, reason="set RUN_REAL_MODEL_TESTS=true to run live model tests")
def test_huggingface_models_load():
    import asyncio
    statuses = asyncio.run(HuggingFaceModels().detailed_status())
    assert statuses["hinspam"][0], statuses["hinspam"][1]
    assert statuses["silverguard"][0], statuses["silverguard"][1]


@pytest.mark.integration
@pytest.mark.skipif(not REAL_RUNTIME, reason="set RUN_REAL_MODEL_TESTS=true to run live model tests")
def test_ollama_models_and_faiss_are_live():
    import asyncio
    async def check():
        client = OllamaClient()
        chat_ok, chat_detail = await client.model_status("qwen3:8b")
        embedding_ok, embedding_detail = await client.model_status("qwen3-embedding:0.6b")
        assert chat_ok, chat_detail
        assert embedding_ok, embedding_detail
        vector = await client.embed("demo seed")
        assert vector
    asyncio.run(check())
    assert SimilarityStore().available