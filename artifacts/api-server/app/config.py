from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Allows source inspection before optional deps are installed.
    load_dotenv = None

if load_dotenv is not None:
    # Load the backend's local .env regardless of the process working directory.
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    port: int = int(os.getenv("PORT", "4444"))
    database_path: str = os.getenv("DATABASE_PATH", "./data/smsense.sqlite3")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/scam_examples.index")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    sms_shield_model: str = os.getenv(
        "SMSENSE_CLASSIFIER_MODEL",
        "hf.co/ganmoor-ai-labs/sms-shield:Q4_K_M",
    )
    indicbert_model: str = os.getenv("INDICBERT_MODEL", "anmolshrivastav/indicbert-scam-classifier-v2")
    indicbert_enabled: bool = os.getenv("INDICBERT_ENABLED", "false").lower() == "true"
    model_load_on_startup: bool = os.getenv("MODEL_LOAD_ON_STARTUP", "false").lower() == "true"
    url_reputation_url: str = os.getenv("URL_REPUTATION_URL", "")
    history_limit: int = int(os.getenv("ANALYSIS_HISTORY_LIMIT", "100"))
    low_threshold: int = int(os.getenv("RISK_LOW_THRESHOLD", "25"))
    suspicious_threshold: int = int(os.getenv("RISK_SUSPICIOUS_THRESHOLD", "50"))
    high_risk_threshold: int = int(os.getenv("RISK_HIGH_RISK_THRESHOLD", "75"))


settings = Settings()