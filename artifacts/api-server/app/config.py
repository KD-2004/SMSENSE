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


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _resolve_setting_path(name: str, default: str) -> str:
    """Anchor relative file paths to the backend, not the caller's working directory."""
    value = Path(os.getenv(name, default)).expanduser()
    if not value.is_absolute():
        value = BACKEND_ROOT / value
    return str(value.resolve())


@dataclass(frozen=True)
class Settings:
    port: int = int(os.getenv("PORT", "4444"))
    database_path: str = _resolve_setting_path("DATABASE_PATH", "./data/smsense.sqlite3")
    faiss_index_path: str = _resolve_setting_path("FAISS_INDEX_PATH", "./data/scam_examples.index")
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
    # Optional Google Safe Browsing Lookup API key. Blank disables external lookups.
    google_safe_browsing_api_key: str = os.getenv("GOOGLE_SAFE_BROWSING_API_KEY", "").strip()
    # Comma-separated browser origins allowed to call this local development API.
    cors_allowed_origins: tuple[str, ...] = tuple(
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://127.0.0.1:5500,http://localhost:5500",
        ).split(",")
        if origin.strip()
    )
    history_limit: int = int(os.getenv("ANALYSIS_HISTORY_LIMIT", "100"))
    low_threshold: int = int(os.getenv("RISK_LOW_THRESHOLD", "25"))
    suspicious_threshold: int = int(os.getenv("RISK_SUSPICIOUS_THRESHOLD", "50"))
    high_risk_threshold: int = int(os.getenv("RISK_HIGH_RISK_THRESHOLD", "75"))
    # Optional enrichment must not keep the primary model result waiting for minutes.
    similarity_timeout_seconds: float = max(
        0.1, float(os.getenv("SMSENSE_SIMILARITY_TIMEOUT_SECONDS", "3.0"))
    )
    url_reputation_timeout_seconds: float = max(
        0.1, float(os.getenv("SMSENSE_URL_REPUTATION_TIMEOUT_SECONDS", "4.0"))
    )
    classifier_timeout_seconds: float = max(
        1.0, float(os.getenv("SMSENSE_CLASSIFIER_TIMEOUT_SECONDS", "300.0"))
    )


settings = Settings()