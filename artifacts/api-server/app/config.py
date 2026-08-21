from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    port: int = int(os.getenv("PORT", "8080"))
    database_path: str = os.getenv("DATABASE_PATH", "./data/bharatphish_guard.sqlite3")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/scam_examples.index")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    hinspam_model: str = os.getenv("HINSPAM_MODEL", "Keshav0av/HinSpam-spam-detection")
    silverguard_model: str = os.getenv("SILVERGUARD_MODEL", "tanishqmudaliar/SilverGuard")
    model_load_on_startup: bool = os.getenv("MODEL_LOAD_ON_STARTUP", "false").lower() == "true"
    url_reputation_url: str = os.getenv("URL_REPUTATION_URL", "")
    history_limit: int = int(os.getenv("ANALYSIS_HISTORY_LIMIT", "100"))
    low_threshold: int = int(os.getenv("RISK_LOW_THRESHOLD", "25"))
    suspicious_threshold: int = int(os.getenv("RISK_SUSPICIOUS_THRESHOLD", "50"))
    high_risk_threshold: int = int(os.getenv("RISK_HIGH_RISK_THRESHOLD", "75"))


settings = Settings()