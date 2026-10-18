"""
Central configuration for the AI Vehicle Manual Assistant backend.
All values can be overridden via environment variables or a .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App ---
    APP_NAME: str = "AI Vehicle Manual Assistant"
    ENV: str = "development"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # --- Storage ---
    MANUALS_DIR: str = "storage/manuals"
    CHROMA_DIR: str = "storage/chroma"
    SQLITE_PATH: str = "storage/app.db"

    # --- LLM / Embeddings provider ---
    # "openai" -> uses OpenAI API (needs OPENAI_API_KEY)
    # "ollama" -> uses a local Llama model served by Ollama (no key needed)
    LLM_PROVIDER: str = "ollama"

    OPENAI_API_KEY: str | None = None
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.1"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # --- RAG tuning ---
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150
    RETRIEVER_TOP_K: int = 5
    MAX_HISTORY_TURNS: int = 6  # how many previous Q/A turns to feed back to the LLM

    # --- Adaptive retrieval (ADAPT-Diag). Defaults are the dev-tuned values in eval/results/tuned.json.
    ADAPTIVE_RAG: bool = True
    # LLM router fallback + query rewriting. Off by default: the thresholds below were tuned without
    # it, and in eval/results the rewrite loop inflated sufficiency on unanswerable queries
    # (correct abstention 0.58 -> 0.11). Turn on only after re-tuning with it enabled.
    ADAPTIVE_USE_LLM: bool = False
    ADAPTIVE_ALPHA: float = 0.5
    ADAPTIVE_BETA: float = 0.5
    ADAPTIVE_GAMMA: float = 0.0
    ADAPTIVE_TAU: float = 0.986
    ADAPTIVE_TAU_M1: float | None = 0.562
    ADAPTIVE_T_MAX: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
