from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "deepseek/deepseek-r1-0528-qwen3-8b:free"
    # Comma-separated model IDs tried, in order, if the requested model errors
    # (rate-limited, down, etc.) — see build_llm_with_fallback() in llm.py,
    # which chains these via LangChain's Runnable.with_fallbacks().
    OPENROUTER_FALLBACK_MODELS: str = "deepseek/deepseek-v4-flash,deepseek/deepseek-v4-pro"

    TAVILY_API_KEY: str = ""

    DATABASE_URL: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis"

    BACKEND_URL: str = "http://localhost:8000"

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "jarvis"
    MINIO_SECRET_KEY: str = "jarvis123"
    MINIO_BUCKET: str = "jarvis-files"
    MINIO_PUBLIC_URL: str = "http://localhost:9000"
    MINIO_SECURE: bool = False

    SANDBOX_DATA_DIR: str = str(Path(__file__).parent.parent.parent / "data" / "sandboxes")

    LLM_CACHE: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
