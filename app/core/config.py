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

    TAVILY_API_KEY: str = ""

    DATABASE_URL: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis"

    BACKEND_URL: str = "http://localhost:8000"

    SANDBOX_DATA_DIR: str = str(Path(__file__).parent.parent.parent / "data" / "sandboxes")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
