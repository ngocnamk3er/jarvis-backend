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

    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "jarvis"
    MINIO_SECRET_KEY: str = "jarvis123"
    MINIO_BUCKET: str = "jarvis-files"
    MINIO_SECURE: bool = False
    MINIO_PUBLIC_URL: str = "http://localhost:9000"

    DATABASE_URL: str = "postgresql://jarvis:jarvis@localhost:5433/jarvis"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
