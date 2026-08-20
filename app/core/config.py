from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.0"
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

    FRONTEND_URL: str = "http://localhost:3000"

    OIDC_ISSUER: str = "http://localhost:8180/realms/jarvis"
    OIDC_AUDIENCE: str = "jarvis-frontend"
    OIDC_JWKS_URL: str = "http://localhost:8180/realms/jarvis/protocol/openid-connect/certs"
    OIDC_JWKS_CACHE_TTL_SECONDS: int = 3600

    LLM_CACHE: bool = False

    # OpenSandbox server (see backend/Makefile's sandbox-server target) —
    # backs the bash tool's per-conversation sandboxes.
    SANDBOX_SERVER_DOMAIN: str = "localhost:8080"
    SANDBOX_API_KEY: str = ""
    SANDBOX_IMAGE: str = "jarvis-sandbox"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
