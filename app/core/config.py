from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Jarvis"
    APP_VERSION: str = "0.1.4"
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

    # jarvis-conversation-service — owns the conversations/subagent_traces
    # tables since the Chapter 2 decomposition; see app/clients/conversation_client.py.
    CONVERSATION_SERVICE_URL: str = "http://localhost:8001"
    # jarvis-file-service — owns the per-user folder/file workspace (tree
    # metadata + MinIO blobs + Qdrant embeddings); see app/clients/file_client.py.
    # Shares INTERNAL_API_KEY below with conversation-service.
    FILE_SERVICE_URL: str = "http://localhost:8002"
    INTERNAL_API_KEY: str = ""

    FRONTEND_URL: str = "http://localhost:3000"

    OIDC_ISSUER: str = "http://localhost:8180/realms/jarvis"
    OIDC_AUDIENCE: str = "jarvis-frontend"
    OIDC_JWKS_URL: str = "http://localhost:8180/realms/jarvis/protocol/openid-connect/certs"
    OIDC_JWKS_CACHE_TTL_SECONDS: int = 3600

    LLM_CACHE: bool = False

    # kubernetes-sigs/agent-sandbox — backs the bash tool + present_file, one
    # dedicated sandbox pod per conversation. See sandbox_manager.py and
    # SANDBOX-SETUP.md. The warm pool name must match a SandboxWarmPool that
    # exists in AGENTSANDBOX_NAMESPACE; `python-sandbox-pool` is the one
    # upstream's own quickstart YAML creates.
    AGENTSANDBOX_NAMESPACE: str = "default"
    AGENTSANDBOX_WARMPOOL: str = "python-sandbox-pool"
    # How long a conversation's sandbox may sit idle before the cluster stops
    # it. Behaves as an idle timeout rather than a fixed deadline because
    # sandbox_manager pushes the expiry out on every use. Expiry deletes the
    # Pod but keeps the volume, so the next message revives the same
    # filesystem in a few seconds — this is a pause, not a teardown, and the
    # value is a CPU/memory-reclaim knob, not a data-retention one.
    AGENTSANDBOX_TTL_SECONDS: int = 1_800
    # The second tier, and the only thing that ever reclaims disk. Expiry
    # above frees the pod but keeps the volume, so a conversation nobody ever
    # deletes would hold its 1Gi forever. This deadline is set on the *claim*,
    # where expiry cascades down and removes the volume too. Also pushed
    # forward on every use, so it measures abandonment rather than age.
    AGENTSANDBOX_MAX_IDLE_SECONDS: int = 604_800  # 7 days

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
