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

    # --- Tool-output offloading (ToolOutputOffloadMiddleware) ---
    # A long tool result (bash dump, fetched web page, big search payload)
    # bloats every subsequent model call. When one is longer than
    # TOOL_OFFLOAD_MAX_CHARS and is no longer among the KEEP_RECENT most recent
    # tool results, the model-facing copy is replaced with a short stub
    # (head+tail preview + a ref) and the full text is stashed in the LangGraph
    # store; the agent can pull it back with the `recall_tool_output` tool.
    # The persisted checkpoint keeps every tool result in full — the UI/history
    # is untouched. Set ENABLED=False to turn the whole thing off.
    TOOL_OFFLOAD_ENABLED: bool = True
    TOOL_OFFLOAD_MAX_CHARS: int = 8000        # ~2k tokens
    TOOL_OFFLOAD_KEEP_RECENT: int = 2         # newest N tool results are never stubbed
    TOOL_OFFLOAD_PREVIEW_HEAD: int = 700
    TOOL_OFFLOAD_PREVIEW_TAIL: int = 300

    # jarvis-sandbox — one shared container backing the bash tool + present_file.
    # Replaced OpenSandbox (its bwrap/userns isolation broke on this host).
    SANDBOX_SERVICE_URL: str = "http://localhost:8003"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
