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

    # Langfuse — the self-hosted trace store in jarvis-deploy/langfuse/.
    # Empty host disables tracing entirely (see app/core/observability.py),
    # which is what local runs and CI get by default.
    LANGFUSE_HOST: str = ""
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    # Stamped on every trace so one Langfuse project can hold more than one
    # cluster without the dashboards averaging them together. Langfuse
    # restricts this to lowercase letters, digits, "-" and "_".
    LANGFUSE_ENVIRONMENT: str = "development"

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

    # Background research: start_research/check_research/cancel_research/
    # list_research, which run the research subagent as a task that outlives the
    # chat turn (app/agents/async_research.py). False leaves the tools out.
    BACKGROUND_RESEARCH: bool = True
    # How many background research tasks may run at once in this process. Each
    # is a full search/fetch loop of LLM calls, so this bounds what one user
    # asking for ten can cost everyone else.
    BACKGROUND_RESEARCH_MAX_RUNNING: int = 4

    # Agentic retrieval for search_files — decompose, rerank, and retry once
    # when the first pass misses. See app/agents/retrieval.py. Costs two LLM
    # calls on the subagent model per search, three when a retry fires, so
    # it's a quality-for-latency trade; turn it off to fall back to one-shot
    # vector search.
    AGENTIC_RAG_ENABLED: bool = True
    # The helper model for decomposition and reflection. Separate from the
    # agent's own model on purpose: that one defaults to a reasoning model at
    # high effort, which made each of these short calls take ~20s. Pick
    # something fast and non-reasoning — these are classification tasks.
    RETRIEVAL_MODEL: str = "deepseek/deepseek-v4-flash"

    # Skills (app/agents/skills.py): SKILL.md instructions loaded on demand.
    # Empty SKILLS_DIR means the app/skills directory shipped with the image.
    SKILLS_ENABLED: bool = True
    SKILLS_DIR: str = ""
    # Deferred tools (app/agents/tool_search.py): comma-separated tool names
    # whose schemas are only sent once the model asks for them with
    # tool_search. Chosen for being large and rarely needed.
    # ocr_image, draw_boxes and view_image are deliberately NOT deferred: a model asked to "draw boxes" on an
    # image did not think of loading them, and drew from coordinates it guessed or skipped the
    # boxes altogether. Offered in every request, their descriptions are in front of it. view_image
    # joined them after a test where, asked what an attached photo showed, the model answered from
    # ocr_image's text alone (it drops diacritics and digits) instead of loading view_image first.
    DEFERRED_TOOLS: str = "generate_visualization_svg,cancel_research,list_research,check_pdf,read_pdf,office_convert,office_recalculate"
    # MCP (app/agents/mcp_tools.py): a JSON object, or a path to a file
    # holding one, naming the servers to load tools from. Empty = none.
    MCP_SERVERS: str = ""
    # Defer every MCP tool, so a server with many tools costs nothing until
    # one is wanted.
    MCP_DEFER: bool = True

    # MCP servers users add themselves from the chat UI (app/agents/user_mcp.py).
    MCP_USER_ENABLED: bool = True
    MCP_USER_MAX_SERVERS: int = 10
    # Fernet key that encrypts the headers (tokens) users give their servers.
    # Unset: servers can still be added, but not with headers.
    #   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    MCP_SECRET_KEY: str = ""
    # Comma-separated hosts exempt from the public-address rule in
    # app/agents/mcp_guard.py — e.g. an MCP server run inside the cluster.
    MCP_USER_ALLOWED_HOSTS: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
