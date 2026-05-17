import os


def require_env(name: str) -> str:
    """Retrieve an environment variable or raise an error."""
    value = os.getenv(name)
    if not value:
        msg = f"Missing environment variable: {name}"
        raise ValueError(msg)
    return value


class Config:
    """Centralized configuration manager for the RAG project."""

    @property
    def openai_api_key(self) -> str:
        return require_env("OPENAI_API_KEY")

    @property
    def supabase_url(self) -> str:
        return require_env("SUPABASE_URL")

    @property
    def supabase_service_key(self) -> str:
        return require_env("SUPABASE_SERVICE_KEY")

    @property
    def reducto_api_key(self) -> str:
        return require_env("REDUCTO_API_KEY")

    @property
    def serpapi_api_key(self) -> str:
        return require_env("SERPAPI_API_KEY")

    @property
    def embedding_model(self) -> str:
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

    @property
    def reasoning_model(self) -> str:
        return os.getenv("REASONING_MODEL", "gpt-5.4-mini")

    @property
    def top_k(self) -> int:
        return int(os.getenv("TOP_K", "10"))


settings = Config()
