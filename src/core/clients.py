from functools import lru_cache

from openai import OpenAI
from supabase import Client, create_client

from src.core.config import settings


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Initialize and retrieve a cached Supabase client."""
    return create_client(settings.supabase_url, settings.supabase_service_key)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """Initialize and retrieve a cached OpenAI client."""
    return OpenAI(api_key=settings.openai_api_key)
