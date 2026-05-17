from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client
from src.core.config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using OpenAI in batches.

    Args:
        texts: A list of strings to embed.

    Returns:
        A list of embedding vectors.
    """
    embeddings: list[list[float]] = []
    if not texts:
        return embeddings

    openai_client = get_openai_client()
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = openai_client.embeddings.create(
            model=settings.embedding_model, input=batch, dimensions=1536
        )
        embeddings.extend([d.embedding for d in resp.data])
    return embeddings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def embed_query(query: str) -> list[float]:
    """Convert a user query into an embedding vector.

    Args:
        query: The user query string.

    Returns:
        A list of floats representing the embedding vector.
    """
    openai_client = get_openai_client()
    resp = openai_client.embeddings.create(
        model=settings.embedding_model, input=[query], dimensions=1536
    )
    return resp.data[0].embedding
