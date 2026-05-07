from typing import Any

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils.embeddings import embed_query

logger = get_logger(__name__)


class RerankResponse(BaseModel):
    """Schema for the LLM reranking output."""

    ids: list[str]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_match_documents(
    query_embedding: list[float],
    client_id: str,
    match_threshold: float = 0.3,
    match_count: int = 10,
) -> list[dict[str, Any]]:
    """Query the Supabase database for chunks similar to the query embedding.

    Args:
        query_embedding: The vectorized user query.
        client_id: The client identifier for scoping data.
        match_threshold: Minimum cosine similarity threshold.
        match_count: Max number of returned results.

    Returns:
        A list of dictionaries representing the DB rows.
    """
    supabase = get_supabase_client()
    payload = {
        "query_embedding": query_embedding,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "filter_client_id": client_id,
    }

    logger.debug(
        f"Calling Supabase RPC match_documents with threshold {match_threshold}"
    )
    res = supabase.rpc("match_documents", payload).execute()

    return res.data if isinstance(res.data, list) else []  # type: ignore


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def rerank_with_llm(
    user_query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[str]:
    """Use an LLM to re-evaluate retrieval results beyond simple vector space.

    Args:
        user_query: The user's original raw text query.
        candidates: The top matching chunks array from vector search.
        top_k: The number of best chunks to filter down to. Defaults to settings.top_k.

    Returns:
        A list of the chunk `id` strings ordered by predicted relevance.
    """
    k = top_k or settings.top_k
    prompt = (
        "You are a relevance judge. Given the user query and a list of document chunks (each with id and text), "
        f"return a JSON object with a single key \"ids\" containing an array of the top {k} chunk ids "
        "that best answer the query, ordered most-to-least relevant.\n\n"
    )

    content = prompt + "User Query:\n" + user_query + "\n\nChunks:\n"
    for c in candidates:
        preview = c.get("chunk_text", "").replace("\n", " ")
        content += (
            f"ID: {c.get('id')}, "
            f"Doc: {c.get('document_name')}|{c.get('document_version')} "
            f"Text: {preview}\n\n"
        )

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": "low"},
            text_format=RerankResponse,
        )
        if resp.output_parsed:
            return [str(i) for i in resp.output_parsed.ids if i is not None][:k]
    except Exception as e:
        logger.error(f"Unexpected error during reranking: {e}", exc_info=True)

    # Fallback to pure similarity based ranking if LLM call fails
    sorted_candidates = sorted(
        candidates,
        key=lambda x: -float(x.get("similarity", 0)),
    )
    return [str(c.get("id")) for c in sorted_candidates if c.get("id") is not None][:k]


def retrieve_context(
    user_query: str,
    client_id: str,
    rerank_with_model: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant context to a query using DB search and reranking.

    Args:
        user_query: The string to search for.
        client_id: The client identifier.
        rerank_with_model: Whether to pass the initial DB hits through LLM relevance check.

    Returns:
        A list of chunks (dicts) matched for generation.
    """
    embedding = embed_query(user_query)
    lower_query = user_query.lower()
    is_version_comparison = any(
        kw in lower_query
        for kw in [
            "vs",
            "version",
            "compare",
            "difference",
            "two reports",
            "change",
            "between",
        ]
    )

    match_threshold = 0.0 if is_version_comparison else 0.2
    match_count = settings.top_k * 2 if is_version_comparison else settings.top_k

    logger.debug(
        "Retrieving context for query: "
        f"'{user_query}' (Version comparison: {is_version_comparison})"
    )
    candidates = call_match_documents(
        embedding,
        client_id,
        match_threshold=match_threshold,
        match_count=match_count,
    )

    if is_version_comparison and candidates:
        # Double check if the query explicitly mentions company names to further filter down to relevant documents
        mentioned_companies = {
            str(c.get("company_name", "")).strip()
            for c in candidates
            if str(c.get("company_name", "")).strip()
            and str(c.get("company_name", "")).strip().lower() in lower_query
        }

        if mentioned_companies:
            candidates = [
                c
                for c in candidates
                if str(c.get("company_name", "")).strip() in mentioned_companies
            ]
        else:
            ranked = sorted(candidates, key=lambda x: -float(x.get("similarity", 0)))
            primary_company = str(ranked[0].get("company_name", "")).strip()
            if primary_company:
                candidates = [
                    c
                    for c in ranked
                    if str(c.get("company_name", "")).strip() == primary_company
                ]

    if not candidates:
        logger.info("No matching documents found in Supabase for the given query.")
        return []

    if rerank_with_model:
        logger.info(f"Reranking {len(candidates)} candidates via LLM...")
        try:
            ranked_ids = rerank_with_llm(user_query, candidates)
            id_set = set(str(i) for i in ranked_ids)
            filtered = [c for c in candidates if str(c.get("id")) in id_set]

            # Preserve order sorted by LLM preferences
            id_to_c = {str(c.get("id")): c for c in filtered}
            ordered = [id_to_c[i] for i in ranked_ids if i in id_to_c]
            if ordered:
                for c in ordered:
                    comp = c.get("company_name", "Unknown")
                    ver = c.get("document_version", "Unknown")
                    c["chunk_text"] = (
                        f"[Company: {comp}, Version: {ver}] \n "
                        f"{c.get('chunk_text', '')}"
                    )
                return ordered
        except Exception as e:
            logger.error(f"Reranking pipeline crashed: {e}", exc_info=True)

    # If rerank_with_model=False or ordering failed, fallback
    fallback = sorted(
        candidates,
        key=lambda x: -float(x.get("similarity", 0)),
    )[: settings.top_k]
    for c in fallback:
        comp = c.get("company_name", "Unknown")
        ver = c.get("document_version", "Unknown")
        c["chunk_text"] = f"[Company: {comp}, Version: {ver}] \n {c.get('chunk_text', '')}"
    return fallback
