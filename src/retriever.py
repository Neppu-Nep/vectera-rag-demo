from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, final

from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils.embeddings import embed_query

logger = get_logger(__name__)

DocType = Literal["Financial Report", "Earnings Call", "Press Release", "Presentation", "Other"]

class RerankResponse(BaseModel):
    """Schema for the LLM reranking output."""

    ids: list[str]


class QueryFilters(BaseModel):
    """Schema for extracted query filters."""

    companies: list[str] | None = None
    years: list[int] | None = None
    quarters: list[str] | None = None
    document_types: list[DocType] | None = None
    search_keywords: str | None = None


def _get_context_text(chunk: dict[str, Any]) -> str:
    raw_text = chunk.get("raw_content")
    if raw_text:
        return str(raw_text)
    return str(chunk.get("chunk_text", ""))


def extract_query_filters(user_query: str) -> QueryFilters:
    """Extract filterable metadata from the user query using an LLM."""
    if not user_query.strip():
        return QueryFilters()

    prompt = (
        "Extract JSON with keys companies, years, quarters, document_types, search_keywords. Return null for missing keys.\n"
        "RULE FOR COMPANIES: ONLY extract company names if the user is explicitly asking about a specific company.\n"
        "RULE FOR YEARS: ONLY extract a year if the user is explicitly asking for a document published in that year (e.g., 'In the 2025 report...'). "
        "DO NOT extract a year if the user is asking about a future projection or historical metric (e.g., 'What will happen in 2027?'). In that case, return null for years.\n"
        "RULE FOR QUARTERS: Only return normalized values Q1/Q2/Q3/Q4 when explicitly referenced.\n"
        "RULE FOR DOCUMENT_TYPES: Map user terms to EXACTLY one of ['Financial Report', 'Earnings Call', 'Press Release', 'Presentation', 'Other']. "
        "(e.g., 'slide deck', 'investor day', 'merger deck' = 'Presentation'. '10-Q', '10-K' = 'Financial Report').\n"
        "RULE FOR SEARCH_KEYWORDS: a clean string of the core nouns and entities optimized for a database full-text search, stripping out conversational words like 'compare', 'what is', or 'summarize'. "
    )

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": f"{prompt}\nQuery:\n{user_query}"}],
            text_format=QueryFilters,
        )
        if resp.output_parsed:
            logger.info(f"Extracted query filters: {resp.output_parsed}")
            return resp.output_parsed
    except Exception as e:
        logger.error(f"Failed to extract query filters: {e}", exc_info=True)

    return QueryFilters()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_match_documents(
    query_embedding: list[float],
    user_query: str,
    client_id: str,
    match_threshold: float = 0.3,
    match_count: int = 10,
    filter_years: list[int] | None = None,
    filter_quarters: list[str] | None = None,
    filter_companies: list[str] | None = None,
    filter_document_types: list[DocType] | None = None,
) -> list[dict[str, Any]]:
    """Query the Supabase database for chunks similar to the query embedding.

    Args:
        query_embedding: The vectorized user query.
        user_query: The raw user query for keyword search.
        client_id: The client identifier for scoping data.
        match_threshold: Minimum cosine similarity threshold.
        match_count: Max number of returned results.
        filter_years: Optional report year filters.
        filter_quarters: Optional report quarter filters.
        filter_companies: Optional company name filters.
        filter_document_types: Optional document type filters.

    Returns:
        A list of dictionaries representing the DB rows.
    """
    supabase = get_supabase_client()
    keyword_query = user_query.strip()

    cleaned_quarters = []
    for q in filter_quarters or []:
        q_norm = str(q).strip().upper()
        if q_norm in {"Q1", "Q2", "Q3", "Q4"}:
            cleaned_quarters.append(q_norm)

    cleaned_companies = [c.strip() for c in filter_companies or [] if c and c.strip()]
    cleaned_document_types = [str(d).strip() for d in filter_document_types or [] if d and d.strip()]

    payload = {
        "query_embedding": query_embedding,
        "user_query": keyword_query or None,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "filter_client_id": client_id,
        "filter_years": filter_years,
        "filter_quarters": cleaned_quarters or None,
        "filter_companies": cleaned_companies or None,
        "filter_document_types": cleaned_document_types or None,
    }

    logger.debug(f"Calling Supabase RPC match_documents with payload: {payload}")
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
        cid = c.get("id")
        comp = c.get("company_name", "Unknown")
        yr = c.get("report_year", "N/A")
        qtr = c.get("report_quarter", "")
        doc_type = c.get("document_type", "Unknown")
        text = _get_context_text(c)[:400]
        
        period = f"{qtr} {yr}".strip() if yr != "N/A" else "Unknown Period"
        
        content += (
            f"ID: {cid} | {comp} | {doc_type} | Period: {period}\n"
            f"Doc: {c.get('document_name')}|{c.get('document_version')}\n"
            f"Excerpt: {text}...\n---\n"
        )

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": content}],
            text_format=RerankResponse,
        )
        if resp.output_parsed:
            return [str(i) for i in resp.output_parsed.ids if i is not None][:k]
    except Exception as e:
        logger.error(f"Unexpected error during reranking: {e}", exc_info=True)

    # Fallback to pure similarity based ranking if LLM call fails
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (-float(x.get("rrf_score") or 0.0), -float(x.get("similarity") or 0.0)),
    )
    return [str(c.get("id")) for c in sorted_candidates if c.get("id") is not None][:k]


def retrieve_context(
    user_query: str,
    client_id: str,
    rerank_with_model: bool = True,
) -> tuple[list[dict[str, Any]], QueryFilters]:
    """Retrieve the most relevant context to a query using DB search and reranking.

    Args:
        user_query: The string to search for.
        client_id: The client identifier.
        rerank_with_model: Whether to pass the initial DB hits through LLM relevance check.

    Returns:
        A tuple of chunks and extracted query filters.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        filters_future = executor.submit(extract_query_filters, user_query)
        embedding_future = executor.submit(embed_query, user_query)
        filters = filters_future.result()
        embedding = embedding_future.result()

    lower_query = user_query.lower()
    is_version_comparison = any(
        kw in lower_query
        for kw in [
            "vs",
            "version",
            "compare",
            "contrast",
            "difference",
            "two reports",
            "change",
            "between",
            "versus",
        ]
    )

    match_threshold = 0.2
    final_count = settings.top_k
    db_fetch_count = final_count

    if is_version_comparison:
        match_threshold = 0.0
        final_count = settings.top_k * 2  # Return more results for version comparisons since they are harder
        db_fetch_count = final_count
        filters.document_types = None # Loosen document type filter for version comparisons since users may not specify and we want to capture all relevant versions

        if rerank_with_llm:
            db_fetch_count = final_count * 3  # Fetch even more from DB for version comparisons when using LLM reranking to give the model more options to choose from

    candidates = call_match_documents(
        embedding,
        filters.search_keywords or user_query,
        client_id,
        match_threshold=match_threshold,
        match_count=db_fetch_count,
        filter_years=filters.years,
        filter_quarters=filters.quarters,
        filter_companies=filters.companies,
        filter_document_types=filters.document_types,
    )

    if not candidates:
        logger.info("No matching documents found in Supabase for the given query.")
        return [], filters

    if rerank_with_model:
        logger.info(f"Reranking {len(candidates)} candidates via LLM...")
        try:
            # Tell the LLM to slice the massive pool down to the final_count
            ranked_ids = rerank_with_llm(user_query, candidates, top_k=final_count)
            id_set = set(str(i) for i in ranked_ids)
            filtered = [c for c in candidates if str(c.get("id")) in id_set]

            # Preserve order sorted by LLM preferences
            id_to_c = {str(c.get("id")): c for c in filtered}
            ordered = [id_to_c[i] for i in ranked_ids if i in id_to_c]
            
            if ordered:
                for c in ordered:
                    comp = c.get("company_name", "Unknown")
                    doc_type = c.get("document_type", "Unknown")
                    yr = c.get("report_year", "N/A")
                    qtr = c.get("report_quarter", "")
                    ver = c.get("document_version", "Unknown")
                    page_num = c.get("page_number", "N/A")
                    
                    context_text = _get_context_text(c)
                    period = f"{qtr} {yr}".strip() if yr != "N/A" else "Unknown Period"
                    
                    c["raw_content"] = (
                        f"[Source: {comp} | Type: {doc_type} | As-Of Period: {period} | "
                        f"Deck Version: {ver} | Page: {page_num}]\n"
                        f"{context_text}"
                    )
                return ordered, filters
        except Exception as e:
            logger.error(f"Reranking pipeline crashed: {e}", exc_info=True)

    # If rerank_with_model=False or ordering failed, fallback
    fallback = sorted(
        candidates,
        key=lambda x: (-float(x.get("rrf_score") or 0.0), -float(x.get("similarity") or 0.0)),
    )[:final_count]
    
    for c in fallback:
        comp = c.get("company_name", "Unknown")
        doc_type = c.get("document_type", "Unknown")
        yr = c.get("report_year", "N/A")
        qtr = c.get("report_quarter", "")
        ver = c.get("document_version", "Unknown")
        page_num = c.get("page_number", "N/A")
        
        context_text = _get_context_text(c)
        period = f"{qtr} {yr}".strip() if yr != "N/A" else "Unknown Period"
        
        c["raw_content"] = (
            f"[Source: {comp} | Type: {doc_type} | As-Of Period: {period} | "
            f"Deck Version: {ver} | Page: {page_num}]\n"
            f"{context_text}"
        )
        
    return fallback, filters