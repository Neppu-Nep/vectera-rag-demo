from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils.embeddings import embed_query, embed_texts

logger = get_logger(__name__)

DocType = Literal[
    "Financial Report", "Earnings Call", "Press Release", "Presentation", "Other"
]


class ChunkScore(BaseModel):
    id: str
    relevance_score: int
    is_relevant: bool


class RerankResponse(BaseModel):
    chunk_scores: list[ChunkScore]


class QueryFilters(BaseModel):
    is_comparison: bool = False
    expanded_queries: list[str] = Field(default_factory=list)
    companies: list[str] | None = None
    years: list[int] | None = None
    quarters: list[str] | None = None
    document_types: list[DocType] | None = None
    search_keywords: str | None = None


class QueryExecutionContext(BaseModel):
    client_id: str
    queries_to_embed: list[str]
    embeddings: list[list[float]]
    match_threshold: float
    match_count: int
    years: list[int] | None = None
    quarters: list[str] | None = None
    companies: list[str] | None = None
    document_types: list[DocType] | None = None


def _get_context_text(chunk: dict[str, Any]) -> str:
    raw_text = chunk.get("raw_content")
    if raw_text:
        return str(raw_text)
    return str(chunk.get("chunk_text", ""))


def extract_query_filters(user_query: str) -> QueryFilters:
    """Extract routing filters and expanded queries from user input."""
    if not user_query.strip():
        return QueryFilters(is_comparison=False, expanded_queries=[])

    prompt = (
        "You are an elite financial search router. Extract search parameters into strict JSON.\n\n"
        "RULES:\n"
        "1. is_comparison: Set to true ONLY if the user is explicitly comparing two time periods, companies, or versions (e.g., 'vs', 'compare', 'difference').\n"
        "2. expanded_queries: Generate 3 variations of the core search intent using financial synonyms to maximize vector retrieval recall.\n"
        "3. companies: Extract canonical company names only if explicitly asked.\n"
        "4. years & quarters: Extract ONLY if filtering for a specific 'As-Of' document period.\n"
        "5. document_types: Map to exactly ['Financial Report', 'Earnings Call', 'Press Release', 'Presentation', 'Other'] ONLY if the user explicitly requests a specific format (e.g., 'in the deck', 'in the 10-K'). If they use generic terms like 'report', 'document', or 'file', return null.\n"
        "6. search_keywords: A clean string of nouns for full-text database search.\n\n"
        "EXAMPLES:\n"
        "User: 'Compare Q3 to Q4 revenue for Company A.'\n"
        'Output: {"is_comparison": true, "expanded_queries":["Q3 vs Q4 revenue", "topline income change", "financial performance quarter over quarter"], "companies": ["Company A"], "years": null, "quarters": ["Q3", "Q4"], "document_types": null, "search_keywords": "revenue"}\n\n'
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

    logger.warning("Router LLM failed. Falling back to default naive routing.")
    return QueryFilters(
        is_comparison=False, expanded_queries=[user_query], search_keywords=user_query
    )


def resolve_entities(raw_companies: list[str]) -> list[str]:
    """Resolve raw company names to canonical names via DB alias matching."""
    if not raw_companies:
        return []

    supabase = get_supabase_client()
    resolved: set[str] = set()

    try:
        embs = embed_texts(raw_companies)
    except Exception:
        embs = [embed_query(c) for c in raw_companies]

    for raw_name, emb in zip(raw_companies, embs, strict=True):
        try:
            res = supabase.rpc(
                "resolve_company_aliases",
                {
                    "raw_query": raw_name,
                    "query_emb": emb,
                    "trgm_threshold": 0.7,
                },
            ).execute()

            if res.data and isinstance(res.data, list) and len(res.data) > 0:
                for row in res.data:
                    resolved.add(row["canonical"])  # type: ignore
            else:
                resolved.add(raw_name.strip().title())
        except Exception as e:
            logger.error(
                f"Entity resolution failed for '{raw_name}': {e}", exc_info=True
            )
            resolved.add(raw_name.strip().title())

    mapped = list(resolved)
    logger.info(f"Resolved raw entities {raw_companies} -> Canonical: {mapped}")
    return mapped


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
    cleaned_document_types = [
        str(d).strip() for d in filter_document_types or [] if d and d.strip()
    ]

    payload = {
        "query_embedding": query_embedding,
        "user_query": keyword_query or None,
        "match_threshold": match_threshold,
        "match_count": match_count,
        "filter_client_id": client_id,
        "filter_years": filter_years or None,
        "filter_quarters": cleaned_quarters or None,
        "filter_companies": cleaned_companies or None,
        "filter_document_types": cleaned_document_types or None,
    }

    logger.debug(f"Calling Supabase RPC match_documents with payload: {payload}")
    res = supabase.rpc("match_documents", payload).execute()

    return res.data if isinstance(res.data, list) else []  # type: ignore


def _fetch_candidates_from_db(ctx: QueryExecutionContext) -> list[dict[str, Any]]:
    cands = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for q_text, q_emb in zip(ctx.queries_to_embed, ctx.embeddings, strict=True):
            futures.append(
                executor.submit(
                    call_match_documents,
                    query_embedding=q_emb,
                    user_query=q_text,
                    client_id=ctx.client_id,
                    match_threshold=ctx.match_threshold,
                    match_count=ctx.match_count,
                    filter_years=ctx.years,
                    filter_quarters=ctx.quarters,
                    filter_companies=ctx.companies,
                    filter_document_types=ctx.document_types,
                )
            )
        for future in as_completed(futures):
            try:
                cands.extend(future.result())
            except Exception as e:
                logger.error(f"DB Fetch thread failed: {e}")
    return cands


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def rerank_with_llm(
    user_query: str, candidates: list[dict[str, Any]], top_k: int | None = None
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
        "You are an elite relevance scoring engine (Cross-Encoder). "
        "Calculate the strict mathematical relevance of each chunk to the user's query.\n"
        "RULES:\n"
        "1. Score each chunk from 0 to 10 (10 = perfect exact match, 0 = irrelevant noise).\n"
        "2. Set is_relevant to false if the score is below 4.\n"
        "3. Output strict JSON containing an array of chunk_scores.\n\n"
    )

    content = f"{prompt}User Query:\n{user_query}\n\nChunks:\n"
    for c in candidates:
        cid = c.get("id")
        comp = c.get("company_name", "Unknown")
        doc_type = c.get("document_type", "Unknown")
        text = _get_context_text(c)[:400]
        content += f"ID: {cid} | {comp} | {doc_type}\nExcerpt: {text}...\n---\n"

        page_number = c.get("page_number", 0)
        document_name = c.get("document_name", "Unknown")
        logger.info(f"Document Name: {document_name} | Page Number: {page_number}")

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": content}],
            text_format=RerankResponse,
        )
        if resp.output_parsed:
            scored = [
                cs
                for cs in resp.output_parsed.chunk_scores
                if cs.is_relevant and cs.id is not None
            ]

            if scored:
                scored.sort(key=lambda x: x.relevance_score, reverse=True)
                logger.info(
                    f"Reranker approved {len(scored)}/{len(candidates)} chunks."
                )
                return [str(cs.id) for cs in scored][:k]
            else:
                logger.warning(
                    "Reranker did not find any relevant chunks above the threshold. Falling back to RRF."
                )
    except Exception as e:
        logger.error(f"Unexpected error during reranking: {e}", exc_info=True)

    # Fallback to pure similarity based ranking if LLM call fails
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (
            -float(x.get("rrf_score") or 0.0),
            -float(x.get("similarity") or 0.0),
        ),
    )
    return [str(c.get("id")) for c in sorted_candidates if c.get("id") is not None][:k]


def retrieve_context(
    user_query: str, client_id: str, rerank_with_model: bool = True
) -> tuple[list[dict[str, Any]], QueryFilters]:
    """Retrieve the most relevant context to a query using DB search and reranking.

    Args:
        user_query: The string to search for.
        client_id: The client identifier.
        rerank_with_model: Whether to pass the initial DB hits through LLM relevance check.

    Returns:
        A tuple of chunks and extracted query filters.
    """

    filters = extract_query_filters(user_query)

    if filters.companies:
        filters.companies = resolve_entities(filters.companies)

    raw_queries = [user_query] + (filters.expanded_queries or [])
    queries_to_embed = list(dict.fromkeys(raw_queries))

    try:
        embeddings = embed_texts(queries_to_embed)
    except Exception:
        logger.warning(
            "Batch embedding failed, falling back to single-query embeddings."
        )
        embeddings = [embed_query(q) for q in queries_to_embed]

    ctx = QueryExecutionContext(
        client_id=client_id,
        queries_to_embed=queries_to_embed,
        embeddings=embeddings,
        match_threshold=0.0 if filters.is_comparison else 0.2,
        match_count=settings.top_k * 2,
        years=filters.years,
        quarters=filters.quarters,
        companies=filters.companies,
        document_types=filters.document_types,
    )

    all_candidates = _fetch_candidates_from_db(ctx)

    if not all_candidates:
        logger.warning(
            "Strict filtering yielded 0 results. Initiating Fallback: dropping temporal and format constraints."
        )
        ctx.years = None
        ctx.quarters = None
        ctx.document_types = None
        all_candidates = _fetch_candidates_from_db(ctx)

    # Deduplicate chunks
    unique_candidates = {}
    for c in all_candidates:
        cid = str(c.get("id"))
        if cid not in unique_candidates:
            unique_candidates[cid] = c
        else:
            if float(c.get("rrf_score") or 0.0) > float(
                unique_candidates[cid].get("rrf_score") or 0.0
            ):
                unique_candidates[cid] = c

    candidates_list = list(unique_candidates.values())

    if not candidates_list:
        logger.info("No matching documents found in Supabase for the given queries.")
        return ([], filters)

    final_count = settings.top_k * 2 if filters.is_comparison else settings.top_k

    if rerank_with_model:
        logger.info(f"Reranking {len(candidates_list)} unique candidates via LLM...")
        ranked_ids = rerank_with_llm(user_query, candidates_list, top_k=final_count)
        id_set = set(str(i) for i in ranked_ids)
        filtered = [c for c in candidates_list if str(c.get("id")) in id_set]

        id_to_c = {str(c.get("id")): c for c in filtered}
        ordered_candidates = [id_to_c[i] for i in ranked_ids if i in id_to_c]
    else:
        ordered_candidates = sorted(
            candidates_list,
            key=lambda x: (
                -float(x.get("rrf_score") or 0.0),
                -float(x.get("similarity") or 0.0),
            ),
        )[:final_count]

    # Final sort: Place the most recent documents at the top of the context window
    ordered_candidates.sort(
        key=lambda x: str(x.get('as_of_date') or '1970-01-01'), 
        reverse=True
    )

    for c in ordered_candidates:
        comp = c.get("company_name", "Unknown")
        doc_type = c.get("document_type", "Unknown")
        yr = c.get("report_year")
        qtr = c.get("report_quarter")
        ver = c.get("document_version", "Unknown")
        page_num = c.get("page_number", "N/A")

        context_text = _get_context_text(c)
        period_parts: list[str] = []
        if qtr:
            period_parts.append(str(qtr).strip())
        if yr not in (None, "N/A"):
            period_parts.append(str(yr).strip())
        period = " ".join(p for p in period_parts if p) or "Unknown Period"

        c["raw_content"] = (
            f"[Source: {comp} | Type: {doc_type} | As-Of Period: {period} | "
            f"Deck Version: {ver} | Page: {page_num}]\n"
            f"{context_text}"
        )

    return (ordered_candidates, filters)
