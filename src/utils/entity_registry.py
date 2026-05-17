import serpapi
from typing import Any

from pydantic import BaseModel

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils.embeddings import embed_texts

logger = get_logger(__name__)


class CompanyProfile(BaseModel):
    is_valid_company: bool
    canonical_name: str
    aliases: list[str]
    description: str


def fetch_search_context(company_name: str) -> str:
    """Fetch external search context for a company name.

    Args:
        company_name: Company name to query.

    Returns:
        Concise text context from SerpApi.
    """
    search_query = f'"{company_name}" company profile OR stock ticker OR headquarters'

    try:
        client = serpapi.Client(api_key=settings.serpapi_api_key)
        data = client.search(
            {
                "engine": "google",
                "q": search_query,
                "hl": "en",
                "gl": "us",
            }
        )

        snippets = []

        if "knowledge_graph" in data:
            kg = data["knowledge_graph"]
            kg_parts = []
            if "title" in kg:
                kg_parts.append(f"Name: {kg['title']}")
            if "entity_type" in kg:
                kg_parts.append(f"Type: {kg['entity_type']}")
            if "headquarters" in kg:
                kg_parts.append(f"HQ: {kg['headquarters']}")
            if "description" in kg:
                kg_parts.append(f"About: {kg['description']}")

            if kg_parts:
                snippets.append(f"[KNOWLEDGE GRAPH]: {' | '.join(kg_parts)}")

        organic_results = data.get("organic_results", [])

        for res in organic_results[:5]:
            title = res.get("title", "")
            snippet = res.get("snippet", "")
            snippets.append(f"[{title}]: {snippet}")

        context = "\n".join(snippets)
        return context if context else "No search context found."

    except Exception as e:
        logger.warning(f"SerpApi request failed for '{company_name}': {e}")
        return "Search failed. Rely on internal knowledge."


def register_discovered_companies(company_names: set[str]) -> None:
    """Validate and upsert discovered company names into the registry.

    Args:
        company_names: Set of company names to validate and register.
    """
    if not company_names:
        return

    supabase = get_supabase_client()

    existing_res = (
        supabase.table("companies")
        .select("canonical_name")
        .in_(
            "canonical_name",
            list(company_names),
        )
        .execute()
    )
    existing_names = {row["canonical_name"] for row in (existing_res.data or [])}  # type: ignore

    new_companies = company_names - existing_names
    if not new_companies:
        logger.debug(
            "All mentioned companies are already registered in the canonical dictionary."
        )
        return

    logger.info(
        f"Discovered {len(new_companies)} new entities. Validating via SerpApi..."
    )

    openai_client = get_openai_client()
    profiles_to_embed: list[CompanyProfile] = []

    for company in new_companies:
        search_context = fetch_search_context(company)

        prompt = f"""You are an elite financial data auditor.
Analyze the live search context for the extracted entity '{company}'.

RULES:
1. is_valid_company: Set to true ONLY if the entity is an actual corporate entity, business, or investment fund. Set to false if it is a financial metric (e.g., 'EBITDA', 'AFFO'), a geographic location, or hallucinated text.
2. canonical_name: The official corporate name derived from the search context. Do not use the raw string if the search reveals the true canonical name.
3. aliases: Provide stock tickers, common abbreviations, and former names found in the context.
4. description: A strict 1-sentence description of their core industry, business model, or sector.

SEARCH CONTEXT:
{search_context}
"""
        try:
            resp = openai_client.responses.parse(
                model=settings.reasoning_model,
                input=[{"role": "user", "content": prompt}],
                text_format=CompanyProfile,
            )
            parsed = resp.output_parsed
            if parsed:
                if parsed.is_valid_company:
                    profiles_to_embed.append(parsed)
                    logger.info(f"Validated entity: {parsed.canonical_name}")
                else:
                    logger.warning(
                        f"Entity rejected by search validation: '{company}' is not a valid company."
                    )
        except Exception as e:
            logger.error(f"Failed to generate profile for {company}: {e}")

    if not profiles_to_embed:
        return

    unique_profiles: dict[str, CompanyProfile] = {}
    for profile in profiles_to_embed:
        key = profile.canonical_name.strip()
        if key and key not in unique_profiles:
            unique_profiles[key] = profile

    if not unique_profiles:
        return

    texts_to_embed = [
        f"{p.canonical_name} {' '.join(p.aliases)}"
        for p in unique_profiles.values()
    ]
    embeddings = embed_texts(texts_to_embed)

    records: list[dict[str, Any]] = []
    for p, emb in zip(unique_profiles.values(), embeddings, strict=True):
        records.append(
            {
                "canonical_name": p.canonical_name,
                "aliases": p.aliases,
                "description": p.description,
                "embedding": emb,
            }
        )

    try:
        supabase.table("companies").upsert(
            records, on_conflict="canonical_name"
        ).execute()
        logger.info(
            f"Successfully registered {len(records)} verified companies to the master dictionary."
        )
    except Exception as e:
        logger.error(f"Failed to upsert new companies to dictionary: {e}")


def canonicalize_company_names(raw_names: list[str]) -> dict[str, list[str]]:
    """Resolve raw company names to canonical names via alias matching.

    Args:
        raw_names: Raw company names from document chunks.

    Returns:
        Mapping of raw name (casefolded) to canonical name list.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for name in raw_names:
        if not name:
            continue
        raw = str(name).strip()
        if not raw or raw.lower() == "unknown":
            continue
        key = raw.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(raw)

    if not cleaned:
        return {}

    supabase = get_supabase_client()

    mapping: dict[str, list[str]] = {}
    for raw_name in cleaned:
        canonical: list[str] = []
        try:
            res = supabase.rpc(
                "resolve_company_aliases_strict",
                {"raw_query": raw_name, "trgm_threshold": 0.7},
            ).execute()
            if res.data and isinstance(res.data, list):
                for row in res.data:
                    name = row.get("canonical") # type: ignore
                    if name:
                        canonical.append(name)
        except Exception as e:
            logger.error(f"Company alias resolve failed for '{raw_name}': {e}")

        if not canonical:
            canonical = [raw_name.strip().title()]

        mapping[raw_name.casefold()] = list(dict.fromkeys(canonical))

    return mapping
