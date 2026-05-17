import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal, get_args

from langchain_text_splitters import MarkdownTextSplitter
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils import parser as document_parser
from src.utils.embeddings import embed_texts
from src.utils.entity_registry import (
    canonicalize_company_names,
    register_discovered_companies,
)

logger = get_logger(__name__)

SlideDocType = Literal["Presentation"]
PlainTextDocType = Literal["Financial Report", "Earnings Call", "Press Release"]

DocType = SlideDocType | PlainTextDocType | Literal["Other"]

SLIDE_DOC_TYPES = set(get_args(SlideDocType))


class MetadataResponse(BaseModel):
    """Schema for document metadata extraction."""

    chain_of_thought: str
    company_name: str
    document_version: str
    report_year: int | None = None
    report_quarter: str | None = None
    document_type: DocType = "Other"


class ChunkEnrichmentResponse(BaseModel):
    enriched_text: str
    mentioned_companies: list[str]


def check_duplicate(client_id: str, column: str, value: str) -> bool:
    """Check if a document with the given hash already exists.

    Args:
        client_id: The client identifier.
        column: The column to check (e.g. 'file_sha256' or 'text_sha256').
        value: The hash value to look for.

    Returns:
        True if a matching row exists for the client.
    """
    supabase = get_supabase_client()
    existing = (
        supabase.table("documents")
        .select("id")
        .eq("client_id", client_id)
        .eq(column, value)
        .limit(1)
        .execute()
    )
    return bool(existing.data)


def extract_metadata(pages: list[str], filename: str) -> dict[str, Any]:
    """Retrieve metadata (company name, version, year, quarter, type) via LLM.

    Args:
        pages: A list of parsed pages from the document.
        filename: The filename of the document for fallback parsing.

    Returns:
        A dictionary with document metadata fields.
    """
    total_pages = len(pages)
    if total_pages <= 15:
        document_chunk = "\n\n".join(
            [f"--- PAGE {i + 1} ---\n{page}" for i, page in enumerate(pages)]
        )
    else:
        head_pages = "\n\n".join(
            [f"--- PAGE {i + 1} ---\n{page}" for i, page in enumerate(pages[:10])]
        )
        tail_pages = "\n\n".join(
            [
                f"--- PAGE {total_pages - 4 + i} ---\n{page}"
                for i, page in enumerate(pages[-5:])
            ]
        )

        document_chunk = (
            head_pages
            + "\n\n...\n[TEXT TRUNCATED TO SAVE CONTEXT]\n...\n\n"
            + tail_pages
        )

    meta_prompt = (
        "You are an expert financial data extractor. Extract a JSON object with the following keys: "
        "chain_of_thought, company_name, document_type, report_year, report_quarter, document_version.\n\n"
        "RULES:\n"
        "1. chain_of_thought: Use this field to briefly think through the document type and 'As Of' dates based on the text. Always put this key first in the JSON.\n"
        "2. company_name: Extract the full, recognizable canonical company name. Do NOT use stock tickers or short marketing acronyms if the full name is known. Drop legal suffixes. (e.g., Use 'Apple' instead of 'Apple Inc.' or 'APPL'). If missing, return null.\n"
        "3. document_type: Classify as EXACTLY one of: 'Financial Report', 'Earnings Call', 'Press Release', 'Presentation', or 'Other'. Use 'Presentation' for slide decks, including investor days, roadshows, company updates, and merger presentations.\n"
        "4. report_year & report_quarter: This is the AS-OF financial reporting date of the document, NOT the publication date.\n"
        "   - Look for explicit text indicating the reporting period (e.g., 'Q3 2025 Results') OR footnotes stating when the data is valid (e.g., 'Balance sheet data as of December 31, 2025', 'metrics for the quarter ended 12/31'). Use this 'As Of' date to determine the quarter and year (e.g., Dec 31 = Q4).\n"
        "   - CRITICAL AS-OF RULE: Presentations often report on the previous quarter's financials. Even if a document is a general 'Roadshow', 'Company Update', or 'Merger Presentation' dated in March 2026, if its financial tables and metrics are explicitly 'As Of' Q4 2025, you MUST assign report_year: 2025 and report_quarter: 'Q4'. Ignore forward-looking projections.\n"
        "   - NULL QUARTER RULE: Only return null for report_quarter if the document contains absolutely no 'As-Of' financial dates or trailing quarter metrics.\n"
        "   - NULL YEAR RULE: Only return null for report_year if the document is purely thematic ('Other' type) with no specific financial period or event date. Otherwise, for general presentations without As-Of data, default to the year of the event/deck.\n"
        "5. document_version: Extract explicit version labels, publication dates, or revisions (e.g., 'March 2026', 'February 2026', 'v2', 'Final'). If missing, return null.\n\n"
        f"Filename: {filename}\n"
        f"Text:\n{document_chunk}"
    )

    openai_client = get_openai_client()
    company_name = "Unknown"
    document_version = "Unknown"
    report_year: int | None = None
    report_quarter: str | None = None
    document_type: DocType = "Other"

    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": meta_prompt}],
            text_format=MetadataResponse,
        )
        if resp.output_parsed:
            company_name = resp.output_parsed.company_name
            document_version = resp.output_parsed.document_version
            report_year = resp.output_parsed.report_year
            report_quarter = resp.output_parsed.report_quarter
            document_type = resp.output_parsed.document_type
            logger.info(
                f"Extracted metadata - Company: {company_name}, Version: {document_version}, Year: {report_year}, Quarter: {report_quarter}, Type: {document_type}"
            )
            logger.debug(f"Chain of Thought: {resp.output_parsed.chain_of_thought}")
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}", exc_info=True)

    if company_name == "Unknown" or document_version == "Unknown":
        match = re.search(r"(?P<company>[A-Za-z]+)_v(?P<version>[0-9]+)", filename)
        if match:
            if company_name == "Unknown":
                company_name = match.group("company")
            if document_version == "Unknown":
                document_version = "v" + match.group("version")

    if report_quarter:
        quarter_match = re.search(r"Q[1-4]", report_quarter.upper())
        report_quarter = quarter_match.group(0) if quarter_match else None

    return {
        "company_name": company_name.strip().title(),
        "document_version": document_version.strip().title(),
        "report_year": report_year,
        "report_quarter": report_quarter,
        "document_type": document_type,
    }


def split_plain_text_chunks(
    text: str, max_chunk_size: int = 4000, overlap_chunk_size: int = 400
) -> list[str]:
    """Split long-form text into bounded, overlapping chunks.

    This path protects against huge single-string artifacts (for example,
    financial reports parsed without explicit page delimiters).
    Uses LangChain's MarkdownTextSplitter for robust splitting.
    """
    if not text or not text.strip():
        return []

    splitter = MarkdownTextSplitter(
        chunk_size=max_chunk_size, chunk_overlap=overlap_chunk_size
    )
    chunks = splitter.split_text(text)
    return chunks


def calculate_as_of_date(year: int | None, quarter: str | None) -> str | None:
    """Build an as-of date string from report year and quarter.

    Args:
        year: Report year.
        quarter: Report quarter label.

    Returns:
        As-of date string or None when year missing.
    """
    if not year:
        return None
    q_map = {"Q1": "03-31", "Q2": "06-30", "Q3": "09-30", "Q4": "12-31"}
    q_norm = quarter.strip().upper() if quarter else None
    suffix = q_map.get(q_norm, "12-31") if q_norm else "12-31"
    return f"{year}-{suffix}"


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def enrich_chunk(chunk_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Enrich a chunk with cleaned text and mentioned company entities.

    Args:
        chunk_text: Raw chunk text.
        metadata: Document metadata used for context.

    Returns:
        Enriched chunk payload with extracted companies.
    """
    company = metadata.get("company_name", "Unknown")
    year = metadata.get("report_year")
    quarter = metadata.get("report_quarter")

    prompt = (
        f"Context: Document authored by {company} for {quarter} {year}.\n\n"
        "You are an elite data extraction assistant. Analyze the provided text or table.\n"
        "1. enriched_text: If the input is a table, write a dense, 3-sentence summary of the metrics. "
        "If it is standard prose, return the text cleaned of OCR artifacts.\n"
        "2. mentioned_companies: Extract a list of all canonical company names mentioned in the text. "
        "Map pronouns (e.g., 'we', 'our') to the authoring company. "
        "Identify competitors mentioned in matrices."
    )

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": f"{prompt}\n\nInput:\n{chunk_text}"}],
            text_format=ChunkEnrichmentResponse,
        )
        if resp.output_parsed:
            return {
                "enriched_text": resp.output_parsed.enriched_text,
                "mentioned_companies": resp.output_parsed.mentioned_companies,
            }
    except Exception as e:
        logger.error(f"Error enriching chunk: {e}", exc_info=True)

    return {"enriched_text": chunk_text, "mentioned_companies": [company]}


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    client_id: str,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Ingest a PDF file by parsing, chunking, and embedding.

    Args:
        file_bytes: The raw PDF byte payload.
        filename: The original name of the PDF.
        client_id: The identifier for the owning client.
        progress_cb: Optional callback function for reporting progress.

    Returns:
        A dictionary with ingestion metrics and status.
    """
    file_sha = hashlib.sha256(file_bytes).hexdigest()

    logger.info(f"Ingesting PDF: {filename} (SHA256: {file_sha})")

    if check_duplicate(client_id, "file_sha256", file_sha):
        logger.info(f"PDF {filename} skipped (already exists).")
        return {
            "skipped": True,
            "reason": "file_sha256_exists",
            "file_sha256": file_sha,
        }

    logger.debug("Parsing document structure...")
    if progress_cb:
        progress_cb("Parsing document structure...")
    
    # Pass the filename into the parser
    pages = document_parser.reducto_parse_financial_pdf(file_bytes=file_bytes, filename=filename)
    full_text = "\n---\n".join(pages)

    if not pages:
        logger.error("Parsing failed or empty text returned.")
        return {"skipped": True, "reason": "parse_failed"}

    text_sha = hashlib.sha256(full_text.encode("utf-8")).hexdigest()

    if check_duplicate(client_id, "text_sha256", text_sha):
        logger.info(f"PDF {filename} skipped (text content already exists).")
        return {
            "skipped": True,
            "reason": "text_sha256_exists",
            "file_sha256": file_sha,
            "text_sha256": text_sha,
        }

    logger.debug("Identifying company and version...")
    if progress_cb:
        progress_cb("Identifying company and version...")
    metadata = extract_metadata(pages, filename)

    doc_type = metadata.get("document_type") or "Other"

    if doc_type in SLIDE_DOC_TYPES:
        chunks = list(pages)
        logger.debug("Using slide-level chunking for presentation artifact.")
        if progress_cb:
            progress_cb("Using slide-level chunking for presentation artifact...")
    else:
        chunks = split_plain_text_chunks(full_text)
        if not chunks:
            chunks = [full_text.strip()]
        logger.debug(f"Using plain-text chunking for {doc_type} artifact.")
        if progress_cb:
            progress_cb(f"Using plain-text chunking for {doc_type} artifact...")

    logger.debug(f"Enriching {len(chunks)} chunks concurrently...")
    if progress_cb:
        progress_cb(f"Enriching {len(chunks)} chunks concurrently...")

    enriched_results = [
        {
            "enriched_text": chunk,
            "mentioned_companies": [metadata.get("company_name", "Unknown")],
        }
        for chunk in chunks
    ]

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(enrich_chunk, chunks[i], metadata): i
            for i in range(len(chunks))
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                enriched_results[idx] = future.result()
            except Exception as e:
                logger.error(
                    f"Chunk enrichment failed at index {idx}: {e}", exc_info=True
                )

    logger.debug("Generating AI search vectors...")
    if progress_cb:
        progress_cb("Generating AI search vectors...")
    summaries = [res["enriched_text"] for res in enriched_results]
    embeddings = embed_texts(summaries)  # type: ignore

    logger.debug("Registering discovered entities...")
    if progress_cb:
        progress_cb("Registering discovered entities...")

    all_discovered_companies = {metadata.get("company_name", "Unknown")}
    for res in enriched_results:
        all_discovered_companies.update(res.get("mentioned_companies", []))

    all_discovered_companies.discard("Unknown")
    register_discovered_companies(all_discovered_companies)

    raw_mentions: list[str] = []
    for res in enriched_results:
        raw_mentions.extend(res.get("mentioned_companies", []))
    raw_mentions.append(metadata.get("company_name", "Unknown"))

    mention_map = canonicalize_company_names(raw_mentions)

    doc_key = str(metadata.get("company_name", "")).strip().casefold()
    doc_canonical = mention_map.get(doc_key, [])
    if doc_canonical:
        metadata["company_name"] = doc_canonical[0]

    for res in enriched_results:
        canonical_mentions: list[str] = []
        for name in res.get("mentioned_companies", []):
            raw = str(name).strip()
            if not raw or raw.lower() == "unknown":
                continue
            canonical_mentions.extend(
                mention_map.get(raw.casefold(), [raw.strip().title()])
            )
        if doc_canonical:
            canonical_mentions.extend(doc_canonical)
        res["mentioned_companies"] = list(dict.fromkeys(canonical_mentions))

    logger.debug("Saving to knowledge base...")
    if progress_cb:
        progress_cb("Saving to knowledge base...")
    supabase = get_supabase_client()

    as_of_date = calculate_as_of_date(
        metadata["report_year"], metadata["report_quarter"]
    )

    try:
        doc_res = (
            supabase.table("documents")
            .insert(
                {
                    "client_id": client_id,
                    "document_name": filename,
                    "company_name": metadata["company_name"],
                    "document_type": metadata["document_type"],
                    "document_version": metadata["document_version"],
                    "report_year": metadata["report_year"],
                    "report_quarter": metadata["report_quarter"],
                    "as_of_date": as_of_date,
                    "file_sha256": file_sha,
                    "text_sha256": text_sha,
                    "status": "INGESTING",
                }
            )
            .execute()
        )
    except Exception as e:
        # Check both the attribute and the string representation to be completely safe
        if (hasattr(e, "code") and e.code == "23505") or "23505" in str(e):
            logger.info(f"Race condition caught. {filename} is already processing.")
            return {"skipped": True, "reason": "concurrent_ingestion"}
        raise

    if isinstance(doc_res.data, list) and len(doc_res.data) > 0:
        document_id = doc_res.data[0]["id"]  # type: ignore
    else:
        raise ValueError("Unexpected response from database during document insertion")

    rows: list[dict[str, Any]] = []
    for i, (chunk_text, res, emb) in enumerate(
        zip(chunks, enriched_results, embeddings, strict=True), start=1
    ):
        rows.append(
            {
                "document_id": document_id,
                "chunk_text": res["enriched_text"],
                "raw_content": chunk_text,
                "page_number": i,
                "embedding": emb,
                "mentioned_companies": res["mentioned_companies"],
            },
        )

    logger.info(f"Inserting {len(rows)} chunks into database...")
    batch_size = 300
    inserted = 0
    try:
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            res = supabase.table("document_chunks").insert(batch).execute()
            inserted += len(res.data) if isinstance(res.data, list) else 0

        # Two-Phase Commit: Flip to READY only if all chunks succeed
        supabase.table("documents").update({"status": "READY"}).eq(
            "id", document_id
        ).execute()
    except Exception as e:
        logger.critical(
            f"Ingestion failed for {document_id}. Attempting rollback cleanup. Error: {e}"
        )
        try:
            supabase.table("documents").delete().eq("id", document_id).execute()
            logger.info(f"Rollback succeeded for document {document_id}.")
        except Exception as rollback_error:
            logger.critical(
                f"Rollback failed for document {document_id}. Orphan may remain. Error: {rollback_error}",
                exc_info=True,
            )
        raise

    logger.info(f"Successfully processed PDF {filename} with {inserted} inserts.")

    return {
        "skipped": False,
        "file_sha256": file_sha,
        "text_sha256": text_sha,
        "chunks_total": len(chunks),
        "inserted": inserted,
        **metadata,
    }
