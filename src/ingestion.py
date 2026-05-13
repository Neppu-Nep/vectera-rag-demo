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
from src.utils import parser as llama_parser
from src.utils.embeddings import embed_texts

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
    # Grab the first 10 and last 5 pages to create a condensed representation for the LLM prompt.
    total_pages = len(pages)
    if total_pages <= 15:
        document_chunk = "\n\n".join([f"--- PAGE {i+1} ---\n{page}" for i, page in enumerate(pages)])
    else:
        head_pages = "\n\n".join([f"--- PAGE {i+1} ---\n{page}" for i, page in enumerate(pages[:10])])
        tail_pages = "\n\n".join([f"--- PAGE {total_pages - 4 + i} ---\n{page}" for i, page in enumerate(pages[-5:])])
        
        document_chunk = (
            head_pages + 
            "\n\n...\n[TEXT TRUNCATED TO SAVE CONTEXT]\n...\n\n" + 
            tail_pages
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
            logger.info(f"Extracted metadata - Company: {company_name}, Version: {document_version}, Year: {report_year}, Quarter: {report_quarter}, Type: {document_type}")
            logger.debug(f"Chain of Thought: {resp.output_parsed.chain_of_thought}")
    except Exception as e:
        logger.error(f"Error fetching metadata: {e}", exc_info=True)

    # Fallback: Try to extract from filename if LLM fails
    if company_name == "Unknown" or document_version == "Unknown":
        match = re.search(r"(?P<company>[A-Za-z]+)_v(?P<version>[0-9]+)", filename)
        if match:
            if company_name == "Unknown":
                company_name = match.group("company")
            if document_version == "Unknown":
                document_version = "v" + match.group("version")

    # Cleanup and normalization
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


def split_plain_text_chunks(text: str, max_chunk_size: int = 4000, overlap_chunk_size: int = 400) -> list[str]:
    """Split long-form text into bounded, overlapping chunks.

    This path protects against huge single-string artifacts (for example,
    financial reports parsed without explicit page delimiters).
    Uses LangChain's MarkdownTextSplitter for robust splitting.
    """
    if not text or not text.strip():
        return []

    splitter = MarkdownTextSplitter(chunk_size=max_chunk_size, chunk_overlap=overlap_chunk_size)
    chunks = splitter.split_text(text)
    return chunks


def page_has_table(page_text: str) -> bool:
    """Detect table-like markdown in a page."""
    if "|---|" in page_text or "| --- |" in page_text:
        return True
    return bool(re.search(r"\|\s*-{3,}", page_text))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_page_summary(page_text: str, metadata: dict[str, Any]) -> str:
    """Generate a dense summary for a slide containing a table."""
    company = metadata.get("company_name", "Unknown")
    year = metadata.get("report_year")
    quarter = metadata.get("report_quarter")
    
    context_prefix = f"This slide belongs to {company}"
    if year:
        context_prefix += f" for {year}"
    if quarter:
        context_prefix += f" {quarter}"
    context_prefix += "."

    prompt = (
        f"{context_prefix}\n"
        "You are a data extraction assistant. The following text is a slide from a financial presentation. "
        "Write a dense, 3-sentence summary of the page's core message and the key metrics inside the table "
        "so it can be easily found via semantic search. Do not hallucinate."
    )

    openai_client = get_openai_client()
    try:
        resp = openai_client.responses.create(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": f"{prompt}\n\nSlide:\n{page_text}"}],
        )
        summary = (resp.output_text or "").strip()
        return summary or page_text
    except Exception as e:
        logger.error(f"Error summarizing slide: {e}", exc_info=True)
        return page_text


def ingest_pdf(file_bytes: bytes, filename: str, client_id: str, progress_cb: Callable[[str], None] | None = None) -> dict[str, Any]:
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
    pages = llama_parser.parse_financial_pdf(file_bytes=file_bytes)
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

    summaries = list(chunks)
    if doc_type in SLIDE_DOC_TYPES:
        table_indexes = [i for i, p in enumerate(chunks) if page_has_table(p)]
        if table_indexes:
            logger.debug(f"Summarizing {len(table_indexes)} table slides...")
            if progress_cb:
                progress_cb(f"Summarizing {len(table_indexes)} table slides...")

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_map = {
                    executor.submit(generate_page_summary, chunks[i], metadata): i
                    for i in table_indexes
                }
                for future in as_completed(future_map):
                    idx = future_map[future]
                    try:
                        summaries[idx] = future.result()
                    except Exception as e:
                        logger.error(f"Slide summary failed: {e}", exc_info=True)
                        summaries[idx] = chunks[idx]

    logger.debug("Generating AI search vectors...")
    if progress_cb:
        progress_cb("Generating AI search vectors...")
    embeddings = embed_texts(summaries)

    logger.debug("Saving to knowledge base...")
    if progress_cb:
        progress_cb("Saving to knowledge base...")
    supabase = get_supabase_client()
    
    try:
        doc_res = supabase.table("documents").insert({
            "client_id": client_id,
            "document_name": filename,
            "company_name": metadata["company_name"],
            "document_type": metadata["document_type"],
            "document_version": metadata["document_version"],
            "report_year": metadata["report_year"],
            "report_quarter": metadata["report_quarter"],
            "file_sha256": file_sha,
            "text_sha256": text_sha,
            "status": "INGESTING",
        }).execute()
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
    for i, (chunk_text, summary_text, emb) in enumerate(zip(chunks, summaries, embeddings, strict=True), start=1):
        rows.append(
            {
                "document_id": document_id,
                "chunk_text": summary_text,
                "raw_content": chunk_text,
                "page_number": i,
                "embedding": emb,
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
        supabase.table("documents").update({"status": "READY"}).eq("id", document_id).execute()
    except Exception as e:
        logger.critical(f"Ingestion failed for {document_id}. Attempting rollback cleanup. Error: {e}")
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
