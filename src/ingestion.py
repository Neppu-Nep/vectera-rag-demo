import hashlib
import re
from typing import Any, Callable
from pydantic import BaseModel

from langchain_text_splitters import MarkdownTextSplitter

from src.core.clients import get_openai_client, get_supabase_client
from src.core.config import settings
from src.core.logger import get_logger
from src.utils import parser as llama_parser
from src.utils.embeddings import embed_texts

logger = get_logger(__name__)


class MetadataResponse(BaseModel):
    """Schema for document metadata extraction."""
    company_name: str
    document_version: str


def _check_duplicate(client_id: str, column: str, value: str) -> bool:
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


def extract_metadata(text: str, filename: str) -> dict[str, str]:
    """Retrieve metadata (company name, version) from the document via LLM.

    Args:
        text: The parsed text of the document.
        filename: The filename of the document for fallback parsing.

    Returns:
        A dictionary with 'company_name' and 'document_version'.
    """
    first_chunk = text[:3000]
    meta_prompt = (
        f"Extract JSON with keys company_name and document_version from the following text.\n"
        f"Text:\n{first_chunk}\n\nRespond using only JSON object."
    )

    openai_client = get_openai_client()
    company_name = "Unknown"
    document_version = "Unknown"

    try:
        resp = openai_client.responses.parse(
            model=settings.reasoning_model,
            input=[{"role": "user", "content": meta_prompt}],
            reasoning={"effort": "low"},
            text_format=MetadataResponse,
        )
        if resp.output_parsed:
            company_name = resp.output_parsed.company_name
            document_version = resp.output_parsed.document_version
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

    return {
        "company_name": company_name.strip().title(),
        "document_version": document_version.strip().title(),
    }


def chunk_document(text: str) -> list[str]:
    """Split document text into chunks for embedding.

    Args:
        text: The full extracted text.

    Returns:
        A list of raw text chunks.
    """
    splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=400)
    return splitter.split_text(text)


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

    if _check_duplicate(client_id, "file_sha256", file_sha):
        logger.info(f"PDF {filename} skipped (already exists).")
        return {
            "skipped": True,
            "reason": "file_sha256_exists",
            "file_sha256": file_sha,
        }

    logger.debug("Parsing document structure...")
    if progress_cb:
        progress_cb("Parsing document structure...")
    text = llama_parser.parse_financial_pdf(file_bytes=file_bytes)
    if not text:
        logger.error("Parsing failed or empty text returned.")
        return {"skipped": True, "reason": "parse_failed"}

    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if _check_duplicate(client_id, "text_sha256", text_sha):
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
    metadata = extract_metadata(text, filename)

    if metadata["company_name"] == "Unknown" or metadata["document_version"] == "Unknown":
        logger.error(f"Metadata extraction failed for {filename}. Cannot ingest.")
        return {"skipped": True, "reason": "metadata_unknown"}

    logger.debug("Splitting into searchable sections...")
    if progress_cb:
        progress_cb("Splitting into searchable sections...")
    prepared_chunks = chunk_document(text)

    logger.debug("Generating AI search vectors...")
    if progress_cb:
        progress_cb("Generating AI search vectors...")
    embeddings = embed_texts(prepared_chunks)

    logger.info("Saving to knowledge base...")
    if progress_cb:
        progress_cb("Saving to knowledge base...")
    supabase = get_supabase_client()
    
    doc_res = supabase.table("documents").insert({
        "client_id": client_id,
        "document_name": filename,
        "company_name": metadata["company_name"],
        "document_version": metadata["document_version"],
        "file_sha256": file_sha,
        "text_sha256": text_sha,
    }).execute()
    
    if isinstance(doc_res.data, list) and len(doc_res.data) > 0:
        document_id = doc_res.data[0]["id"]  # type: ignore
        if document_id is None:
            raise ValueError("No id returned from documents insert")
    else:
        raise ValueError("Unexpected response from database during document insertion")

    rows: list[dict[str, Any]] = []
    for chunk_text, emb in zip(prepared_chunks, embeddings, strict=False):
        rows.append(
            {
                "document_id": document_id,
                "chunk_text": chunk_text,
                "embedding": emb,
            },
        )

    logger.info(f"Inserting {len(rows)} chunks into database...")
    inserted = 0
    for i in range(0, len(rows), 500):
        batch = rows[i : i + 500]
        res = supabase.table("document_chunks").insert(batch).execute()
        inserted += len(res.data) if isinstance(res.data, list) else len(batch)

    logger.info(f"Successfully processed PDF {filename} with {inserted} inserts.")

    return {
        "skipped": False,
        "file_sha256": file_sha,
        "text_sha256": text_sha,
        "chunks_total": len(prepared_chunks),
        "inserted": inserted,
        **metadata,
    }
