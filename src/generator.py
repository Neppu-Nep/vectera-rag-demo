from typing import Any

from openai.types.responses import ResponseInputItemParam
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an elite financial analyst. Answer the user's query based ONLY on the provided context.\n"
    "NO EXTERNAL KNOWLEDGE: Do not use any outside information. If the answer is not in the context, you MUST still state that clearly AND cite the documents you checked to reach that conclusion (e.g., 'Based on the checked reports, I cannot find... [^1]').\n"
    "MARKDOWN FORMATTING: Strictly use GitHub Flavored Markdown. ALWAYS ensure there is a blank line (double newline) before and after any markdown table. Use markdown tables for any financial data or numerical comparisons whenever possible.\n"
    "CURRENCY & MATH: Escaping dollar signs is CRITICAL. To avoid being misinterpreted as LaTeX math symbols, you MUST escape all dollar signs with a backslash (e.g., use \\$100 instead of $100).\n"
    
    "FINANCIAL PERIOD ACCURACY: When quoting metrics or financials, ALWAYS specify the 'As-Of Period' provided in the source headers so the user knows exactly what timeframe the data represents.\n"
    
    "CITATIONS REQUIRED: You must cite your sources using standard Markdown Footnotes. Every single response MUST contain at least one citation. Do not return any text without standard Markdown Footnotes.\n"
    "Place a superscript number like [^1] or [^2] directly after the relevant sentence or paragraph. If a sentence uses multiple sources, use [^1][^2].\n"
    # "At the bottom of your response, add a 'Sources' section mapping the numbers to the Company, Document Type, As-Of Period, and Page (e.g., [^1]: Public Storage - Presentation (Q4 2025), Page 4)."
)


def reorder_lost_in_the_middle(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reorder chunks to put the most relevant at the beginning and end.
    
    LLMs suffer from 'lost in the middle' syndrome where they ignore context
    in the middle of a large prompt. This reorders [1, 2, 3, 4, 5] into [1, 3, 5, 4, 2].
    """
    if not chunks:
        return []
        
    reordered = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            reordered.insert(0, chunk)
        else:
            reordered.append(chunk)
            
    return reordered


def combine_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format the retrieved chunks into a single string for the LLM prompt."""
    parts = []

    for chunk in chunks:
        cid = chunk.get("id", "Unknown ID")
        doc = chunk.get("document_name", "Unknown Document")
        comp = chunk.get("company_name", "Unknown Company")
        doc_type = chunk.get("document_type", "Unknown Type")
        yr = chunk.get("report_year", "N/A")
        qtr = chunk.get("report_quarter", "")
        ver = chunk.get("document_version", "Unknown Version")
        page = chunk.get("page_number", "N/A")
        
        period = f"{qtr} {yr}".strip() if yr != "N/A" else "Unknown Period"
        
        text = chunk.get("raw_content") or chunk.get("chunk_text", "")

        header = (
            f"--- SOURCE ID: {cid} ---\n"
            f"Company: {comp}\n"
            f"Document: {doc}\n"
            f"Type: {doc_type}\n"
            f"As-Of Period: {period}\n"
            f"Version/Date: {ver}\n"
            f"Page: {page}\n"
            f"Content:\n"
        )
        chunk_block = header + text + "\n\n"
        parts.append(chunk_block)

    return "".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(user_query: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    """Pass user query and injected constraints into the final generation LLM call.

    Args:
        user_query: The plain text user query.
        retrieved_chunks: A list of document chunks that have high relevance.

    Returns:
        The generated answer string from the model.
    """
    logger.info("Generating final answer with openai...")
    
    # Lost in the Middle: Reorder chunks to leverage Primacy/Recency bias
    retrieved_chunks = reorder_lost_in_the_middle(retrieved_chunks)
    context = combine_context_chunks(retrieved_chunks)

    messages: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_query}"},
    ]

    openai_client = get_openai_client()
    try:
        response = openai_client.responses.create(
            model=settings.reasoning_model,
            input=messages,
            temperature=0.0,
        )
        answer = (response.output_text or "").strip()
        logger.debug(f"Generated text: {answer[:200]}...")
        return answer
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return "An error occurred while generating the answer. Please try again."