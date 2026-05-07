import re
from typing import Any

from openai.types.responses import ResponseInputItemParam
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = (
    "You are an elite financial analyst. Answer the user's query based ONLY on the provided context.\n"
    "NO EXTERNAL KNOWLEDGE: Do not use any outside information. If the answer is not in the context, state: 'I do not have enough information in the provided documents to answer this query.'\n"
    "MARKDOWN FORMATTING: Strictly use GitHub Flavored Markdown. ALWAYS ensure there is a blank line (double newline) before and after any markdown table. "
    "Use markdown tables for any financial data or numerical comparisons whenever possible.\n"
    "CURRENCY & MATH: Escaping dollar signs is CRITICAL. To avoid being misinterpreted as LaTeX math symbols, you MUST escape all dollar signs with a backslash (e.g., use \\$100 instead of $100).\n"
    "CITATIONS REQUIRED: You must cite your sources using the exact format: [Document Name, VersionValue] at the end of the sentence or paragraph. "
    "If citing after a table, ensure the citation is on its own new line below the table, NOT inside the table rows.\n"
    "CONFLICT RESOLUTION: If retrieved chunks contain conflicting numbers or facts across different versions, do NOT state an average. "
    "You MUST state: 'Source A says X [A, v1]. Source B says Y [B, v2].'\n"
    "VISUAL LIMITATIONS: You cannot see charts or graphs. If the query requires visual analysis, state your limitation but summarize the textual data available.\n"
)


def _parse_citations(answer: str) -> list[tuple[str, str]]:
    """Extract citations in [Document Name, VersionValue] format.

    Args:
        answer: The generated answer text to scan.

    Returns:
        A list of tuples containing (document_name, version).
    """
    matches = re.findall(r"\[([^,\]]+),\s*([^\]]+)\]", answer)
    return [(doc.strip(), ver.strip()) for doc, ver in matches]


def combine_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved DB chunks into a clean context string for the LLM.

    Args:
        chunks: A list of dicts representing the raw DB document chunks.

    Returns:
        A formatted string with headers and chunk payload text.
    """
    parts = []
    for c in chunks:
        doc = c.get("document_name", "Unknown")
        ver = c.get("document_version", "Unknown")
        cid = c.get("id")
        text = c.get("chunk_text", "")

        header = f"Source: {doc}, Version: {ver}, Chunk ID: {cid}\n"
        parts.append(header + text)

    return "\n\n".join(parts)


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
    context = combine_context_chunks(retrieved_chunks)

    messages: list[ResponseInputItemParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_query}"},
    ]

    openai_client = get_openai_client()
    resp = openai_client.responses.create(
        model=settings.reasoning_model,
        input=messages,
        reasoning={"effort": "low"},
    )

    answer = (resp.output_text or "").strip()

    logger.debug(f"Generated text: {answer[:200]}...")
    return answer
