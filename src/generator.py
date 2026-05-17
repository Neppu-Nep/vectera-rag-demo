from typing import Any
from collections import defaultdict

from openai.types.responses import ResponseInputItemParam
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.clients import get_openai_client
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an elite financial analyst. Answer the user's query based ONLY on the provided context.

NO EXTERNAL KNOWLEDGE: If the answer is not in the context, state that clearly and cite the documents you checked.

TEMPORAL RECENCY (CRITICAL):
If the context contains historical data across different versions, and the user DOES NOT explicitly ask for a comparison, you MUST output ONLY the most recent metric available. Explicitly suppress the older data (e.g., "As of Q4 2025, revenue was \\$100M."). Do not summarize the past unless asked.

CONFLICT RESOLUTION & VERSION AWARENESS:
When given multiple versions of a company's documents (e.g., Q3 vs Q4). 
1. NEVER average or silently merge conflicting numbers across different periods or versions.
2. If data differs between versions, explicitly state the differences. 

TAXONOMY MISMATCHES (FORGIVING GENERATION):
If the user asks for data from a specific document type but the provided context contains the answer in a different document type, you MUST still answer the question. Politely note the substitution.

MARKDOWN & CITATIONS: 
1. Use GitHub Flavored Markdown. 
2. Escape all dollar signs with a backslash. 
3. AGGREGATE CITATIONS: Place footnotes at the end of a cohesive section.
4. If multiple sources apply, separate them with a space.
5. You MUST provide the footnote definitions at the VERY END using the format:
[^1]: [File Name] | [Version] | Page: [Page Number]

THINKING PROCESS:
Before writing your final answer, use a <thinking> block to identify the different document versions provided and map out any chronological changes. 

OUTPUT RULES (STRICT):
1. ZERO REPETITION: Never output the same information twice. 
2. USE TABLES FOR COMPARISONS: Always use Markdown tables when comparing lists, rankings, or multiple metrics.
3. NO FLUFF: Get straight to the data.
"""


def combine_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Combine retrieved chunks into a single context string, grouped by company and document version to enable Document-Family Reasoning.

    Args:
        chunks (list[dict[str, Any]]): List of retrieved chunks with metadata.

    Returns:
        str: Combined context string with clear demarcations for company and document version.
    """

    # Group by Company, then by Version to enforce Document-Family Reasoning
    grouped_chunks = defaultdict(lambda: defaultdict(list))

    for chunk in chunks:
        comp = chunk.get("company_name", "Unknown Company")
        ver = chunk.get("document_version", "Unknown Version")
        grouped_chunks[comp][ver].append(chunk)

    parts = []
    for comp, versions in grouped_chunks.items():
        parts.append(f"=== COMPANY: {comp} ===")
        for ver, v_chunks in versions.items():
            parts.append(f"\n--- VERSION: {ver} ---")
            for chunk in v_chunks:
                cid = chunk.get("id", "Unknown ID")
                doc = chunk.get("document_name", "Unknown Document")
                doc_type = chunk.get("document_type", "Other")
                yr = chunk.get("report_year", "N/A")
                qtr = chunk.get("report_quarter", "")
                page = chunk.get("page_number", "N/A")
                period = f"{qtr} {yr}".strip() if yr != "N/A" else "Unknown Period"
                text = chunk.get("raw_content") or chunk.get("chunk_text", "")

                header = f"\n[Doc: {doc} | Type: {doc_type} | As-Of: {period} | Page: {page} | ChunkID: {cid}]\n"
                parts.append(header + text)

    return "\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(
    user_query: str, retrieved_chunks: list[dict[str, Any]], is_comparison: bool = False
) -> str:
    """Pass user query and injected constraints into the final generation LLM call.

    Args:
        user_query: The plain text user query.
        retrieved_chunks: A list of document chunks that have high relevance.
        is_comparison: Flag indicating if the Router detected this as a comparison query, which will trigger additional instructions in the prompt.

    Returns:
        The generated answer string from the model.
    """
    logger.info("Generating final answer with conflict-resolution reasoning...")

    context = combine_context_chunks(retrieved_chunks)

    dynamic_prompt = SYSTEM_PROMPT
    if is_comparison:
        dynamic_prompt += (
            "\n\nCRITICAL ROUTING INSTRUCTION: The system has detected this is a COMPARISON query. "
            "You MUST explicitly compare the metrics, calculate the deltas, and point out any conflicting "
            "data between the provided document versions."
        )

    messages: list[ResponseInputItemParam] = [
        {"role": "system", "content": dynamic_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_query}"},
    ]

    openai_client = get_openai_client()
    try:
        response = openai_client.responses.create(
            model=settings.reasoning_model, input=messages, temperature=0.0
        )
        answer = (response.output_text or "").strip()

        if "</thinking>" in answer:
            answer = answer.split("</thinking>")[-1].strip()

        logger.debug(f"Generated text: {answer[:200]}...")
        return answer
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return "An error occurred while generating the answer. Please try again."
