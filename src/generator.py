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

CONFLICT RESOLUTION & VERSION AWARENESS (CRITICAL):
When given multiple versions of a company's documents (e.g., Q3 vs Q4). 
1. NEVER average or silently merge conflicting numbers across different periods or versions.
2. If data differs between versions, explicitly state the differences. For example: "In Q3, revenue was \\$100M, but in Q4 it was \\$120M, indicating a 20% increase."

TAXONOMY MISMATCHES (FORGIVING GENERATION):
If the user asks for data from a specific document type (e.g., "in the deck") but the provided context contains the answer in a different document type (e.g., "Press Release"), you MUST still answer the question. Politely note the substitution (e.g., "I couldn't find this in the presentation, but the Q4 Press Release states...").

MARKDOWN & CITATIONS: 
1. Use GitHub Flavored Markdown. 
2. Escape all dollar signs with a backslash (e.g., \\$100). 
3. AGGREGATE CITATIONS: Do not spam footnotes on every single sentence or table row. Place the footnote at the end of a paragraph or the end of a cohesive section.
4. If multiple sources apply, separate them with a space (e.g., [^1] [^2]).
5. Footnotes must be unique per source. If you cite the exact same document, reuse the same footnote number.
6. You MUST provide the footnote definitions at the VERY END of your answer using the format:
[^1]: [File Name] | [Version] | Page: [Page Number]
[^2]: [File Name] | [Version] | Page: [Page Number]

THINKING PROCESS:
Before writing your final answer, you must use a <thinking> block to identify the different document versions provided and map out any conflicting data points or chronological changes. 

OUTPUT RULES (STRICT):
1. ZERO REPETITION: Never output the same information twice. Do not write a 'Bottom Line' or 'Summary' that just repeats the data you already provided.
2. USE TABLES FOR COMPARISONS: If you are comparing lists, rankings, or multiple metrics across two periods, ALWAYS use Markdown tables. Do not write massive lists of text bullet points.
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
        comp = chunk.get('company_name', 'Unknown Company')
        ver = chunk.get('document_version', 'Unknown Version')
        grouped_chunks[comp][ver].append(chunk)

    parts = []
    for comp, versions in grouped_chunks.items():
        parts.append(f"=== COMPANY: {comp} ===")
        for ver, v_chunks in versions.items():
            parts.append(f"\n--- VERSION: {ver} ---")
            for chunk in v_chunks:
                cid = chunk.get('id', 'Unknown ID')
                doc = chunk.get('document_name', 'Unknown Document')
                doc_type = chunk.get('document_type', 'Other')
                yr = chunk.get('report_year', 'N/A')
                qtr = chunk.get('report_quarter', '')
                page = chunk.get('page_number', 'N/A')
                period = f"{qtr} {yr}".strip() if yr != 'N/A' else 'Unknown Period'
                text = chunk.get('raw_content') or chunk.get('chunk_text', '')

                header = f"\n[Doc: {doc} | Type: {doc_type} | As-Of: {period} | Page: {page} | ChunkID: {cid}]\n"
                parts.append(header + text)

    return "\n".join(parts)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_answer(user_query: str, retrieved_chunks: list[dict[str, Any]], is_comparison: bool = False) -> str:
    """Pass user query and injected constraints into the final generation LLM call.

    Args:
        user_query: The plain text user query.
        retrieved_chunks: A list of document chunks that have high relevance.
        is_comparison: Flag indicating if the Router detected this as a comparison query, which will trigger additional instructions in the prompt.

    Returns:
        The generated answer string from the model.
    """
    logger.info("Generating final answer with conflict-resolution reasoning...")

    # Document-Family Grouping
    context = combine_context_chunks(retrieved_chunks)

    # Dynamically adjust the prompt based on the Router's intent detection
    dynamic_prompt = SYSTEM_PROMPT
    if is_comparison:
        dynamic_prompt += (
            "\n\nCRITICAL ROUTING INSTRUCTION: The system has detected this is a COMPARISON query. "
            "You MUST explicitly compare the metrics, calculate the deltas, and point out any conflicting "
            "data between the provided document versions."
        )

    messages: list[ResponseInputItemParam] =[
        {"role": "system", "content": dynamic_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nUser Query: {user_query}"}
    ]

    openai_client = get_openai_client()
    try:
        response = openai_client.responses.create(
            model=settings.reasoning_model,
            input=messages,
            temperature=0.0
        )
        answer = (response.output_text or "").strip()

        # Remove the <thinking> block from the final output shown to the user
        if "</thinking>" in answer:
            answer = answer.split("</thinking>")[-1].strip()

        logger.debug(f"Generated text: {answer[:200]}...")
        return answer
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        return "An error occurred while generating the answer. Please try again."