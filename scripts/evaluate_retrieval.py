import os
from typing import Any
from dotenv import load_dotenv

from src.retriever import retrieve_context

load_dotenv()

CLIENT_ID = os.getenv("EVAL_CLIENT_ID", "Vectera_Capital_Fund")
K = int(os.getenv("TOP_K", "5"))

GOLDEN_DATASET: list[dict[str, Any]] = [
    {
        "query": "How did Digital Realty's total Enterprise Value and Equity Market Capitalization change between Q3 2025 and Q4 2025 ?",
        "expected_doc_versions": ["March 2026", "December 2025"],
        "expected_years": [2025],
        "expected_quarters": ["Q3", "Q4"],
        "expected_companies": ["Digital Realty"],
        "expected_substrings": ["$60 Bn", "$79 Bn", "$54 Bn", "$73 Bn", "Total Enterprise Value", "Equity Market Capitalization"],
        "expected_page_numbers": [3]
    },
    {
        "query": "Compare the total bookings at 100% share for Q3 2025 versus Q4 2025. Did the backlog grow or shrink by the end of the year for Digital Realty?",
        "expected_doc_versions": ["March 2026", "December 2025"],
        "expected_years": [2025],
        "expected_quarters": ["Q3", "Q4"],
        "expected_companies": ["Digital Realty"],
        "expected_substrings": ["$201M", "$400M", "$852M", "$1.4Bn"],
        "expected_page_numbers": []
    },
    {
        "query": "What specific shift in global data center AI workloads is projected to happen by 2027 according to Digital Realty?",
        "expected_doc_versions": ["March 2026"],
        "expected_years": [],
        "expected_quarters": [],
        "expected_companies": ["Digital Realty"],
        "expected_substrings": ["AI Inference overtakes AI Training"],
        "expected_page_numbers": [10]
    },
    {
        "query": "Did Boston Properties present any forward-looking leasing volume forecasts in their Deck that were not included or were presented differently in their Investor Presentation?",
        "expected_doc_versions": ["2025 Investor Day", "March 20, 2026"],
        "expected_years": [],
        "expected_quarters": [],
        "expected_companies": ["Boston Properties"],
        "expected_substrings": ["10M SF", "6.5M SF"],
        "expected_page_numbers": [69, 73]
    },
    {
        "query": "Compare the Q4 2025 occupancy rates across Realty Income, Boston Properties, and EastGroup Properties. Which company and asset class reported the highest occupancy?",
        "expected_doc_versions": ["February 2026", "March 20, 2026", "February 2026"],
        "expected_years": [],
        "expected_quarters": [],
        "expected_companies": ["Realty Income", "Boston Properties", "EastGroup Properties"],
        "expected_substrings": ["98.9", "89.4", "97.0"],
        "expected_page_numbers": [3, 5, 29]
    }
]


def _matches_expectation(
    chunk: dict[str, Any],
    expected_versions: list[str],
    expected_companies: list[str],
    expected_page_numbers: list[int] | None = None,
    expected_substrings: list[str] | None = None,
) -> bool:
    version = str(chunk.get("document_version", "")).strip().lower()
    company = str(chunk.get("company_name", "")).strip().lower()
    if (version and version not in [v.lower() for v in expected_versions]) or not any(company.find(c.lower()) != -1 for c in expected_companies):
        print(f"Chunk version '{version}' or company '{company}' does not match expected versions {expected_versions} or companies {expected_companies}")
        return False

    if expected_page_numbers:
        page_number = chunk.get("page_number", -1)
        if page_number not in expected_page_numbers:
            print(f"Chunk page number {page_number} not in expected {expected_page_numbers}")
            return False

    if expected_substrings:
        haystack = " ".join(
            str(chunk.get(field, "")) for field in ("raw_content", "chunk_text")
        ).lower()
        if not any(term.lower() in haystack for term in expected_substrings):
            print(f"Chunk does not contain any expected substrings {expected_substrings}")
            return False

    return True


def main() -> None:
    missing_env = [
        name
        for name in ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
        if not os.getenv(name)
    ]
    if missing_env:
        print(
            "Missing required environment variables for retrieval eval: "
            + ", ".join(missing_env)
        )
        return

    hits = 0
    total = len(GOLDEN_DATASET)

    print(f"Running retrieval eval for client='{CLIENT_ID}', K={K}")
    for i, item in enumerate(GOLDEN_DATASET, start=1):
        query = item["query"]
        expected_versions = item["expected_doc_versions"]
        expected_companies = item["expected_companies"]
        expected_page_numbers = item.get("expected_page_numbers")
        expected_substrings = item.get("expected_substrings")

        try:
            chunks, _filters = retrieve_context(query, CLIENT_ID, rerank_with_model=False)
        except Exception as exc:
            print(f"Eval aborted on query {i}: {exc}")
            return

        print(f"Retrieved {len(chunks)} chunks for query '{query}' with filters: {_filters}")
        top_k = chunks[:K]
        hit = any(
            _matches_expectation(
                c,
                expected_versions,
                expected_companies,
                expected_page_numbers=expected_page_numbers,
                expected_substrings=expected_substrings,
            )
            for c in top_k
        )
        hits += int(hit)

        status = "HIT" if hit else "MISS"
        expectation_bits = [f"version in {expected_versions}", f"company in {expected_companies}"]
        if expected_page_numbers:
            expectation_bits.append(f"page in {expected_page_numbers}")
        if expected_substrings:
            expectation_bits.append(f"contains '{', '.join(expected_substrings)}'")
        print(f"[{i}/{total}] {status} | query='{query}' | expected={' | '.join(expectation_bits)}\n\n")

    hit_rate = hits / total if total else 0.0
    print(f"\nHit Rate@{K}: {hit_rate:.2%} ({hits}/{total})")


if __name__ == "__main__":
    main()
