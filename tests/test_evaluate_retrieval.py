from scripts.evaluate_retrieval import _matches_expectation


def test_matches_expectation_requires_content_evidence() -> None:
    chunk = {
        "document_version": "v2",
        "company_name": "Acme",
        "page_number": 14,
        "chunk_text": "Revenue increased to 14.2M during the quarter.",
        "raw_content": "Revenue increased to 14.2M during the quarter.",
    }

    assert _matches_expectation(chunk, ["v2"], ["Acme"], expected_substrings=["14.2M"])
    assert _matches_expectation(chunk, ["v2"], ["Acme"], expected_page_numbers=[14, 20])
    assert not _matches_expectation(
        chunk, ["v2"], ["Acme"], expected_substrings=["occupancy"]
    )
    assert not _matches_expectation(chunk, ["v2"], ["Acme"], expected_page_numbers=[15])
