from unittest.mock import patch, MagicMock

from src.generator import generate_answer


def test_citation_format_validation() -> None:
    chunks = [
        {
            "document_name": "Acme",
            "document_version": "v1",
            "chunk_text": "Revenue was 10M.",
        },
    ]
    # We can test the validation loop directly if we patch the LLM,
    # but an easy smoke test is just to ensure generate_answer runs the loop.
    with patch("src.generator.get_openai_client") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock Responses API response to include a valid citation
        mock_client.responses.create.return_value = MagicMock(
            output_text="Revenue was 10M [Acme, v1]."
        )

        answer = generate_answer("What was revenue?", chunks)
        assert "[Acme, v1]" in answer
        assert mock_client.responses.create.call_count == 1
