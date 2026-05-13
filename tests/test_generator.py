from unittest.mock import patch, MagicMock

from src.generator import generate_answer


def test_generate_answer_strips_thinking_block() -> None:
    chunks = [
        {
            "document_name": "Acme",
            "document_version": "v1",
            "chunk_text": "Revenue was 10M.",
        },
    ]
    with patch("src.generator.get_openai_client") as mock_openai:
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Mock Responses API response with a <thinking> block
        mock_client.responses.create.return_value = MagicMock(
            output_text="<thinking>analysis</thinking>\nFinal answer."
        )

        answer = generate_answer("What was revenue?", chunks)
        assert answer == "Final answer."
        assert mock_client.responses.create.call_count == 1
