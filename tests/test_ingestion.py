from src.ingestion import ingest_pdf
from unittest.mock import patch, MagicMock

@patch("src.ingestion.get_supabase_client")
@patch("src.ingestion.embed_texts")
@patch("src.ingestion.llama_parser.parse_financial_pdf")
@patch("src.ingestion.extract_metadata")
@patch("src.ingestion.check_duplicate")
def test_ingestion_smoke(mock_dup, mock_meta, mock_parse, mock_embed, mock_supabase) -> None:
    # Setup mocks
    mock_parse.return_value = ["Page 1 text.", "Page 2 text."]
    mock_meta.return_value = {
        "company_name": "TestCompany",
        "document_version": "v1",
        "report_year": 2025,
        "report_quarter": "Q3",
        "document_type": "Presentation",
    }
    mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_dup.return_value = False
    
    mock_db = MagicMock()
    mock_db.table().insert().execute.return_value = MagicMock(data=[{"id": 1}])
    mock_supabase.return_value = mock_db

    # Test
    result = ingest_pdf(b"dummy bytes", "test.pdf", "client_1")
    
    # Assert
    assert not result.get("skipped", False)
    assert result.get("inserted", 0) > 0
    mock_parse.assert_called_once()
    mock_meta.assert_called_once()
    mock_embed.assert_called_once()
    assert any(
        call.args
        and isinstance(call.args[0], dict)
        and call.args[0].get("document_type") == "Presentation"
        for call in mock_db.table.return_value.insert.call_args_list
    )
    assert mock_db.table().insert().execute.call_count == 2
