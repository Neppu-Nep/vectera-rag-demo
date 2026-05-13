from types import SimpleNamespace

from src import retriever


def test_embed_query_and_rpc(monkeypatch) -> None:
    # Mock embedding
    monkeypatch.setattr(retriever, "embed_texts", lambda q: [[0.1, 0.2, 0.3] for _ in q])
    class FakeEmbeddings:
        def create(self, model, input, **kwargs):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

    class FakeClient:
        embeddings = FakeEmbeddings()

    monkeypatch.setattr(retriever, "get_openai_client", lambda: FakeClient())

    # Mock supabase rpc
    def fake_rpc(payload):
        class R:
            def execute(self):
                return type(
                    "X",
                    (),
                    {
                        "data": [
                            {
                                "id": 1,
                                "document_name": "DocA",
                                "document_version": "v1",
                                "chunk_text": "A",
                                "similarity": 0.9,
                            },
                            {
                                "id": 2,
                                "document_name": "DocB",
                                "document_version": "v2",
                                "chunk_text": "B",
                                "similarity": 0.8,
                            },
                        ],
                    },
                )

        return R()

    class FakeSupabase:
        def rpc(self, name, payload):
            return fake_rpc(payload)

    monkeypatch.setattr(
        retriever,
        "get_supabase_client",
        lambda: FakeSupabase(),
    )
    monkeypatch.setattr(
        retriever,
        "extract_query_filters",
        lambda _: retriever.QueryFilters(),
    )

    # Call retrieve_context (without LLM rerank to simplify)
    results, _filters = retriever.retrieve_context(
        "user_query",
        "client_123",
        rerank_with_model=False,
    )
    assert isinstance(results, list)
    assert len(results) <= retriever.settings.top_k

def test_version_comparison_retrieval(monkeypatch) -> None:
    monkeypatch.setenv("TOP_K", "2")

    # Mock embedding
    monkeypatch.setattr(retriever, "embed_texts", lambda q: [[0.1] for _ in q])
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1])
    
    # Mock supabase rpc to return chunks from both v1 and v2
    match_thresholds: list[float] = []

    def fake_rpc(*args, **kwargs):
        match_thresholds.append(kwargs.get("match_threshold", 0.0))
        return [
            {"id": 1, "document_name": "A", "document_version": "v1", "chunk_text": "v1 text", "similarity": 0.5},
            {"id": 2, "document_name": "A", "document_version": "v1", "chunk_text": "v1 text", "similarity": 0.4},
            {"id": 3, "document_name": "A", "document_version": "v2", "chunk_text": "v2 text", "similarity": 0.1},
        ]
    
    monkeypatch.setattr(retriever, "call_match_documents", fake_rpc)
    monkeypatch.setattr(
        retriever,
        "extract_query_filters",
        lambda _: retriever.QueryFilters(is_comparison=True),
    )
    
    # Ask a version comparison query
    results, _filters = retriever.retrieve_context(
        "What changed between v1 and v2?",
        "client_1",
        rerank_with_model=False
    )

    assert _filters.is_comparison
    assert match_thresholds and all(threshold == 0.0 for threshold in match_thresholds)
    
    # Ensure chunks from BOTH versions are represented, despite v2 having low similarity (0.1)
    versions = set([c.get("document_version") for c in results])
    assert "v1" in versions
    assert "v2" in versions
