from src.app.retrieval import HybridRetriever


def test_hybrid_retriever_handles_empty_store():
    retriever = HybridRetriever()
    results = retriever.retrieve("Test question")
    assert isinstance(results, list)
