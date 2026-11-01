"""
Validates ChromaVectorStore against the real, ingested knowledge base.

Unlike test_rag_pipeline.py (which uses PolicyRetriever()'s default
MockVectorStore), this test explicitly wires PolicyRetriever to the
real ChromaVectorStore so we're testing actual Chroma behavior.

Run from the backend directory:
    python -m pytest tests/test_chroma_retrieval_real.py -v
"""

from app.rag.chroma_store import ChromaVectorStore
from app.rag.retriever import PolicyRetriever
from app.rag.embeddings import get_embedding_provider


def get_chroma_retriever() -> PolicyRetriever:
    """Builds a PolicyRetriever backed by the real ChromaVectorStore."""
    store = ChromaVectorStore(embedding_provider=get_embedding_provider())
    return PolicyRetriever(vector_store=store)


def test_exact_policy_query_retrieval_chroma():
    """Verify exact policy query returns correct document with high relevance."""
    retriever = get_chroma_retriever()
    query = "How do I challenge an error in continuous assessment marks?"
    sources = retriever.retrieve(query, department="Academic Affairs", limit=3)
    assert len(sources) > 0
    assert sources[0].document_id == "DOC-ACAD-001"
    assert "Clause 4.1" in sources[0].clause
    assert sources[0].relevance_score >= 0.70


def test_version_selection_prefers_latest_approved_chroma():
    """Verify superseded version (v1.0) is ignored when active v2.0 exists."""
    retriever = get_chroma_retriever()
    sources = retriever.retrieve(
        "continuous assessment grievance timeline",
        department="Academic Affairs",
        limit=5
    )
    for s in sources:
        if s.document_id == "DOC-ACAD-001":
            assert s.document_version == "2.0"


def test_expired_and_draft_documents_excluded_chroma():
    """Verify expired and draft documents are excluded from active retrieval."""
    retriever = get_chroma_retriever()
    sources = retriever.retrieve("test policy document", limit=10)
    retrieved_ids = [s.document_id for s in sources]
    assert "DOC-TEST-EXP-099" not in retrieved_ids
    assert "DOC-TEST-DFT-098" not in retrieved_ids