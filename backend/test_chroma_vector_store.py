
"""
Standalone smoke test for ChromaVectorStore.

Run from the repository root:
    python test_chroma_vector_store.py

This test:
1. Creates three fake KnowledgeChunk objects.
2. Inserts them into ChromaDB.
3. Searches for matching content.
4. Verifies ranking and version filtering.
5. Retains the test database to avoid Windows cleanup issues.
"""

import shutil
from pathlib import Path
from typing import List

from app.rag.chroma_store import ChromaVectorStore
from app.schemas.contracts import KnowledgeChunk


# Fixed directory: avoids TemporaryDirectory cleanup while
# ChromaDB's SQLite/HNSW files may still be locked on Windows.
TEST_DATA_DIR = Path("./test_chroma_data")


class FakeEmbeddingProvider:
    """Deterministic fake embeddings; no external API required."""

    dimension = 3

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []

        for text in texts:
            text = text.lower()

            if "leave" in text or "vacation" in text:
                embeddings.append([1.0, 0.0, 0.0])
            elif "expense" in text or "reimbursement" in text:
                embeddings.append([0.0, 1.0, 0.0])
            else:
                embeddings.append([0.0, 0.0, 1.0])

        return embeddings


def make_chunk(
    chunk_id: str,
    document_id: str,
    title: str,
    content: str,
    version: str = "1.0",
) -> KnowledgeChunk:
    """Build a KnowledgeChunk for the smoke test."""

    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=title,
        document_version=version,
        section_id="section-1",
        heading=title,
        clause="1.1",
        department_id="HR",
        department="HR",
        document_type="POLICY",
        approval_status="APPROVED",
        effective_date="2025-01-01",
        expiry_date=None,
        source="smoke-test",
        content=content,
        excerpt=content[:100],
        keywords=["test"],
        token_count=len(content.split()),
    )


def main():
    print("=" * 60)
    print("ChromaVectorStore smoke test")
    print("=" * 60)

    # Remove stale test data BEFORE opening ChromaDB.
    # Do not delete this directory while the client is active.
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

    embedding_provider = FakeEmbeddingProvider()

    chunks = [
        make_chunk(
            chunk_id="chunk-leave",
            document_id="policy-leave",
            title="Leave Policy",
            content="Employees can request annual leave and vacation.",
        ),
        make_chunk(
            chunk_id="chunk-expense",
            document_id="policy-expense",
            title="Expense Policy",
            content="Employees can request expense reimbursement.",
        ),
        make_chunk(
            chunk_id="chunk-travel",
            document_id="policy-travel",
            title="Travel Policy",
            content="Business travel requires manager approval.",
        ),
    ]

    store = ChromaVectorStore(
        embedding_provider=embedding_provider,
        collection_name="smoke_test_chunks",
        persist_directory=str(TEST_DATA_DIR.resolve()),
    )

    # ---------------------------------------------------------
    # Test 1: Add chunks
    # ---------------------------------------------------------
    print("\n[1] Adding chunks...")

    success = store.add_chunks(chunks)

    assert success is True, "add_chunks() did not return True"

    actual_count = store.collection.count()

    assert actual_count == 3, (
        f"Expected 3 chunks, got {actual_count}"
    )

    print(f"PASS: Added {actual_count} chunks.")

    # ---------------------------------------------------------
    # Test 2: Search for leave policy
    # ---------------------------------------------------------
    print("\n[2] Searching for leave policy...")

    query = "annual leave vacation"

    query_vector = embedding_provider.embed_batch([query])[0]

    results = store.search_by_vector(
        query_vector=query_vector,
        query_text=query,
        top_k=3,
        active_only=True,
        min_threshold=0.0,
    )

    assert results, "search_by_vector() returned no results"

    print(f"PASS: Search returned {len(results)} result(s).")

    for index, (chunk, score) in enumerate(results, start=1):
        print(f"\nResult {index}")
        print(f"  Document: {chunk.document_title}")
        print(f"  Document ID: {chunk.document_id}")
        print(f"  Version: {chunk.document_version}")
        print(f"  Approval: {chunk.approval_status}")
        print(f"  Score: {score}")
        print(f"  Content: {chunk.content}")

    # ---------------------------------------------------------
    # Test 3: Verify relevance ranking
    # ---------------------------------------------------------
    print("\n[3] Checking ranking...")

    first_chunk = results[0][0]

    assert first_chunk.document_id == "policy-leave", (
        "Expected leave policy to rank first, "
        f"but got {first_chunk.document_id}"
    )

    print("PASS: Leave policy ranked first.")

    # ---------------------------------------------------------
    # Test 4: Explicit version filtering
    # ---------------------------------------------------------
    print("\n[4] Testing version filtering...")

    version_results = store.search_by_vector(
        query_vector=query_vector,
        query_text=query,
        top_k=3,
        active_only=True,
        version="1.0",
        min_threshold=0.0,
    )

    assert version_results, "Version-filtered search returned no results"

    assert all(
        chunk.document_version == "1.0"
        for chunk, _ in version_results
    ), "Version filter returned an unexpected version"

    print("PASS: Explicit version filter works.")

    # ---------------------------------------------------------
    # Test 5: Dictionary-based search interface
    # ---------------------------------------------------------
    print("\n[5] Testing search() dictionary interface...")

    dictionary_results = store.search(
        query=query,
        limit=3,
        department="HR",
        min_threshold=0.0,
    )

    assert dictionary_results, "search() returned no results"

    required_keys = {
        "document_id",
        "title",
        "document_version",
        "relevance_score",
        "content",
    }

    for result in dictionary_results:
        missing_keys = required_keys - result.keys()

        assert not missing_keys, (
            f"Search result is missing keys: {missing_keys}"
        )

    print(
        "PASS: search() returned dictionaries with the expected keys."
    )

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)

    print(f"\nTest database retained at: {TEST_DATA_DIR.resolve()}")
    print(
        "The database was not deleted during cleanup. "
        "Close Python before removing it."
    )


if __name__ == "__main__":
    main()
