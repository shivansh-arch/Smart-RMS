"""
One-time ingestion script: loads real policy documents and indexes
them into ChromaVectorStore.

Run from the backend directory:
    python ingest_to_chroma.py
"""

import json
from pathlib import Path

from app.config import settings
from app.rag.chunker import SemanticDocumentChunker
from app.rag.chroma_store import ChromaVectorStore
from app.rag.embeddings import get_embedding_provider


def main():
    print("=" * 60)
    print("Ingesting knowledge_documents.json into ChromaDB")
    print("=" * 60)

    # Step 1: Load the raw documents.
    doc_path = settings.MOCK_DATA_DIR / "knowledge_documents.json"
    print(f"\n[1] Loading documents from: {doc_path}")

    with open(doc_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"    Found {len(documents)} documents.")

    # Step 2: Chunk every document.
    print("\n[2] Chunking documents...")

    all_chunks = []
    for doc in documents:
        chunks = SemanticDocumentChunker.chunk_document(doc)
        all_chunks.extend(chunks)

    print(f"    Produced {len(all_chunks)} chunks.")

    # Step 3: Build the real ChromaVectorStore.
    print("\n[3] Connecting to ChromaDB...")

    store = ChromaVectorStore(
        embedding_provider=get_embedding_provider()
    )

    print(f"    Existing chunk count before ingestion: {store.collection.count()}")

    # Step 4: Add all chunks.
    print("\n[4] Adding chunks to ChromaDB (this embeds every chunk)...")

    success = store.add_chunks(all_chunks)

    if not success:
        print("    FAILED: add_chunks() returned False.")
        return

    final_count = store.collection.count()
    print(f"    Done. Collection now holds {final_count} chunks.")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()