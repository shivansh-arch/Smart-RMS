
"""
Persistent ChromaDB vector store for Smart RMS.

Features:
- Add and update knowledge chunks
- Vector similarity search
- Active policy and version filtering
- Hybrid dense and lexical scoring
- Dictionary results compatible with PolicyRetriever
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import chromadb

from app.config import settings
from app.rag.vector_store import VectorStore
from app.schemas.contracts import KnowledgeChunk


logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent vector store backed by ChromaDB."""

    def __init__(
        self,
        embedding_provider,
        collection_name: str = "knowledge_chunks",
        persist_directory: Optional[str] = None,
    ):
        self.embedding_provider = embedding_provider

        configured_directory = getattr(
            settings,
            "CHROMA_PERSIST_DIRECTORY",
            "./chroma_data",
        )

        self.persist_directory = str(
            Path(persist_directory or configured_directory)
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------------------------------------------------------
    # Add chunks
    # ---------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> bool:
        """Add or update chunks in ChromaDB."""

        if not chunks:
            return True

        if embeddings is None:
            texts = [
                (
                    f"{chunk.document_title} "
                    f"{chunk.heading} "
                    f"{chunk.clause} "
                    f"{chunk.content}"
                )
                for chunk in chunks
            ]

            embeddings = self.embedding_provider.embed_batch(texts)

        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number "
                "of chunks."
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.content)

            metadata = {
                "document_id": chunk.document_id,
                "document_title": chunk.document_title,
                "document_version": chunk.document_version,
                "section_id": chunk.section_id or "",
                "heading": chunk.heading or "",
                "clause": chunk.clause or "",
                "department_id": chunk.department_id or "",
                "department": chunk.department or "",
                "document_type": chunk.document_type or "",
                "approval_status": chunk.approval_status or "",
                # These fields are strings in KnowledgeChunk.
                "effective_date": chunk.effective_date or "",
                "expiry_date": chunk.expiry_date or "",
                "source": chunk.source or "",
                "excerpt": chunk.excerpt or "",
                "keywords": json.dumps(chunk.keywords or []),
                "token_count": int(chunk.token_count or 0),
            }

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info("Upserted %d chunks into ChromaDB", len(chunks))
        return True

    # ---------------------------------------------------------
    # Search by vector
    # ---------------------------------------------------------

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        department: Optional[str] = None,
        active_only: bool = True,
        version: Optional[str] = None,
        min_threshold: float = 0.0,
        query_text: Optional[str] = None,
    ) -> List[Tuple[KnowledgeChunk, float]]:
        """Search by embedding and return (chunk, score) tuples."""

        if top_k <= 0:
            return []

        total_chunks = self.collection.count()

        if total_chunks == 0:
            return []

        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=total_chunks,
            include=["metadatas", "documents", "distances"],
        )

        ids = response["ids"][0]
        metadatas = response["metadatas"][0]
        documents = response["documents"][0]
        distances = response["distances"][0]

        # Find the latest approved version of each document.
        latest_versions = {}

        for metadata in metadatas:
            if not metadata:
                continue

            if metadata.get("approval_status") != "APPROVED":
                continue

            document_id = metadata.get("document_id")
            document_version = metadata.get("document_version")

            if not document_id or not document_version:
                continue

            current_version = latest_versions.get(document_id)

            if (
                current_version is None
                or document_version > current_version
            ):
                latest_versions[document_id] = document_version

        today = datetime.now(timezone.utc).date().isoformat()

        query_words = (
            set(query_text.lower().split())
            if query_text
            else set()
        )

        department_normalized = (
            department.lower().strip()
            if department
            else None
        )

        candidates = []

        for chunk_id, metadata, content, distance in zip(
            ids,
            metadatas,
            documents,
            distances,
        ):
            if metadata is None or content is None or distance is None:
                continue

            try:
                document_id = metadata["document_id"]
                document_version = metadata.get(
                    "document_version",
                    "1.0",
                )

                approval_status = metadata.get(
                    "approval_status",
                    "",
                )

                effective_date = metadata.get(
                    "effective_date",
                    "",
                )

                expiry_date = metadata.get(
                    "expiry_date",
                    "",
                )

                # Filter inactive or unapproved policies.
                if active_only:
                    if approval_status != "APPROVED":
                        continue

                    if effective_date and effective_date > today:
                        continue

                    if expiry_date and expiry_date < today:
                        continue

                    if version is None:
                        latest_version = latest_versions.get(document_id)

                        if (
                            latest_version is not None
                            and document_version != latest_version
                        ):
                            continue

                # Filter to a requested version if supplied.
                if (
                    version is not None
                    and document_version != version
                ):
                    continue

                stored_department = str(
                    metadata.get("department", "")
                ).lower().strip()

                if (
                    department_normalized
                    and stored_department
                    and department_normalized not in stored_department
                    and stored_department not in department_normalized
                ):
                    continue

                # Dense similarity from cosine distance.
                raw_similarity = 1.0 - float(distance)
                score = max(0.0, raw_similarity) * 0.72

                # Lexical matching bonus.
                if query_words:
                    searchable_text = " ".join(
                        [
                            str(metadata.get("document_title", "")),
                            str(metadata.get("heading", "")),
                            str(metadata.get("clause", "")),
                            str(content),
                            str(metadata.get("keywords", "")),
                        ]
                    ).lower()

                    matching_words = sum(
                        1
                        for word in query_words
                        if len(word) > 3 and word in searchable_text
                    )

                    score += min(matching_words * 0.09, 0.25)

                # Department relevance bonus.
                if (
                    department_normalized
                    and stored_department
                    and (
                        department_normalized in stored_department
                        or stored_department in department_normalized
                    )
                ):
                    score = min(score + 0.18, 0.98)

                relevance_score = round(min(score, 0.99), 4)

                if relevance_score < min_threshold:
                    continue

                chunk = self._metadata_to_chunk(
                    chunk_id=chunk_id,
                    metadata=metadata,
                    content=content,
                )

                candidates.append((chunk, relevance_score))

            except (KeyError, TypeError, ValueError) as exc:
                logger.warning(
                    "Skipping malformed ChromaDB record %s: %s",
                    chunk_id,
                    exc,
                )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = []
        seen = set()

        for chunk, relevance_score in candidates:
            key = (chunk.document_id, chunk.clause)

            if key in seen:
                continue

            seen.add(key)
            results.append((chunk, relevance_score))

            if len(results) >= top_k:
                break

        return results

    # ---------------------------------------------------------
    # Search by text
    # ---------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        department: Optional[str] = None,
        min_threshold: float = 0.0,
        version: Optional[str] = None,
        active_only: bool = True,
    ) -> List[dict]:
        """
        Search using text and return dictionaries for PolicyRetriever.

        The limit and min_threshold parameters match the caller's
        keyword arguments in retriever.py.
        """

        if not query or not query.strip() or limit <= 0:
            return []

        query_embedding = self.embedding_provider.embed_batch([query])[0]

        ranked_chunks = self.search_by_vector(
            query_vector=query_embedding,
            query_text=query,
            top_k=limit,
            department=department,
            active_only=active_only,
            version=version,
            min_threshold=min_threshold,
        )

        results = []

        for chunk, relevance_score in ranked_chunks:
            results.append(
                {
                    "id": chunk.chunk_id,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "title": chunk.document_title,
                    "document_title": chunk.document_title,
                    "document_version": chunk.document_version,
                    "section_id": chunk.section_id,
                    "heading": chunk.heading,
                    "clause": chunk.clause,
                    "department_id": chunk.department_id,
                    "department": chunk.department,
                    "document_type": chunk.document_type,
                    "approval_status": chunk.approval_status,
                    "effective_date": chunk.effective_date,
                    "expiry_date": chunk.expiry_date,
                    "source": chunk.source,
                    "content": chunk.content,
                    "excerpt": chunk.excerpt or chunk.content[:200],
                    "keywords": chunk.keywords,
                    "relevance_score": relevance_score,
                }
            )

        return results

    # ---------------------------------------------------------
    # Reconstruct KnowledgeChunk
    # ---------------------------------------------------------

    @staticmethod
    def _metadata_to_chunk(
        chunk_id: str,
        metadata: dict,
        content: str,
    ) -> KnowledgeChunk:
        """Reconstruct a KnowledgeChunk from ChromaDB metadata."""

        keywords_value = metadata.get("keywords") or "[]"

        if isinstance(keywords_value, str):
            keywords = json.loads(keywords_value)
        elif isinstance(keywords_value, list):
            keywords = keywords_value
        else:
            keywords = []

        return KnowledgeChunk(
            chunk_id=chunk_id,
            document_id=metadata["document_id"],
            document_title=metadata.get("document_title", ""),
            document_version=metadata.get(
                "document_version",
                "1.0",
            ),
            section_id=metadata.get("section_id") or "",
            heading=metadata.get("heading") or "",
            clause=metadata.get("clause") or "",
            department_id=metadata.get("department_id") or None,
            department=metadata.get("department") or "",
            document_type=metadata.get("document_type", "POLICY"),
            approval_status=metadata.get("approval_status", ""),
            effective_date=metadata.get("effective_date") or "",
            expiry_date=metadata.get("expiry_date") or None,
            source=metadata.get("source") or "",
            content=content,
            excerpt=metadata.get("excerpt") or None,
            keywords=keywords,
            token_count=int(metadata.get("token_count") or 0),
        )