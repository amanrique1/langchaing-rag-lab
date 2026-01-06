import json
import numpy as np
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.application.ports.chunk_store import ChunkStore
from src.application.ports.embedding_model import EmbeddingModel
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rag_docs"
DEFAULT_PERSIST_DIRECTORY = "./filesystem_db"


class FileSystemChunkStore(ChunkStore):
    """
    A persistent chunk store backed by the local file system using JSON files.

    **Architecture:**
    Each chunk is saved as an individual `.json` file containing its ID, content,
    metadata, and pre-computed vector embedding.

    **Dual Collection Mode:**
    If enabled, maintains two parallel directories:
    1. `/content`: Stores the actual document chunks.
    2. `/metadata`: Stores text summaries of the metadata.

    Both search modes return COMPLETE chunks with content populated.
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_model: EmbeddingModel = None,
        persist_directory: str = None,
        **kwargs
    ):
        """
        Initialize the file system chunk store.

        Args:
            collection_name (str, optional): Name used for subdirectory organization.
            embedding_model (EmbeddingModel, optional): The model used to generate embeddings.
            persist_directory (str, optional): The root directory path. None = use default.
            **kwargs: Additional parameters:
                - dual_collection (bool): If True, creates separate 'content' and 'metadata' 
                  directories. Defaults to True.
        """
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.persist_directory = Path(persist_directory or DEFAULT_PERSIST_DIRECTORY)
        self.embedding_model = embedding_model
        self.dual_collection = kwargs.get('dual_collection', True)

        self.output_dir = self.persist_directory / self.collection_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dual_collection:
            self.content_dir = self.output_dir / "content"
            self.metadata_dir = self.output_dir / "metadata"
            self.content_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.content_dir = self.output_dir
            self.metadata_dir = None

        logger.info(f"FileSystemChunkStore initialized with collection: {self.collection_name} at {self.persist_directory}")

    def _get_file_path(self, directory: Path, chunk_id: str) -> Path:
        """Generates a safe file path for a given chunk ID."""
        safe_id = chunk_id.replace("/", "_").replace("\\", "_")
        return directory / f"{safe_id}.json"

    def save(self, chunks: List[Chunk]) -> None:
        """
        Persists a list of chunks to the file system.

        Pre-computes vector embeddings for both content and metadata
        and saves them into the JSON file.

        Args:
            chunks (List[Chunk]): The list of domain objects to save.
        """
        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None) or str(hash(chunk.content))

            # Pre-compute Content Embedding
            embedding = None
            if self.embedding_model:
                emb_vector = self.embedding_model.embed_query(chunk.content)
                if isinstance(emb_vector, np.ndarray):
                    embedding = emb_vector.tolist()
                else:
                    embedding = emb_vector

            content_data = {
                "id": chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "embedding": embedding
            }

            content_file = self._get_file_path(self.content_dir, chunk_id)
            with open(content_file, "w", encoding="utf-8") as f:
                json.dump(content_data, f, indent=2)

            # Save Metadata (Dual Mode)
            if self.dual_collection and self.metadata_dir:
                metadata_summary = self._create_metadata_summary(chunk)

                meta_embedding = None
                if self.embedding_model:
                    meta_vector = self.embedding_model.embed_query(metadata_summary)
                    if isinstance(meta_vector, np.ndarray):
                        meta_embedding = meta_vector.tolist()
                    else:
                        meta_embedding = meta_vector

                metadata_data = {
                    "id": chunk_id,
                    "content": metadata_summary,
                    "metadata": chunk.metadata,
                    "embedding": meta_embedding,
                    "original_chunk_id": chunk_id  # Link back to content
                }

                metadata_file = self._get_file_path(self.metadata_dir, chunk_id)
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata_data, f, indent=2)

        logger.info(f"Saved {len(chunks)} chunks to filesystem")

    def _create_metadata_summary(self, chunk: Chunk) -> str:
        """Transforms a metadata dictionary into a searchable string format."""
        metadata_parts = []
        for key, value in chunk.metadata.items():
            metadata_parts.append(f"{key}: {value}")
        return " | ".join(metadata_parts)

    def delete(self, chunk_id: str, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> None:
        """Deletes a specific chunk and its corresponding metadata file from storage."""
        content_file = self._get_file_path(self.content_dir, chunk_id)
        if content_file.exists():
            content_file.unlink()
            logger.debug(f"Deleted content file for chunk: {chunk_id}")

        if self.dual_collection and self.metadata_dir:
            metadata_file = self._get_file_path(self.metadata_dir, chunk_id)
            if metadata_file.exists():
                metadata_file.unlink()
                logger.debug(f"Deleted metadata file for chunk: {chunk_id}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> List[SearchResult]:
        """
        Performs a search over stored JSON files.

        IMPORTANT: Both modes return COMPLETE chunks with content populated.
        - content mode: Searches content directory, returns full chunks
        - metadata mode: Searches metadata directory, then loads content from content directory

        Args:
            query (str): The search text.
            top_k (int): Max results to return.
            filter (Optional[Dict]): Exact-match metadata filters.
            mode (str): 'content' to search document text, 'metadata' to search metadata summaries.

        Returns:
            List[SearchResult]: Results with complete chunks.
        """
        if not self.embedding_model:
            # Fallback to keyword search
            chunks = self._keyword_search(query, top_k)
            return [
                SearchResult(chunk=chunk, score=0.5, rank=i+1, retrieval_method="keyword")
                for i, chunk in enumerate(chunks)
            ]

        # Determine target directory for search
        search_dir = self.metadata_dir if (mode == "metadata" and self.metadata_dir) else self.content_dir

        # Embed query once
        query_embedding = self.embedding_model.embed_query(query)

        raw_results = []
        retrieval_method = f"semantic_{mode}"

        for file_path in search_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    if filter and not self._matches_filter(data["metadata"], filter):
                        continue

                    chunk_embedding = data.get("embedding")
                    if not chunk_embedding:
                        continue

                    # Calculate Score
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)

                    raw_results.append({
                        "chunk_id": data["id"],
                        "score": float(similarity),
                        "data": data
                    })
            except Exception as e:
                logger.warning(f"Failed to process file {file_path}: {e}")
                continue

        # Sort descending by score
        raw_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = raw_results[:top_k]

        # Build final results with COMPLETE chunks
        final_results = []

        if mode == "metadata":
            # For metadata search, we need to load content from content directory
            chunk_ids = [res["chunk_id"] for res in top_results]
            complete_chunks_map = {c.chunk_id: c for c in self.get_by_ids(chunk_ids)}

            for rank, res in enumerate(top_results, start=1):
                chunk_id = res["chunk_id"]
                complete_chunk = complete_chunks_map.get(chunk_id)

                if complete_chunk:
                    final_results.append(SearchResult(
                        chunk=complete_chunk,
                        score=1.0 / (1.0 + res["score"]),
                        retrieval_method=retrieval_method,
                        rank=rank
                    ))
        else:
            # Content search - data already contains full content
            for rank, res in enumerate(top_results, start=1):
                data = res["data"]
                chunk = Chunk(content=data["content"], metadata=data["metadata"])
                chunk.chunk_id = data["id"]

                final_results.append(SearchResult(
                    chunk=chunk,
                    score=1.0 / (1.0 + res["score"]),
                    retrieval_method=retrieval_method,
                    rank=rank
                ))

        return final_results

    def _keyword_search(self, query: str, top_k: int) -> List[Chunk]:
        """Performs a naive case-insensitive substring search over the JSON files."""
        relevant_chunks = []
        for file_path in self.content_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if query.lower() in data["content"].lower():
                        chunk = Chunk(content=data["content"], metadata=data["metadata"])
                        chunk.chunk_id = data["id"]
                        relevant_chunks.append(chunk)
                        if len(relevant_chunks) >= top_k:
                            break
            except Exception:
                continue
        return relevant_chunks

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """Checks if the metadata dictionary matches the provided filter criteria."""
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computes the cosine similarity between two embedding vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieves complete chunks by their IDs from the content directory.

        This is highly efficient - O(k) where k is the number of requested IDs.
        Direct file access by ID without any scanning.

        Args:
            chunk_ids (List[str]): The list of IDs to fetch.

        Returns:
            List[Chunk]: Complete chunks with content and metadata.
        """
        chunks = []
        for chunk_id in chunk_ids:
            file_path = self._get_file_path(self.content_dir, chunk_id)
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunk = Chunk(content=data["content"], metadata=data["metadata"])
                        chunk.chunk_id = data["id"]
                        chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to load chunk {chunk_id}: {e}")

        logger.debug(f"Retrieved {len(chunks)}/{len(chunk_ids)} chunks by ID")
        return chunks

    def clear(self) -> None:
        """Destructively clears the entire chunk store."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleared filesystem store: {self.output_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.dual_collection:
            self.content_dir.mkdir(parents=True, exist_ok=True)
            if self.metadata_dir:
                self.metadata_dir.mkdir(parents=True, exist_ok=True)