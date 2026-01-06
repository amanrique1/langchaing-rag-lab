import os
import shutil
import hashlib
import logging
from typing import List, Dict, Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.application.ports.chunk_store import ChunkStore
from src.application.ports.embedding_model import EmbeddingModel
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult
from src.domain.services.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rag_docs"
DEFAULT_PERSIST_DIRECTORY = "./chroma_db"


class ChromaChunkStore(ChunkStore):
    """
    A concrete implementation of ChunkStore using ChromaDB.

    This store implements a 'Dual Collection' strategy:
    1. Content Collection: Vectors derived from the raw text content.
    2. Metadata Collection: Vectors derived from a string representation of metadata.

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
        Initialize the ChromaChunkStore with embedding capabilities.

        Args:
            collection_name (str, optional): The base name for the collections. Defaults to 'rag_docs'.
            embedding_model (EmbeddingModel): The model used to generate vectors. Required.
            persist_directory (str, optional): Directory to persist the data. Defaults to './chroma_db'.
            **kwargs: Additional parameters:
                - dual_collection (bool): If True, creates secondary collection for metadata. Default True.

        Raises:
            ValueError: If embedding_model is not provided.
        """
        if not embedding_model:
            raise ValueError("An embedding model must be provided to initialize ChromaChunkStore.")

        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIRECTORY
        self.dual_collection = kwargs.get('dual_collection', True)
        self._embeddings = embedding_model

        # Lazy loading placeholders
        self._content_vector_store: Optional[Chroma] = None
        self._metadata_vector_store: Optional[Chroma] = None

        logger.info(f"ChromaChunkStore initialized with collection: {self.collection_name}")

    @property
    def content_collection(self) -> Chroma:
        """
        Lazily initializes and returns the primary content collection.

        Returns:
            Chroma: The LangChain Chroma object for the content collection.
        """
        if self._content_vector_store is None:
            self._content_vector_store = Chroma(
                collection_name=f"{self.collection_name}_content",
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings,
            )
        return self._content_vector_store

    @property
    def metadata_collection(self) -> Optional[Chroma]:
        """
        Lazily initializes and returns the secondary metadata collection.

        Returns:
            Optional[Chroma]: The LangChain Chroma object, or None if dual_collection is False.
        """
        if not self.dual_collection:
            return None

        if self._metadata_vector_store is None:
            self._metadata_vector_store = Chroma(
                collection_name=f"{self.collection_name}_metadata",
                persist_directory=self.persist_directory,
                embedding_function=self._embeddings,
            )
        return self._metadata_vector_store

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes metadata dictionaries to ensure compatibility with ChromaDB's requirements.

        ChromaDB only accepts str, int, float, or bool for metadata values.
        Lists and NoneTypes must be converted.

        Args:
            metadata (Dict[str, Any]): The raw metadata dictionary.

        Returns:
            Dict[str, Any]: A flat, type-safe dictionary.
        """
        filtered = {}
        for key, value in metadata.items():
            if value is None:
                filtered[key] = ""
            elif isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, list):
                filtered[key] = ", ".join(str(v) for v in value)
            else:
                filtered[key] = str(value)
        return filtered

    @staticmethod
    def _generate_stable_id(content: str) -> str:
        """
        Generates a deterministic MD5 hash for the given string content.

        Used to ensure that re-ingesting the same text results in the same ID,
        preventing duplicates.

        Args:
            content (str): The text content to hash.

        Returns:
            str: The hexadecimal MD5 hash.
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def save(self, chunks: List[Chunk]) -> None:
        """
        Persists a list of chunks into the vector database.

        If dual_collection is enabled, this splits the chunk into:
        1. A content document (indexed by content text).
        2. A metadata document (indexed by metadata text string).

        Args:
            chunks (List[Chunk]): The domain chunks to save.
        """
        content_docs = []
        content_ids = []
        metadata_docs = []
        metadata_ids = []
        has_metadata_collection = self.metadata_collection is not None

        for chunk in chunks:
            # 1. ID Generation
            c_id = getattr(chunk, "chunk_id", None) or self._generate_stable_id(chunk.content)
            content_ids.append(c_id)

            # 2. Prepare Content Document
            safe_metadata = self._sanitize_metadata(chunk.metadata)
            safe_metadata['chunk_id'] = c_id
            content_docs.append(
                Document(page_content=chunk.content, metadata=safe_metadata)
            )

            # 3. Prepare Metadata Document
            if has_metadata_collection:
                # Convert metadata dict to a searchable string
                metadata_text = MetadataManager.create_searchable_string(chunk.metadata)

                # Store chunk_id to link back to content
                shadow_metadata = {
                    "chunk_id": c_id,
                    "is_metadata_doc": True,
                    "original_filename": chunk.metadata.get('filename', '')
                }

                safe_shadow_metadata = self._sanitize_metadata(shadow_metadata)

                metadata_docs.append(
                    Document(page_content=metadata_text, metadata=safe_shadow_metadata)
                )
                metadata_ids.append(f"{c_id}_meta")

        # 4. Batch Upload
        self.content_collection.add_documents(documents=content_docs, ids=content_ids)

        if has_metadata_collection and metadata_docs:
            self.metadata_collection.add_documents(documents=metadata_docs, ids=metadata_ids)

        logger.info(f"Saved {len(chunks)} chunks to ChromaDB")

    def delete(self, chunk_id: str, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> None:
        """
        Deletes a chunk and its associated metadata entry from the database.

        Args:
            chunk_id (str): The ID of the chunk to remove.
            where (dict, optional): Metadata filter for deletion.
            where_document (dict, optional): Content filter for deletion.
        """
        self.content_collection.delete(ids=[chunk_id], where=where, where_document=where_document)

        if self.metadata_collection:
            self.metadata_collection.delete(ids=[f"{chunk_id}_meta"], where=where, where_document=where_document)

        logger.debug(f"Deleted chunk: {chunk_id}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> List[SearchResult]:
        """
        Performs a semantic search and returns typed results with normalized scores.

        IMPORTANT: Both modes return COMPLETE chunks with content populated.
        - content mode: Searches content collection, returns full chunks
        - metadata mode: Searches metadata collection, then hydrates with content from content collection

        Args:
            query (str): The search string.
            top_k (int): Number of results to retrieve.
            filter (Optional[Dict]): Metadata filters to apply.
            mode (str): 'content' to search chunk text, 'metadata' to search metadata text.

        Returns:
            List[SearchResult]: Results with complete chunks and scores normalized to 0.0-1.0.
        """
        # 1. Select Collection
        if mode == "content":
            collection = self.content_collection
        elif mode == "metadata":
            collection = self.metadata_collection
        else:
            raise ValueError(f"Invalid search mode: '{mode}'. Must be 'content' or 'metadata'.")

        # 2. Safety Checks
        if collection is None:
            if mode == "metadata":
                logger.warning("Metadata collection not available, returning empty results")
                return []
            raise ValueError("Content collection is not initialized.")

        # 3. Perform Search
        try:
            docs_with_scores = collection.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter
            )
        except Exception as e:
            logger.error(f"Search failed in {mode} mode: {e}")
            return []

        results = []
        retrieval_method = f"semantic_{mode}"

        # 4. If metadata search, we need to hydrate with actual content
        if mode == "metadata":
            # Extract chunk_ids from metadata search results
            chunk_ids = []
            scores_map = {}
            ranks_map = {}

            for rank, (doc, raw_score) in enumerate(docs_with_scores, start=1):
                chunk_id = doc.metadata.get('chunk_id')
                if chunk_id:
                    chunk_ids.append(chunk_id)
                    scores_map[chunk_id] = raw_score
                    ranks_map[chunk_id] = rank

            # Batch fetch complete chunks from content collection
            if chunk_ids:
                complete_chunks = self.get_by_ids(chunk_ids)

                # Build results with complete chunks
                for chunk in complete_chunks:
                    chunk_id = getattr(chunk, 'chunk_id', None)
                    if chunk_id and chunk_id in scores_map:
                        raw_score = scores_map[chunk_id]
                        rank = ranks_map[chunk_id]
                        similarity_score = 1.0 / (1.0 + raw_score)

                        results.append(SearchResult(
                            chunk=chunk,
                            score=similarity_score,
                            retrieval_method=retrieval_method,
                            rank=rank
                        ))

                # Sort by original rank
                results.sort(key=lambda x: x.rank)
        else:
            # Content search - already have complete chunks
            for rank, (doc, raw_score) in enumerate(docs_with_scores, start=1):
                chunk = Chunk(content=doc.page_content, metadata=doc.metadata)
                chunk.chunk_id = doc.metadata.get('chunk_id', self._generate_stable_id(doc.page_content))

                similarity_score = 1.0 / (1.0 + raw_score)

                results.append(SearchResult(
                    chunk=chunk,
                    score=similarity_score,
                    retrieval_method=retrieval_method,
                    rank=rank
                ))

        return results

    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieves complete chunks by their IDs using ChromaDB's optimized .get() method.

        Args:
            chunk_ids (List[str]): The IDs to fetch.

        Returns:
            List[Chunk]: Complete chunks with content and metadata.
        """
        if not chunk_ids:
            return []

        try:
            result_dict = self.content_collection.get(ids=chunk_ids)

            chunks = []
            if result_dict and 'documents' in result_dict and result_dict['documents']:
                documents = result_dict['documents']
                metadatas = result_dict.get('metadatas', [{} for _ in documents])
                ids = result_dict.get('ids', [])

                for i, content in enumerate(documents):
                    meta = metadatas[i] if metadatas[i] is not None else {}
                    chunk = Chunk(content=content, metadata=meta)
                    chunk.chunk_id = ids[i]
                    chunks.append(chunk)

            logger.debug(f"Retrieved {len(chunks)}/{len(chunk_ids)} chunks by ID")
            return chunks
        except Exception as e:
            logger.error(f"Error retrieving chunks by ID: {e}")
            return []

    def clear(self) -> None:
        """Irreversibly deletes all data in the vector store and removes local files."""
        if self.collection_name:
            try:
                if self._content_vector_store:
                    self._content_vector_store.delete_collection()
                    logger.info(f"Deleted content collection: {self.collection_name}_content")
            except Exception as e:
                logger.warning(f"Failed to delete content collection: {e}")

            try:
                if self._metadata_vector_store:
                    self._metadata_vector_store.delete_collection()
                    logger.info(f"Deleted metadata collection: {self.collection_name}_metadata")
            except Exception as e:
                logger.warning(f"Failed to delete metadata collection: {e}")

        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Removed persist directory: {self.persist_directory}")
            except OSError as e:
                logger.warning(f"Failed to remove persist directory: {e}")

        self._content_vector_store = None
        self._metadata_vector_store = None