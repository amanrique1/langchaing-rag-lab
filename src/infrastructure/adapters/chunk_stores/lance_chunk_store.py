import os
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional

import lancedb
from lancedb.table import Table

from src.application.ports.chunk_store import ChunkStore
from src.application.ports.embedding_model import EmbeddingModel
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult
from src.domain.services.metadata_manager import MetadataManager

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "rag_docs"
DEFAULT_PERSIST_DIRECTORY = "./lancedb"


class LanceChunkStore(ChunkStore):
    """
    A concrete implementation of ChunkStore using LanceDB.

    This store implements hybrid search with:
    1. Vector search on content (semantic)
    2. BM25 (FTS) search on metadata fields (keyword-based)

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
        Initialize the LanceChunkStore with embedding capabilities.

        Args:
            collection_name (str, optional): The name for the table. Defaults to 'rag_docs'.
            embedding_model (EmbeddingModel): The model used to generate vectors. Required.
            persist_directory (str, optional): Directory to persist the data. Defaults to './lancedb'.
            **kwargs: Additional parameters (reserved for future use).

        Raises:
            ValueError: If embedding_model is not provided.
        """
        if not embedding_model:
            raise ValueError("An embedding model must be provided to initialize LanceChunkStore.")

        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIRECTORY
        self._embeddings = embedding_model

        os.makedirs(self.persist_directory, exist_ok=True)

        self.db = lancedb.connect(self.persist_directory)
        self._table: Optional[Table] = None

        logger.info("LanceChunkStore initialized successfully")

    @property
    def table(self) -> Optional[Table]:
        """Lazily initializes and returns the LanceDB table."""
        if self._table is None:
            try:
                self._table = self.db.open_table(self.collection_name)
                logger.debug(f"Opened existing table: {self.collection_name}")
            except FileNotFoundError:
                logger.debug(f"Table {self.collection_name} does not exist yet")
        return self._table

    def _create_fts_index(self):
        """Create full-text search index for BM25-style retrieval."""
        if self._table:
            try:
                self._table.create_fts_index(
                    ["searchable_metadata", "filename", "section_title", "breadcrumbs"],
                    replace=True,
                    use_tantivy=True
                )
                logger.info("FTS index created successfully")
            except Exception as e:
                logger.warning(f"Could not create FTS index: {e}")

    def save(self, chunks: List[Chunk]) -> None:
        """
        Persists a list of chunks into the vector database with FTS indexing.

        Creates records with:
        - Vector embeddings for semantic search
        - Searchable metadata text for BM25 search
        - Individual metadata fields for filtering

        Args:
            chunks (List[Chunk]): The domain chunks to save.
        """
        if not chunks:
            logger.warning("No chunks to save")
            return

        records = []

        for chunk in chunks:
            chunk_id = getattr(chunk, "chunk_id", None) or self._generate_stable_id(chunk.content)

            try:
                vector = self._embeddings.embed_query(chunk.content)
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk {chunk_id}: {e}")
                continue

            searchable_metadata = MetadataManager.create_searchable_string(chunk.metadata)
            safe_metadata = self._sanitize_metadata(chunk.metadata)

            record = {
                "chunk_id": chunk_id,
                "content": chunk.content,
                "vector": vector,
                "searchable_metadata": searchable_metadata,

                "filename": safe_metadata.get("filename", ""),
                "section_title": safe_metadata.get("section_title", ""),
                "breadcrumbs": safe_metadata.get("breadcrumbs", ""),
                "root_doc_title": safe_metadata.get("root_doc_title", ""),
                "extracted_keywords": safe_metadata.get("extracted_keywords", ""),

                "page": safe_metadata.get("page"),
                "chunk_index": safe_metadata.get("chunk_index", 0),
                "total_chunks": safe_metadata.get("total_chunks", 0),

                "source": safe_metadata.get("source", ""),
                "language_scope": safe_metadata.get("language_scope", "en_es"),
            }
            records.append(record)

        if self._table is None:
            try:
                self._table = self.db.create_table(
                    self.collection_name,
                    data=records,
                    mode="overwrite"
                )
                logger.info(f"Created new table '{self.collection_name}' with {len(records)} chunks")
            except Exception as e:
                logger.error(f"Failed to create table: {e}")
                raise
        else:
            try:
                self._table.add(records)
                logger.info(f"Added {len(records)} chunks to existing table")
            except Exception as e:
                logger.error(f"Failed to add records: {e}")
                raise

        self._create_fts_index()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> List[SearchResult]:
        """
        Performs search with support for separate content/metadata retrieval.

        IMPORTANT: Both modes return COMPLETE chunks with content populated.
        LanceDB stores everything in one table, so no hydration needed.

        Args:
            query (str): The search string.
            top_k (int): Number of results to retrieve.
            filter (Optional[Dict[str, Any]]): Metadata filters to apply.
            mode (str): 'content' for vector search, 'metadata' for BM25 search.

        Returns:
            List[SearchResult]: Results with complete chunks.
        """
        if self.table is None:
            logger.warning("Cannot search: table not initialized")
            return []

        if mode == "content":
            return self._search_content(query, top_k, filter)
        elif mode == "metadata":
            return self._search_metadata(query, top_k, filter)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'content' or 'metadata'")

    def _search_content(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Pure vector similarity search on content field."""
        try:
            query_vector = self._embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        try:
            search = self._table.search(query_vector).limit(top_k)

            if filter:
                filter_string = self._build_filter_string(filter)
                search = search.where(filter_string)

            lance_results = search.to_list()

            return self._lance_to_search_results(lance_results, "semantic_content")
        except Exception as e:
            logger.error(f"Content search failed: {e}")
            return []

    def _search_metadata(
        self,
        query: str,
        top_k: int,
        filter: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """
        BM25 full-text search on metadata fields ONLY.

        Searches across:
        - metadata (consolidated metadata string)
        - filename
        - section_title

        Does NOT search the content field - that's for vector search.
        Returns COMPLETE chunks since LanceDB stores everything in one table.
        """
        try:
            # Search using FTS (only metadata fields are indexed)
            search = self._table.search(self._sanitize_fts_query(query), query_type="fts").limit(top_k)

            if filter:
                filter_string = self._build_filter_string(filter)
                search = search.where(filter_string)

            lance_results = search.to_list()

            logger.info(f"Metadata search results found: {len(lance_results)}")

            return self._lance_to_search_results(lance_results, "bm25_metadata")
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def search_by_filename(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        BM25 search heavily weighted toward filename field.

        Query syntax: "filename:<query> OR <query>"
        This boosts matches in the filename field.
        """
        sanitized_query = self._sanitize_fts_query(query)
        boosted_query = f"filename:{sanitized_query} OR {sanitized_query}"

        try:
            search = self._table.search(boosted_query, query_type="fts").limit(top_k)

            if filter:
                filter_string = self._build_filter_string(filter)
                search = search.where(filter_string)

            lance_results = search.to_list()
            return self._lance_to_search_results(lance_results, "bm25_filename")
        except Exception as e:
            logger.error(f"Filename search failed: {e}")
            return []

    def search_by_section(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        BM25 search focused on section titles and breadcrumbs.

        Query syntax: "section_title:<query> breadcrumbs:<query> OR <query>"
        This boosts matches in structural metadata.
        """
        sanitized_query = self._sanitize_fts_query(query)
        boosted_query = f"section_title:{sanitized_query} breadcrumbs:{sanitized_query} OR {sanitized_query}"

        try:
            search = self._table.search(boosted_query, query_type="fts").limit(top_k)

            if filter:
                filter_string = self._build_filter_string(filter)
                search = search.where(filter_string)

            lance_results = search.to_list()
            return self._lance_to_search_results(lance_results, "bm25_section")
        except Exception as e:
            logger.error(f"Section search failed: {e}")
            return []

    def get_by_ids(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Retrieves complete chunks by their IDs efficiently using batch scanning.

        Args:
            chunk_ids (List[str]): The IDs to fetch.

        Returns:
            List[Chunk]: Complete chunks with content and metadata.
        """
        if not chunk_ids or self.table is None:
            return []

        try:
            id_set = set(chunk_ids)
            chunks = []

            scanner = self._table.to_batches(
                columns=["chunk_id", "content", "filename", "section_title",
                        "breadcrumbs", "root_doc_title", "extracted_keywords",
                        "page", "chunk_index", "total_chunks", "source", "language_scope"]
            )

            for batch in scanner:
                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict['chunk_id'])

                for i in range(num_rows):
                    chunk_id = batch_dict['chunk_id'][i]

                    if chunk_id in id_set:
                        metadata = self._extract_metadata_from_batch(batch_dict, i)

                        chunk = Chunk(
                            content=batch_dict['content'][i],
                            metadata=metadata
                        )
                        chunk.chunk_id = chunk_id
                        chunks.append(chunk)

                        id_set.remove(chunk_id)
                        if not id_set:
                            break

                if not id_set:
                    break

            logger.debug(f"Retrieved {len(chunks)}/{len(chunk_ids)} chunks by ID")
            return chunks

        except Exception as e:
            logger.error(f"Error retrieving chunks by ID: {e}")
            return self._get_by_ids_fallback(chunk_ids)

    def _get_by_ids_fallback(self, chunk_ids: List[str]) -> List[Chunk]:
        """
        Fallback method for get_by_ids using direct table scan.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of Chunk objects retrieved from the table
        """
        try:
            chunks = []
            id_set = set(chunk_ids)

            for batch in self._table.to_batches():
                batch_dict = batch.to_pydict()
                num_rows = len(batch_dict.get('chunk_id', []))

                for i in range(num_rows):
                    chunk_id = batch_dict['chunk_id'][i]
                    if chunk_id in id_set:
                        metadata = self._extract_metadata_from_batch(batch_dict, i)
                        chunk = Chunk(
                            content=batch_dict['content'][i],
                            metadata=metadata
                        )
                        chunk.chunk_id = chunk_id
                        chunks.append(chunk)

            logger.debug(f"Retrieved {len(chunks)} chunks via fallback method")
            return chunks
        except Exception as e:
            logger.error(f"Fallback retrieval also failed: {e}")
            return []

    def _extract_metadata_from_batch(self, batch_dict: Dict, index: int) -> Dict[str, Any]:
        """
        Helper to extract metadata from a batch dictionary at a specific index.

        Args:
            batch_dict: Dictionary containing batch data
            index: Index of the row to extract metadata from

        Returns:
            Dictionary containing extracted metadata
        """
        return {
            "chunk_id": batch_dict['chunk_id'][index],
            "filename": batch_dict.get("filename", [""])[index],
            "section_title": batch_dict.get("section_title", [""])[index],
            "breadcrumbs": batch_dict.get("breadcrumbs", [""])[index],
            "root_doc_title": batch_dict.get("root_doc_title", [""])[index],
            "extracted_keywords": batch_dict.get("extracted_keywords", [""])[index],
            "page": batch_dict.get("page", [None])[index],
            "chunk_index": batch_dict.get("chunk_index", [0])[index],
            "total_chunks": batch_dict.get("total_chunks", [0])[index],
            "source": batch_dict.get("source", [""])[index],
            "language_scope": batch_dict.get("language_scope", ["en_es"])[index],
        }

    def delete(self, chunk_id: str, where: Optional[Dict[str, Any]] = None, where_document: Optional[Dict[str, Any]] = None) -> None:
        """
        Deletes a chunk from the database.

        Args:
            chunk_id: ID of the chunk to delete
            where: Optional filter conditions
            where_document: Optional document-level filter conditions

        Returns:
            None
        """
        if self.table is None:
            logger.warning("Cannot delete: table not initialized")
            return

        try:
            filter_string = f"chunk_id = '{chunk_id}'"

            if where:
                additional_filter = self._build_filter_string(where)
                filter_string = f"{filter_string} AND {additional_filter}"

            self._table.delete(filter_string)
            logger.info(f"Deleted chunk: {chunk_id}")
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")

    def get_client(self) -> Any:
        """Return the LanceDB connection."""
        return self.db

    def clear(self) -> None:
        """Irreversibly deletes all data in the table."""
        try:
            # Always attempt to drop, regardless of whether table is loaded in memory
            self.db.drop_table(self.collection_name)
            logger.info(f"Dropped table: {self.collection_name}")
            self._table = None
        except Exception as e:
            logger.warning(f"Could not drop table {self.collection_name}: {e}")
            self._table = None

    # ========================
    # Helper Methods
    # ========================

    def _lance_to_search_results(
        self,
        lance_results: List[Dict],
        method: str
    ) -> List[SearchResult]:
        """
        Convert LanceDB results to SearchResult domain objects with COMPLETE chunks.

        Args:
            lance_results: List of LanceDB search results
            method: Search method used (e.g., "bm25", "cosine")

        Returns:
            List of SearchResult objects
        """
        results = []

        for rank, result in enumerate(lance_results, start=1):
            metadata = self._extract_metadata(result)

            # LanceDB returns complete records - content is always present
            chunk = Chunk(
                content=result["content"],
                metadata=metadata
            )
            chunk.chunk_id = result["chunk_id"]

            distance = result.get("_distance", 0)
            similarity_score = self._normalize_score(distance, method)

            results.append(SearchResult(
                chunk=chunk,
                score=similarity_score,
                retrieval_method=method,
                rank=rank
            ))

        return results

    def _extract_metadata(self, result: Dict) -> Dict[str, Any]:
        """
        Reconstruct metadata dictionary from LanceDB record.

        Args:
            result: LanceDB search result record

        Returns:
            Dictionary containing metadata fields
        """
        return {
            "chunk_id": result["chunk_id"],
            "filename": result.get("filename", ""),
            "section_title": result.get("section_title", ""),
            "breadcrumbs": result.get("breadcrumbs", ""),
            "root_doc_title": result.get("root_doc_title", ""),
            "extracted_keywords": result.get("extracted_keywords", ""),
            "page": result.get("page"),
            "chunk_index": result.get("chunk_index", 0),
            "total_chunks": result.get("total_chunks", 0),
            "source": result.get("source", ""),
            "language_scope": result.get("language_scope", "en_es"),
        }

    def _normalize_score(self, distance: float, method: str) -> float:
        """
        Normalize scores to 0-1 range based on search method.

        Args:
            distance: Distance value to normalize
            method: Search method used (e.g., "bm25", "cosine")

        Returns:
            Normalized score value
        """
        if "bm25" in method or "fts" in method:
            # FTS scores are relevance-based
            return min(distance / (distance + 10.0), 1.0)
        else:
            # Vector distances
            return 1.0 / (1.0 + distance)

    def _build_filter_string(self, filter: Dict[str, Any]) -> str:
        """
        Convert dictionary filter to LanceDB SQL-like WHERE clause.

        Args:
            filter: Dictionary containing filter conditions

        Returns:
            SQL-like WHERE clause string
        """
        conditions = []
        for key, value in filter.items():
            if value is None:
                conditions.append(f"{key} IS NULL")
            elif isinstance(value, str):
                escaped_value = value.replace("'", "''")
                conditions.append(f"{key} = '{escaped_value}'")
            elif isinstance(value, bool):
                conditions.append(f"{key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                conditions.append(f"{key} = {value}")
            else:
                conditions.append(f"{key} = '{str(value)}'")

        return " AND ".join(conditions) if conditions else "1=1"

    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes metadata dictionaries for storage compatibility.

        Args:
            metadata: Dictionary containing metadata fields

        Returns:
            Sanitized metadata dictionary
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

        Args:
            content: String content to hash

        Returns:
            MD5 hash of the content
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """
        Sanitize query string for FTS (Tantivy) by escaping special characters
        and normalizing whitespace/newlines.

        - Preserves simple terms and phrases
        - Removes problematic syntax
        - Falls back to token-based search if query is too complex

        Args:
            query (str): Raw query string

        Returns:
            str: Sanitized safe query for FTS
        """
        if not query or not query.strip():
            return '""'

        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query).strip()

        # Check if query is "simple" (no special chars except space, hyphen, dot)
        if re.match(r'^[\w\s\.\-]+$', query):
            return query  # Already safe

        # For complex queries, extract and escape phrases and terms
        result_parts = []

        # Extract quoted phrases first
        phrases = re.findall(r'"([^"]*)"', query)
        for phrase in phrases:
            # Escape internal quotes and add back
            safe_phrase = phrase.replace('\\', '\\\\').replace('"', '\\"')
            result_parts.append(f'"{safe_phrase}"')

        # Remove quoted sections and extract remaining terms
        query_without_phrases = re.sub(r'"[^"]*"', ' ', query)
        tokens = re.findall(r'[\w][\w\.\-]*', query_without_phrases)

        # Add tokens that are long enough
        result_parts.extend([t for t in tokens if len(t) >= 2])

        if not result_parts:
            return '""'

        return ' '.join(result_parts)