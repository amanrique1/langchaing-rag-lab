import json
import numpy as np
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.embedding_model import EmbeddingModel
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult

DEFAULT_OUTPUT_DIR = "./output_chunks"


class FileSystemChunkStore(ChunkStore):
    """
    A persistent chunk store backed by the local file system using JSON files.

    **Architecture:**
    This store treats the file system as a NoSQL-style database. Each chunk is saved 
    as an individual `.json` file containing its ID, content, metadata, and pre-computed 
    vector embedding.

    **Dual Collection Mode:**
    If enabled, this store maintains two parallel directories:
    1. `/content`: Stores the actual document chunks.
    2. `/metadata`: Stores text summaries of the metadata (e.g., "Source: report.pdf").
       This allows for "Metadata-First" retrieval strategies.

    **Performance Characteristics:**
    - **Write:** Fast (O(1) file write per chunk).
    - **Read (ID):** Fast (O(1) file read).
    - **Search (Vector):** Slow / O(N). It performs a linear scan (Brute Force) over 
      all JSON files. Suitable for <10,000 chunks, but too slow for production scale.
    """
    
    def __init__(
        self,
        local_dir: str = None,
        embedding_model: EmbeddingModel = None,
        dual_collection: bool = True
    ):
        """
        Initialize the file system chunk store.
        
        Args:
            local_dir (str, optional): The root directory path for storing JSON files. 
                Defaults to "./output_chunks".
            embedding_model (EmbeddingModel, optional): The model used to generate embeddings. 
                If provided, embeddings are computed during `save()`. If missing, 
                search falls back to simple keyword matching.
            dual_collection (bool): If True, creates separate 'content' and 'metadata' 
                directories for specialized retrieval. Defaults to True.
        """
        self.output_dir = Path(local_dir or DEFAULT_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.dual_collection = dual_collection
        
        # Create subdirectories for dual collection mode
        if self.dual_collection:
            self.content_dir = self.output_dir / "content"
            self.metadata_dir = self.output_dir / "metadata"
            self.content_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.content_dir = self.output_dir
            self.metadata_dir = None

    def _get_file_path(self, directory: Path, chunk_id: str) -> Path:
        """
        Generates a safe file path for a given chunk ID.
        
        replaces slashes with underscores to prevent directory traversal issues
        when chunk IDs contain paths (e.g. 'data/doc.md').
        """
        safe_id = chunk_id.replace("/", "_")
        return directory / f"{safe_id}.json"

    def save(self, chunks: List[Chunk]) -> None:
        """
        Persists a list of chunks to the file system.
        
        **Critical Optimization:** 
        This method pre-computes vector embeddings for both content and metadata 
        (if available) and saves them into the JSON file. This shifts the heavy 
        lifting of inference to 'Write Time', making 'Read Time' (Search) significantly faster.

        Args:
            chunks (List[Chunk]): The list of domain objects to save.
        """
        for chunk in chunks:
            # 1. Generate Robust ID
            # Use chunk_id if it exists, otherwise hash the content to prevent duplicates/overwrites
            chunk_id = getattr(chunk, "chunk_id", None) or str(hash(chunk.content))
            
            # 2. Pre-compute Content Embedding
            embedding = None
            if self.embedding_model:
                emb_vector = self.embedding_model.embed_query(chunk.content)
                # Ensure it is a standard list for JSON serialization
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
            
            # Save to disk
            content_file = self._get_file_path(self.content_dir, chunk_id)
            with open(content_file, "w", encoding="utf-8") as f:
                json.dump(content_data, f, indent=2)
            
            # 3. Save Metadata (Dual Mode)
            if self.dual_collection and self.metadata_dir:
                metadata_summary = self._create_metadata_summary(chunk)
                
                # Embed the summary
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
                    "metadata": chunk.metadata, # Keep original metadata for reference
                    "embedding": meta_embedding 
                }

                metadata_file = self._get_file_path(self.metadata_dir, chunk_id)
                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata_data, f, indent=2)

    def _create_metadata_summary(self, chunk: Chunk) -> str:
        """
        Transforms a metadata dictionary into a searchable string format.

        This is used for the 'Metadata Collection'. It converts key-value pairs 
        into a natural language-like string that an embedding model can understand.

        Example:
            Input: {"source": "report.pdf", "page": 10}
            Output: "source: report.pdf | page: 10"

        Args:
            chunk (Chunk): The source chunk.

        Returns:
            str: A stringified representation of the metadata.
        """
        metadata_parts = []
        for key, value in chunk.metadata.items():
            metadata_parts.append(f"{key}: {value}")
        return " | ".join(metadata_parts)

    def delete(self, chunk_id: str) -> None:
        """
        Deletes a specific chunk and its corresponding metadata file from storage.

        This method attempts to remove files from both the 'content' directory 
        and the 'metadata' directory (if dual collection is active). If the file 
        does not exist, the operation is skipped silently.

        Args:
            chunk_id (str): The unique identifier of the chunk to remove.
        """
        content_file = self._get_file_path(self.content_dir, chunk_id)
        if content_file.exists():
            content_file.unlink()
        
        if self.dual_collection and self.metadata_dir:
            metadata_file = self._get_file_path(self.metadata_dir, chunk_id)
            if metadata_file.exists():
                metadata_file.unlink()

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        mode: str = "content"
    ) -> List[SearchResult]:
        """
        Performs a search over stored JSON files.
        
        This method automatically chooses the best search strategy:
        1. **Vector Search:** Used if `embedding_model` was provided during initialization.
        2. **Keyword Search:** Used as a fallback if no model is available.

        Args:
            query (str): The search text.
            top_k (int): Max results to return.
            filter (Optional[Dict]): Exact-match metadata filters (e.g. {"source": "doc.pdf"}).
            mode (str): 'content' to search document text, 'metadata' to search metadata summaries.

        Returns:
            List[SearchResult]: A list of results containing the Chunk, the score (0-1), 
            and the rank.
        """
        if not self.embedding_model:
            # Fallback to keyword search with dummy scores if no model available
            chunks = self._keyword_search(query, top_k)
            return [
                SearchResult(chunk=chunk, score=None, rank=i+1, retrieval_method="keyword")
                for i, chunk in enumerate(chunks)
            ]
        
        # Determine target directory
        search_dir = self.metadata_dir if (mode == "metadata" and self.metadata_dir) else self.content_dir
        
        # Embed query once
        query_embedding = self.embedding_model.embed_query(query)
        
        raw_results = []
        retrieval_method = f"semantic_{mode}"
        
        # O(N) Scan
        for file_path in search_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Exact Match Filtering
                    if filter and not self._matches_filter(data["metadata"], filter):
                        continue
                    
                    # Retrieve Embedding
                    chunk_embedding = data.get("embedding")
                    if not chunk_embedding:
                        continue 
                    
                    # Calculate Score
                    similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                    
                    chunk = Chunk(content=data["content"], metadata=data["metadata"])
                    
                    raw_results.append({
                        "chunk": chunk,
                        "score": float(similarity),
                        "method": retrieval_method
                    })
            except Exception:
                continue 
        
        # Sort descending by score (Highest similarity first)
        raw_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = raw_results[:top_k]
        
        # Map to Output Objects
        final_results = []
        for rank, res in enumerate(top_results, start=1):
            final_results.append(SearchResult(
                chunk=res["chunk"],
                score=(1.0/(1.0+res["score"])),
                retrieval_method=res["method"],
                rank=rank
            ))
            
        return final_results

    def _keyword_search(self, query: str, top_k: int) -> List[Chunk]:
        """
        Performs a naive case-insensitive substring search over the JSON files.

        **Complexity:** O(N) - Linear Scan.
        
        This iterates through every file in the content directory, loads the JSON, 
        and checks if the query string exists inside the content field.

        Args:
            query (str): The text to find.
            top_k (int): The limit on results.

        Returns:
            List[Chunk]: Matches found (unordered).
        """
        relevant_chunks = []
        # glob matches all JSON files
        # Note: Keyword search only targets content directory
        for file_path in self.content_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if query.lower() in data["content"].lower():
                        relevant_chunks.append(
                            Chunk(content=data["content"], metadata=data["metadata"])
                        )
                        if len(relevant_chunks) >= top_k:
                            break
            except Exception:
                continue
        return relevant_chunks
        return final_results

    def _matches_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Checks if the metadata dictionary matches the provided filter criteria.

        This implementation uses strict exact matching logic. For a match to occur,
        every key-value pair in the 'filter' dictionary must exist and be equal 
        in the 'metadata' dictionary (AND logic).

        Args:
            metadata (Dict[str, Any]): The metadata associated with a chunk.
            filter (Dict[str, Any]): The filter criteria (e.g., {"source": "report.pdf"}).

        Returns:
            bool: True if the metadata satisfies all filter conditions, False otherwise.
        """
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Computes the cosine similarity between two embedding vectors.

        Formula: $ (A . B) / (||A|| * ||B||) $

        Args:
            vec1 (List[float]): The first vector (e.g., query embedding).
            vec2 (List[float]): The second vector (e.g., chunk embedding).

        Returns:
            float: A score between -1.0 and 1.0 indicating similarity.
                1.0 means identical direction.
                0.0 means orthogonal (no correlation).
                -1.0 means opposite direction.
        """
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
        Retrieves specific chunks by their IDs from the content directory.

        This method attempts to load each requested ID. If a file is missing or 
        corrupt, it is skipped without raising an error (best-effort retrieval).

        Args:
            chunk_ids (List[str]): The list of IDs to fetch.

        Returns:
            List[Chunk]: A list containing the successfully loaded chunks. 
            May be shorter than the input list if some IDs were not found.
        """
        chunks = []
        for chunk_id in chunk_ids:
            file_path = self._get_file_path(self.content_dir, chunk_id)
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        chunks.append(
                            Chunk(content=data["content"], metadata=data["metadata"])
                        )
                except Exception:
                    pass
        return chunks

    def clear(self) -> None:
        """
        Destructively clears the entire chunk store.

        This method recursively deletes the `output_dir` (removing all JSON files) 
        and then recreates the empty directory structure. 
        
        **Warning:** This operation cannot be undone.
        """
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.dual_collection:
            self.content_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)