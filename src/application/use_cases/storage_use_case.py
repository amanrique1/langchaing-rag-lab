from typing import List, Optional
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore
from src.domain.services.storage_service import StorageService


class StorageUseCase:
    """Orchestrates storage operations."""
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: Optional[str] = None,
        local_dir: Optional[str] = None,
        dual_collection: bool = True
    ):
        """Initialize storage use case with dependencies.
        
        Args:
            embedding_model: Embedding model for vector search
            collection_name: Name of the Chroma collection (for ChromaDB)
            local_dir: Directory path (for FileSystem storage)
            dual_collection: Whether to enable dual collection storage
        """
        # Create appropriate chunk store based on provided parameters
        chunk_store: ChunkStore
        if collection_name:
            chunk_store = ChromaChunkStore(
                collection_name=collection_name,
                embedding_model=embedding_model,
                dual_collection=dual_collection
            )
        elif local_dir:
            chunk_store = FileSystemChunkStore(
                local_dir=local_dir,
                embedding_model=embedding_model,
                dual_collection=dual_collection
            )
        else:
            raise ValueError("Either collection_name or local_dir must be provided")
        
        self.chunk_store = chunk_store
        
        # Create storage service
        self.storage_service = StorageService(self.chunk_store)
    
    def save(self, chunks: List[Chunk]) -> None:
        """Save chunks to storage.
        
        Args:
            chunks: List of chunks to save
        """
        self.storage_service.save(chunks)
    
    def clear(self) -> None:
        """Clear all chunks from storage."""
        self.storage_service.clear()
