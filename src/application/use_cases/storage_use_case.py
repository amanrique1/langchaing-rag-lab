from typing import List
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk
from src.domain.services.storage_service import StorageService


class StorageUseCase:
    """Orchestrates storage operations."""
    
    def __init__(
        self,
        chunk_store: ChunkStore
    ):
        """Initialize storage use case with dependencies.
        
        Args:
            chunk_store: Chunk store for storage operations
        """        
        self.storage_service = StorageService(chunk_store)
    
    def save(self, chunks: List[Chunk]) -> None:
        """Save chunks to storage.
        
        Args:
            chunks: List of chunks to save
        """
        self.storage_service.save(chunks)
    
    def clear(self) -> None:
        """Clear all chunks from storage."""
        self.storage_service.clear()
