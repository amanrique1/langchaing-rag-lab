from typing import List
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk


class StorageService:
    """Service for managing chunk storage operations."""
    
    def __init__(self, chunk_store: ChunkStore):
        """Initialize storage service.
        
        Args:
            chunk_store: Unified chunk store
        """
        self.chunk_store = chunk_store
    
    def save(self, chunks: List[Chunk]) -> None:
        """Save chunks to storage.
        
        Args:
            chunks: List of chunks to save
        """
        self.chunk_store.save(chunks)
    
    def clear(self) -> None:
        """Clear all chunks from storage."""
        self.chunk_store.clear()
