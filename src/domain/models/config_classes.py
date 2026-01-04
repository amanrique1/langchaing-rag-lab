from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.domain.models.enums import StorageType

@dataclass(frozen=True)
class StorageConfig:
    """
    Configuration for data storage. Frozen to allow hashing for caching.
    
    Storage backends:
    - LANCE: Default, high-performance hybrid search (vector + BM25)
    - CHROMA: Alternative vector store with dual collection support
    - FILESYSTEM: Local JSON storage for development/testing
    
    All stores accept:
    - collection_name: Collection/table name for organizing data
    - persist_directory: Storage location (None = use store's default)
    """
    storage_type: StorageType
    collection_name: str
    persist_directory: Optional[str] = None
    dual_collection: bool = True

    @classmethod
    def resolve(
        cls, 
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None, 
        use_filesystem: bool = False,
        use_chroma: bool = False,
        dual_collection: bool = True
    ) -> "StorageConfig":
        """
        Factory method to resolve storage configuration from arguments.
        
        Priority:
        1. --filesystem → FILESYSTEM
        2. --chroma → CHROMA
        3. Default → LANCE
        
        Args:
            collection_name: Collection/table name for organizing data
            persist_directory: Storage directory path (None = use store default)
            use_filesystem: Use filesystem storage
            use_chroma: Use ChromaDB instead of LanceDB
            dual_collection: Enable dual collection mode
            
        Returns:
            StorageConfig: Resolved configuration
        """
        
        # Determine storage type (priority: filesystem > chroma > lance)
        if use_filesystem:
            storage_type = StorageType.FILESYSTEM
        elif use_chroma:
            storage_type = StorageType.CHROMA
        else:
            storage_type = StorageType.LANCE
        
        if persist_directory and persist_directory.strip() == "":
            persist_directory = None
        
        if collection_name and collection_name.strip() == "":
            collection_name = None

        return cls(
            storage_type=storage_type,
            collection_name=collection_name,
            persist_directory=persist_directory,
            dual_collection=dual_collection
        )


@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    source_path: str
    strategy: str
    strategy_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryConfig:
    """Configuration for query-based tasks (search and talk)."""
    query: str
    top_k: int
    num_candidates: int = 20
    use_reranking: bool = True