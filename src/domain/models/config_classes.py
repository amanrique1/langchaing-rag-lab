from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.domain.models.enums import StorageType

@dataclass(frozen=True)
class StorageConfig:
    """Configuration for data storage. Frozen to allow hashing for caching."""
    storage_type: StorageType
    output_loc: str
    dual_collection: bool = True

    @classmethod
    def resolve(
        cls, 
        chroma_collection: Optional[str] = None, 
        local_dir: Optional[str] = None, 
        dual_collection: bool = True
    ) -> "StorageConfig":
        """Factory method to resolve storage configuration from CLI args."""
        if local_dir:
            return cls(
                storage_type=StorageType.FILESYSTEM,
                output_loc=local_dir,
                dual_collection=dual_collection
            )
        elif chroma_collection:
            return cls(
                storage_type=StorageType.CHROMA,
                output_loc=chroma_collection,
                dual_collection=dual_collection
            )
        else:
            return cls(
                storage_type=StorageType.CHROMA,
                output_loc="default_collection",
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