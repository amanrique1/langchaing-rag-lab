from dataclasses import dataclass
from typing import Dict, Any

from domain.models.enums import StorageType

@dataclass
class StorageConfig:
    """Configuration for data storage."""
    storage_type: StorageType
    location: str

@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    source_path: str
    strategy: str
    strategy_config: Dict[str, Any]

@dataclass
class QueryConfig:
    """Configuration for query-based tasks (search and talk)."""
    query: str
    top_k: int
    num_candidates: int = 20
    use_reranking: bool = True