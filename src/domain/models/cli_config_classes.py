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
class TalkConfig:
    """Configuration for the 'talk' or 'search' tasks."""
    query: str
    top_k: int