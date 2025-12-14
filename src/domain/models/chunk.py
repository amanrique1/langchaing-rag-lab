from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None

    def __post_init__(self):
        """Generate chunk_id if not provided."""
        if self.chunk_id is None:
            source = self.metadata.get('source', 'unknown')
            chunk_index = self.metadata.get('chunk_index', 0)
            self.chunk_id = f"{source}_{chunk_index}"
