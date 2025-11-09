from typing import List, Dict, Any
from application.ports.document_loader import DocumentLoader
from application.ports.chunk_store import ChunkStore
from src.domain.services.chunking_service import ChunkingService
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from src.domain.strategies.length_based_chunking import LengthBasedChunkingStrategy
from src.domain.strategies.structure_based_chunking import (
    StructureBasedChunkingStrategy,
)
from src.domain.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.models.chunk import Chunk


class ChunkingUseCase:
    def __init__(self, document_loader: DocumentLoader):
        self.document_loader = document_loader
        self.strategies: Dict[str, ChunkingStrategy] = {
            "length_based": LengthBasedChunkingStrategy,
            "structure_based": StructureBasedChunkingStrategy,
            "semantic": SemanticChunkingStrategy,
        }

    def execute(
        self,
        source: str,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        loader_mode: str = "single",
    ) -> List[Chunk]:
        documents = self.document_loader.load(source)

        strategy_class = self.strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Invalid strategy: {strategy_name}")

        strategy = strategy_class(**strategy_config)
        chunking_service = ChunkingService(strategy)

        chunks = chunking_service.chunk_documents(documents)
        return chunks
