from typing import List, Dict, Any
from src.application.ports.document_loader import DocumentLoader
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

# Strategies
from src.domain.services.chunking.strategies.full_doc_chunking import FullDocChunkingStrategy
from src.domain.services.chunking.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.services.chunking.strategies.length_based_chunking import LengthBasedChunkingStrategy
from src.domain.services.chunking.strategies.structure_based_chunking import StructureBasedChunkingStrategy

class IngestionUseCase:
    """
    Orchestrates the complete ingestion pipeline:
    Loading -> Chunking -> Saving.
    """

    def __init__(
        self,
        document_loader: DocumentLoader,
        chunk_store: ChunkStore,
        embedding_model: EmbeddingModel | None = None
    ):
        self.document_loader = document_loader
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model

        # Strategy Registry
        self.strategies = {
            "length_based": LengthBasedChunkingStrategy,
            "structure_based": StructureBasedChunkingStrategy,
            "semantic": SemanticChunkingStrategy,
            "full_doc": FullDocChunkingStrategy,
        }

    def ingest(
        self,
        source: str,
        strategy_name: str,
        strategy_config: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Loads documents, chunks them according to configuration, and saves to storage.
        Returns the list of processed chunks.
        """
        # 1. Resolve Strategy
        strategy_class = self.strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Invalid strategy: {strategy_name}")

        # Inject embedding model specifically for semantic strategy
        if strategy_name == "semantic" and self.embedding_model:
            strategy_config["embedding_model"] = self.embedding_model

        strategy = strategy_class(**strategy_config)

        # 2. Load
        documents = self.document_loader.load(source)

        # 3. Chunk
        chunks = strategy.chunk(documents)

        # 4. Save
        self.chunk_store.save(chunks)

        return chunks

    def clear_storage(self) -> None:
        """Clears the underlying storage."""
        self.chunk_store.clear()