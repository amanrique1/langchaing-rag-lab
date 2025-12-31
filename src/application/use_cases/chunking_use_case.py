from typing import List, Dict, Any
from application.ports.document_loader import DocumentLoader
from src.application.ports.chunking_strategy import ChunkingStrategy
from src.domain.services.chunking.strategies.full_doc_chunking import FullDocChunkingStrategy
from src.domain.services.chunking.strategies.semantic_chunking import SemanticChunkingStrategy
from src.domain.services.chunking.strategies.length_based_chunking import LengthBasedChunkingStrategy
from src.domain.services.chunking.strategies.structure_based_chunking import (
    StructureBasedChunkingStrategy,
)
from src.domain.models.chunk import Chunk


class ChunkingUseCase:
    """
    Orchestrates the process of loading documents and dividing them into chunks
    using a configurable strategy.

    This use case acts as the coordinator between the data access layer (DocumentLoader)
    and the domain logic (ChunkingService/Strategies).

    Attributes:
        document_loader (DocumentLoader): The port instance used to fetch raw documents.
        strategy (ChunkingStrategy | None): The specific chunking strategy instance 
            currently active for the execution.
    """

    def __init__(self, document_loader: DocumentLoader):
        """
        Initialize the use case.

        Args:
            document_loader (DocumentLoader): An implementation of the DocumentLoader
                interface responsible for loading source text.
        """
        self.document_loader = document_loader
        self.strategies = {
            "length_based": LengthBasedChunkingStrategy,
            "structure_based": StructureBasedChunkingStrategy,
            "semantic": SemanticChunkingStrategy,
            "full_doc": FullDocChunkingStrategy,
        }

    def execute(
        self, 
        source: str, 
        strategy_name: str, 
        strategy_config: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Loads documents from the source and chunks them based on the selected strategy.

        Args:
            source (str): The source file path/directory to be passed to the document loader.
            strategy_name (str): The name of the chunking strategy to apply. 
                Valid options are: "length_based", "structure_based", "semantic", "full_doc".
            strategy_config (Dict[str, Any]): A dictionary of configuration parameters 
                specific to the chosen strategy (e.g., chunk_size, overlap).

        Returns:
            List[Chunk]: A list of domain Chunk objects resulting from the process.

        Raises:
            ValueError: If the provided `strategy_name` is not recognized.
        """

        strategy_class = self.strategies.get(strategy_name)
        if not strategy_class:
            raise ValueError(f"Invalid strategy: {strategy_name}")

        # Instantiate the strategy with the provided configuration
        strategy = strategy_class(**strategy_config)
        
        documents = self.document_loader.load(source)
        chunks = strategy.chunk(documents)
        
        return chunks