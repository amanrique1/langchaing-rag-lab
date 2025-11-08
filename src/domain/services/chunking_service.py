from typing import List
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy


class ChunkingService:
    def __init__(self, chunking_strategy: ChunkingStrategy):
        self.chunking_strategy = chunking_strategy

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        return self.chunking_strategy.chunk(documents)
