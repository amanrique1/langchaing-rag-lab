import re
from typing import List
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from src.domain.services.metadata_manager import MetadataManager


class FullDocChunkingStrategy(ChunkingStrategy):
    """
    A concrete implementation of ChunkingStrategy that preserves documents as single chunks.

    This strategy performs a 1-to-1 mapping: each input Document is converted 
    into exactly one Chunk. No splitting or text modification occurs.
    
    This is ideal for:
    1. Documents that are already small (e.g., emails, social media posts).
    2. Scenarios utilizing LLMs with large context windows where splitting is unnecessary.
    """

    def __init__(self):
        """
        Initialize the whole-document chunking strategy.
        
        No configuration parameters are required for this strategy.
        """
        pass

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Converts a list of domain Documents into Chunks without splitting.

        Args:
            documents (List[Document]): A list of domain Document objects.

        Returns:
            List[Chunk]: A list of Chunk objects. The length of this list will 
            match the length of the input list.
        """
        all_chunks = []
        
        for doc in documents:
            # Attempt to find a "Title" (H1) to act as the hierarchy context
            # Since it's a full doc, the 'hierarchy' is just the document title.
            title = self._extract_h1_title(doc.content)
            hierarchy = [title] if title else []

            std_metadata = MetadataManager.normalize_metadata(
                doc_metadata=doc.metadata,
                chunk_content=doc.content,
                chunk_index=0,
                total_chunks=1,
                hierarchy=hierarchy
            )

            # Create a single chunk containing the entire document content
            chunk = Chunk(
                content=doc.content,
                metadata=std_metadata
            )
            all_chunks.append(chunk)
            
        return all_chunks
    
    def _extract_h1_title(self, content: str) -> str:
        """
        Helper to find the first H1 header (# Title) to use as context.

        Args:
            content (str): The document content to search within.

        Returns:
            str: The extracted H1 title, or None if no H1 is found.
        """
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None