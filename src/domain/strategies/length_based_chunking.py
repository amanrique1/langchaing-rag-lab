import re
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from src.domain.models.document import Document
from langchain_core.documents import Document as LangchainDocument
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from src.domain.models.enums import LengthBasedChunkingMode
from src.domain.services.metadata_manager import MetadataManager


class LengthBasedChunkingStrategy(ChunkingStrategy):
    """
    A concrete implementation of ChunkingStrategy that splits text based on length constraints.

    This strategy utilizes LangChain's splitters to divide documents.
    - For CHARACTER mode, it uses RecursiveCharacterTextSplitter with separators optimized 
      for Markdown (Paragraphs > Lines > Words).
    - For TOKEN mode, it splits strictly based on token counts.

    Attributes:
        chunk_size (int): The maximum size of a single chunk (measured in characters
            or tokens, depending on the mode).
        chunk_overlap (int): The number of characters or tokens to overlap between
            consecutive chunks to maintain context.
        mode (LengthBasedChunkingMode): The mode determining whether to count
            by characters or tokens.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        mode: LengthBasedChunkingMode = LengthBasedChunkingMode.CHARACTER,
    ):
        """
        Initialize the length-based chunking strategy.

        Args:
            chunk_size (int): The target maximum size for each chunk.
            chunk_overlap (int): The amount of overlap between adjacent chunks.
            mode (LengthBasedChunkingMode): Specifies the unit of measurement
                (CHARACTER or TOKEN). Defaults to LengthBasedChunkingMode.CHARACTER.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Splits a list of domain Documents into smaller Chunks. 
        It also enriches the metadata with chunk indexing information.

        Args:
            documents (List[Document]): A list of domain Document objects to be split.

        Returns:
            List[Chunk]: A list of Chunk objects containing the split text and
            preserved/enriched metadata.

        Raises:
            ValueError: If the configured `mode` is not a valid LengthBasedChunkingMode.
        """
        if self.mode == LengthBasedChunkingMode.CHARACTER:
            # We explicitly define separators to prioritize Markdown structure.
            # 1. \n\n (Paragraphs) - Keep paragraphs together
            # 2. \n   (Lines/Headers) - Keep lines together
            # 3. " "  (Words) - Keep words together
            # 4. ""   (Chars) - Last resort hard cut
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
        elif self.mode == LengthBasedChunkingMode.TOKEN:
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        all_chunks = []
        for doc in documents:
            langchain_document = LangchainDocument(
                page_content=doc.content, metadata=doc.metadata
            )
            doc_chunks = splitter.split_documents([langchain_document])

            total_chunks = len(doc_chunks)

            # Map back to domain Chunk objects with enriched metadata
            for i, doc_chunk in enumerate(doc_chunks):
                content = doc_chunk.page_content
                
                # Best Effort Context
                headers_in_chunk = self._find_headers_in_text(content)
                
                # Standardize
                # The 'extracted_keywords' are CRITICAL here.
                # They act as the semantic glue for chunks that have no structural context.
                std_metadata = MetadataManager.normalize_metadata(
                    doc_metadata=doc.metadata,
                    chunk_content=content,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    hierarchy=headers_in_chunk
                )

                chunk = Chunk(
                    content=content,
                    metadata=std_metadata
                )
                all_chunks.append(chunk)
                
        return all_chunks

    def _find_headers_in_text(self, text: str) -> List[str]:
        """
        Scans the split chunk for any markdown headers it might contain.
        Returns a flat list of headers found, e.g., ["Section A", "Subsection B"]
        """
        headers = []
        # Matches # Header, ## Header, etc.
        matches = re.findall(r'^(#+)\s+(.+)$', text, re.MULTILINE)
        for _, title in matches:
            headers.append(title.strip())
        return headers