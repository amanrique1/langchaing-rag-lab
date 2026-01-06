from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document
from src.domain.models.chunk import Chunk
from src.application.ports.chunking_strategy import ChunkingStrategy
from src.domain.services.metadata_manager import MetadataManager


class StructureBasedChunkingStrategy(ChunkingStrategy):
    """
    A chunking strategy that respects the structural hierarchy of Markdown documents.

    This strategy employs a "Two-Pass" approach:
    1. **Structural Split:** The document is first divided into logical sections based on
       Markdown headers (#, ##, ###). This ensures that text under "Section A" is never
       merged with text under "Section B".
    2. **Length-Based Split:** If a specific section is still too large (e.g., a long
       chapter), it is further split by character count using recursive rules.

    Attributes:
        chunk_size (int): The maximum size of a single chunk (in characters).
        chunk_overlap (int): The number of characters to overlap between chunks within the same section.
        strip_headers (bool): If True, removes the header text from the chunk content
            (keeping it in metadata only). If False, keeps it in the text.
        headers_to_split (List[Tuple[str, str]]): Configuration of Markdown headers to track.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_headers: bool = False,
        max_header_levels: int = 4,
    ):
        """
        Initialize the structure-based chunking strategy.

        Args:
            chunk_size (int): Target size for the final chunks. Defaults to 1000.
            chunk_overlap (int): Overlap size for internal splitting. Defaults to 200.
            strip_headers (bool): Whether to remove the header text from the content
                body. Defaults to False (recommended to keep False to preserve LLM context).
            max_header_levels (int): How deep to parse the markdown tree (e.g., 3 means
                parsing #, ##, and ###). Defaults to 4.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_headers = strip_headers

        # Dynamically generate header configuration
        # Result example: [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        self.headers_to_split = [
            ("#" * i, f"Header {i}") for i in range(1, max_header_levels + 1)
        ]

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Splits documents preserving Markdown structure.

        Args:
            documents (List[Document]): The domain documents to process.

        Returns:
            List[Chunk]: Chunks enriched with structural metadata (section titles).
        """
        all_chunks = []

        for doc in documents:
            # Step 1: Split the document by logical structure (Headers)
            # This ensures we don't mix semantic contexts (e.g. Intro vs Conclusion).
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split,
                strip_headers=self.strip_headers,
            )

            # header_docs are split strictly by sections.
            header_docs = header_splitter.split_text(doc.page_content)

            # Step 2: Prepare the secondary splitter for sections that are too long
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # Step 3: Process the sections
            doc_sub_chunks = []
            for header_doc in header_docs:
                # If a section is larger than chunk_size, split it further recursively.
                # If it's smaller, this simply wraps it in a list.
                splits = text_splitter.split_documents([header_doc])
                doc_sub_chunks.extend(splits)

            # Step 4: Convert LangChain chunks to Domain Chunks
            for i, sub_chunk in enumerate(doc_sub_chunks):
                # Reconstruct hierarchy list from metadata keys
                hierarchy = []
                for _, header_key in self.headers_to_split:
                    if header_key in sub_chunk.metadata:
                        hierarchy.append(sub_chunk.metadata[header_key])

                # STANDARDIZATION CALL
                std_metadata = MetadataManager.normalize_metadata(
                    doc_metadata=doc.metadata,
                    chunk_content=sub_chunk.page_content,
                    chunk_index=i,
                    total_chunks=len(doc_sub_chunks),
                    hierarchy=hierarchy
                )

                chunk = Chunk(content=sub_chunk.page_content, metadata=std_metadata)
                all_chunks.append(chunk)

        return all_chunks