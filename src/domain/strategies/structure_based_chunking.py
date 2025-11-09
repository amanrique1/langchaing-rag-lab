from typing import List
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy


class StructureBasedChunkingStrategy(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_headers: bool = False,
        max_header_levels: int = 4,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strip_headers = strip_headers
        self.headers_to_split = [
            ("#" * i, f"Header {i}") for i in range(1, max_header_levels + 1)
        ]

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            # Step 1: Split the document by headers
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=self.headers_to_split,
                strip_headers=self.strip_headers,
            )
            header_docs = header_splitter.split_text(doc.content)

            # Step 2: Split each header section by character count
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

            # Step 3: Collect all the split LangChain documents for the entire source document
            doc_sub_chunks = []
            for header_doc in header_docs:
                splits = text_splitter.split_documents([header_doc])
                doc_sub_chunks.extend(splits)

            # Step 4: Iterate through the collected sub-chunks to create final Chunk objects
            for i, sub_chunk in enumerate(doc_sub_chunks):
                chunk = Chunk(
                    content=sub_chunk.page_content,
                    metadata={
                        **sub_chunk.metadata,
                        **doc.metadata,
                        "chunk_index": i, # This index is now unique for the document
                        "total_chunks_in_doc": len(doc_sub_chunks),
                    },
                )
                all_chunks.append(chunk)

        return all_chunks