from typing import List
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter
from src.domain.models.document import Document
from langchain_core.documents import Document as LangchainDocument
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from src.domain.models.enums import LengthBasedChunkingMode


class LengthBasedChunkingStrategy(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        mode: LengthBasedChunkingMode = LengthBasedChunkingMode.CHARACTER,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        if self.mode == LengthBasedChunkingMode.CHARACTER:
            splitter = CharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator="",
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
            for i, doc_chunk in enumerate(doc_chunks):
                chunk = Chunk(
                    content=doc_chunk.page_content,
                    metadata={
                        **doc_chunk.metadata,
                        "chunk_index": i,
                        "total_chunks_in_doc": len(doc_chunks),
                    },
                )
                all_chunks.append(chunk)
        return all_chunks
