from typing import List
from src.domain.models.chunk import Chunk
from src.domain.models.enums import StorageType
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from infrastructure.adapters.chunk_stores.chroma_chunk_store import (
    ChromaChunkStore,
)
from infrastructure.adapters.chunk_stores.file_system_chunk_store import (
    FileSystemChunkStore,
)


class StorageUseCase:
    def __init__(self, store_type: StorageType, output_loc: str = None):
        if store_type == StorageType.LOCAL:
            self.chunk_store = FileSystemChunkStore(output_loc)
        else:
            self.chunk_store = ChromaChunkStore(output_loc)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def save(self, chunks: List[Chunk]) -> None:
        self.chunk_store.save(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Chunk]:
        # Generate embedding for the query
        query_embedding = self.embeddings.embed_query(query)

        # Retrieve relevant chunks
        relevant_chunks = self.chunk_store.search(query_embedding, top_k=top_k)

        return relevant_chunks

    def clear(self) -> None:
        self.chunk_store.clear()
