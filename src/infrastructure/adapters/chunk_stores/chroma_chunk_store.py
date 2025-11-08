import os
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

class ChromaChunkStore(ChunkStore):
    def __init__(self, collection_name: str = "rag_docs", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def add(self, chunk: Chunk):
        """Adds a single chunk to the vector store."""

        chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', 0)}"
        document = Document(page_content=chunk.content, metadata=chunk.metadata)

        self.vector_store.add_documents(documents=[document], ids=[chunk_id])

    def get(self, chunk_id: str) -> Chunk | None:
        """Retrieves a single chunk by its ID."""

        results = self.vector_store.get(ids=[chunk_id])
        if results and results["ids"]:
            return Chunk(
                content=results["documents"][0],
                metadata=results["metadatas"][0],
            )
        return None

    def delete(self, chunk_id: str):
        """Deletes a single chunk by its ID."""
        self.vector_store.delete(ids=[chunk_id])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Searches for similar chunks using a query embedding."""

        docs = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding, k=top_k
        )

        return [Chunk(content=doc.page_content, metadata=doc.metadata) for doc in docs]

    def clear(self):
        """Clears a specific collection or the entire database."""

        if self.collection_name:
            try:
                self.vector_store.delete_collection()
                # Recreate the collection after deletion
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings,
                )
            except Exception:
                # Handle cases where the collection might not exist
                pass
        else:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
