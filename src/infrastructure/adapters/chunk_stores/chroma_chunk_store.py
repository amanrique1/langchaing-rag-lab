import os
import shutil
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

class ChromaChunkStore(ChunkStore):
    def __init__(self, collection_name: str = "rag_docs", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add(self, chunk: Chunk):
        """Adds a single chunk to the vector store."""

        chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', 0)}"

        self.collection.add(
            embeddings=[self.embeddings.embed_query(chunk.content)],
            documents=[chunk.content],
            metadatas=[chunk.metadata],
            ids=[chunk_id]
        )

    def get(self, chunk_id: str) -> Chunk | None:
        results = self.collection.get(ids=[chunk_id])
        if results["ids"]:
            return Chunk(
                content=results["documents"][0],
                metadata=results["metadatas"][0],
            )
        return None

    def delete(self, chunk_id: str):
        self.collection.delete(ids=[chunk_id])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        chunks = []
        if results:
            for content, metadata in zip(results["documents"][0], results["metadatas"][0]):
                chunks.append(Chunk(content=content, metadata=metadata))

        return chunks

    def clear(self):
        """Clears a specific collection or the entire database."""
        if self.collection_name:
            try:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(name=self.collection_name)
            except ValueError:
                # Collection may not exist, which is fine.
                pass
        else:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)