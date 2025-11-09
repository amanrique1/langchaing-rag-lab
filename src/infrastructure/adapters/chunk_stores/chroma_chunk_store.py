import os
import shutil
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk

DEFAULT_COLLECTION_NAME = "rag_docs"

class ChromaChunkStore(ChunkStore):
    def __init__(self, collection_name: str = None):
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME
        self.persist_directory = "./chroma_db"
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

    def save(self, chunks: list[Chunk]):
        """Adds a single chunk to the vector store."""
        documents = []
        chunk_ids = []
        for chunk in chunks:
            chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', 0)}"
            document = Document(page_content=chunk.content, metadata=chunk.metadata)
            documents.append(document)
            chunk_ids.append(chunk_id)

        self.vector_store.add_documents(documents=documents, ids=chunk_ids)

    def get(
        self,
        chunk_id: str,
        where: dict = None,
        limit: int = None,
        offset: int = None,
        where_document: dict = None,
        include: list = None,
    ) -> Chunk | None:
        """Retrieves a single chunk by its ID."""

        results = self.vector_store.get(
            ids=[chunk_id],
            where=where,
            limit=limit,
            offset=offset,
            where_document=where_document,
            include=include,
        )
        if results and results["ids"]:
            return Chunk(
                content=results["documents"][0],
                metadata=results["metadatas"][0],
            )
        return None

    def delete(self, chunk_id: str, where: dict = None, where_document: dict = None):
        """Deletes a single chunk by its ID."""
        self.vector_store.delete(
            ids=[chunk_id], where=where, where_document=where_document
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter: dict = None
    ) -> list[Chunk]:
        """Searches for similar chunks using a query embedding."""

        docs = self.vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=top_k,
            filter=filter
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
