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
        self._embeddings = None
        self._vector_store = None

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """
        Lazily initializes and returns the embeddings model.
        The model is only created when this property is first accessed.
        """
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return self._embeddings

    @property
    def vector_store(self) -> Chroma:
        """
        Lazily initializes and returns the Chroma vector store.
        The vector store is only created when this property is first accessed,
        which also triggers the initialization of the embeddings model.
        """
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    def save(self, chunks: list[Chunk]):
        """Adds a list of chunks to the vector store."""
        documents = []
        chunk_ids = []
        for chunk in chunks:
            chunk_id = f"{chunk.metadata.get('source', 'doc')}_{chunk.metadata.get('chunk_index', 0)}"
            document = Document(page_content=chunk.content, metadata=chunk.metadata)
            documents.append(document)
            chunk_ids.append(chunk_id)

        # Accessing self.vector_store will trigger lazy initialization if needed
        self.vector_store.add_documents(documents=documents, ids=chunk_ids)

    def delete(self, chunk_id: str, where: dict = None, where_document: dict = None):
        """Deletes a single chunk by its ID."""
        self.vector_store.delete(
            ids=[chunk_id], where=where, where_document=where_document
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict = None
    ) -> list[Chunk]:
        """Searches for similar chunks using a query string."""

        docs = self.vector_store.similarity_search(
            query=query,
            k=top_k,
            filter=filter
        )

        return [Chunk(content=doc.page_content, metadata=doc.metadata) for doc in docs]

    def clear(self):
        """
        Clears a specific collection or the entire database.
        Resets the internal vector store instance, allowing it to be
        lazily re-initialized on the next access.
        """
        if self.collection_name:
            try:
                # self.vector_store property will initialize if it hasn't already
                self.vector_store.delete_collection()
            except Exception:
                # Handle cases where the collection might not exist
                pass
        else:
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

        # Invalidate the vector store instance. It will be recreated on next access.
        self._vector_store = None