from typing import List
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.language_model import LanguageModel
from src.domain.models.chunk import Chunk


class TalkUseCase:
    def __init__(self, language_model: LanguageModel, chunk_store: ChunkStore):
        self.language_model = language_model
        self.chunk_store = chunk_store

    def execute(self, query: str, top_k: int = 5) -> str:
        """
        Executes the question-answering chain using the provided query and context.
        """
        # Retrieve relevant chunks
        relevant_chunks = self.chunk_store.search(query, top_k=top_k)

        # A simple guard clause is cleaner
        if not relevant_chunks:
            return "No relevant information found to answer the query. Please try rephrasing your question."

        # The chain is invoked with a dictionary matching the variables in the template
        response = self.language_model.get_answer(query, relevant_chunks)

        return response