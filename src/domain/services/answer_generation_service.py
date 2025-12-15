from typing import List
from src.application.ports.language_model import LanguageModel
from src.domain.models.chunk import Chunk


class AnswerGenerationService:
    """Service for generating answers using search and LLM."""
    
    def __init__(
        self,
        language_model: LanguageModel
    ):
        """Initialize answer generation service.
        
        Args:
            language_model: Language model for answer generation
        """
        self.language_model = language_model

    def generate_answer(
        self,
        query: str,
        chunks: List[Chunk]
    ) -> str:
        """Generate answer by searching for relevant chunks and using LLM.
        
        Args:
            query: User's question
            chunks: List of chunks to use for answer generation
            
        Returns:
            Generated answer string
        """
        
        if not chunks:
            return "No relevant information found to answer the query. Please try rephrasing your question."
        
        # Generate answer using LLM
        return self.language_model.get_answer(query, chunks)
