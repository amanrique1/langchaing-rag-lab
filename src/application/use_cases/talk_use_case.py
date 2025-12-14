from typing import Optional
from src.application.ports.language_model import LanguageModel
from src.application.ports.reranker import Reranker
from src.application.ports.chunk_store import ChunkStore
from src.domain.services.answer_generation_service import AnswerGenerationService
from src.domain.services.search_service import SearchService


class TalkUseCase:
    """Orchestrates answer generation - coordinates dependencies and services."""
    
    def __init__(
        self,
        language_model: LanguageModel,
        chunk_store: ChunkStore,
        reranker: Optional[Reranker] = None
    ):
        """Initialize talk use case with dependencies.
        
        Args:
            language_model: Language model for answer generation
            chunk_store: Chunk store for vector search
            reranker: Optional[Reranker] = None
        """

        # Create search service
        self.search_service = SearchService(
            chunk_store=chunk_store,
            reranker=reranker
        )
        
        # Create answer generation service
        self.answer_service = AnswerGenerationService(
            language_model=language_model
        )
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> str:
        """Execute answer generation - delegates to answer service.
        
        Args:
            query: User's question
            top_k: Number of chunks for answer generation
            num_candidates: Number of candidates before reranking
            use_reranking: Whether to use LLM reranking
            
        Returns:
            Generated answer
        """
        # Get relevant chunks via search service
        chunks = self.search_service.search(
            query, top_k, num_candidates, use_reranking
        )
        
        return self.answer_service.generate_answer(
            query, chunks
        )