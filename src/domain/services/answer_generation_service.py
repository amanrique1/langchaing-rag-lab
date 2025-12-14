from typing import Optional
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.language_model import LanguageModel
from src.application.ports.reranker import Reranker
from src.domain.services.search_service import SearchService


class AnswerGenerationService:
    """Service for generating answers using search and LLM."""
    
    def __init__(
        self,
        language_model: LanguageModel,
        chunk_store: ChunkStore,
        reranker: Optional[Reranker] = None
    ):
        """Initialize answer generation service.
        
        Args:
            language_model: Language model for answer generation
            chunk_store: Unified chunk store
            reranker: Optional reranker for search refinement
        """
        self.language_model = language_model
        # Create search service internally to avoid service-to-service dependency
        self.search_service = SearchService(
            chunk_store=chunk_store,
            reranker=reranker
        )
    
    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> str:
        """Generate answer by searching for relevant chunks and using LLM.
        
        Args:
            query: User's question
            top_k: Number of chunks to use for answer generation
            num_candidates: Number of candidates to retrieve before reranking
            use_reranking: Whether to use LLM reranking
            
        Returns:
            Generated answer string
        """
        # Get relevant chunks via search service
        chunks = self.search_service.search(
            query, top_k, num_candidates, use_reranking
        )
        
        if not chunks:
            return "No relevant information found to answer the query. Please try rephrasing your question."
        
        # Generate answer using LLM
        return self.language_model.get_answer(query, chunks)
