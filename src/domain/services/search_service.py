from typing import List, Optional
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.reranker import Reranker
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult
from src.domain.services.ensemble_retriever_service import EnsembleRetrieverService


class SearchService:
    """Service for searching chunks with ensemble and reranking capabilities."""
    
    def __init__(
        self,
        chunk_store: ChunkStore,
        reranker: Optional[Reranker] = None
    ):
        """Initialize search service with chunk store and language model.
        
        Args:
            chunk_store: Unified chunk store (manages content and metadata collections)
            reranker: Optional reranker instance
        """
        self.chunk_store = chunk_store
        
        # Initialize reranker
        self.reranker = reranker
        
        # Initialize ensemble retriever if chunk store supports it
        self.ensemble_retriever = None
        if hasattr(chunk_store, 'dual_collection') and chunk_store.dual_collection:
            self.ensemble_retriever = EnsembleRetrieverService(chunk_store)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> List[Chunk]:
        """Search for relevant chunks with optional reranking.
        
        Ensemble retrieval is automatically used if dual collections were enabled
        at initialization time.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            num_candidates: Number of candidates to retrieve before reranking
            use_reranking: Whether to use LLM reranking
            
        Returns:
            List of relevant chunks
        """
        # Automatically use ensemble if available (based on dual_collection setting)
        if self.ensemble_retriever:
            results = self.ensemble_retriever.execute(query, num_candidates)
        else:
            results = self.chunk_store.search(query, num_candidates)
        
        # Rerank if enabled
        if use_reranking and self.reranker:
            results = self.reranker.rerank(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Extract chunks from results
        return [r.chunk for r in results]
