from typing import List, Optional
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.reranker import Reranker
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.chunk import Chunk
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore
from src.domain.services.search_service import SearchService


class SearchUseCase:
    """Orchestrates search operation - coordinates dependencies and services.
    Organize the embedding, retrieval, ensemble and reranking steps.
    """
    
    def __init__(
        self,
        chunk_store: ChunkStore,
        reranker: Optional[Reranker] = None
    ):
        """Initialize search use case with dependencies.
        
        Args:
            chunk_store: Chunk store for search operations
            reranker: Optional reranker for search refinement
        """
                
        self.chunk_store = chunk_store
        
        # Create search service with dependencies
        self.search_service = SearchService(
            chunk_store=self.chunk_store,
            reranker=reranker
        )
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        num_candidates: int = 20,
        use_reranking: bool = True
    ) -> List[Chunk]:
        """Execute search - delegates to search service.
        
        Args:
            query: Search query
            top_k: Number of final results
            num_candidates: Number of candidates before reranking
            use_reranking: Whether to use LLM reranking
            
        Returns:
            List of relevant chunks
        """
        return self.search_service.search(
            query, top_k, num_candidates, use_reranking
        )
