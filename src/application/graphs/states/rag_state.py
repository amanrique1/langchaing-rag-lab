from typing import TypedDict, List, Optional, Annotated, Any, Dict
import operator
from langchain_core.messages import BaseMessage
from src.domain.models.chunk import Chunk
from src.domain.models.search_result import SearchResult

class RAGState(TypedDict):
    """
    Represents the state of our conversational RAG graph.
    """
    # User/Session Context
    user_id: str
    session_id: str

    # Essential inputs
    query: str
    expanded_queries: Optional[List[str]]

    # Validation flags
    is_safe_fast: bool
    is_safe_semantic: bool
    error: Optional[str]

    # Retrieval (Granular results)
    content_results: Optional[List[SearchResult]]
    metadata_results: Optional[List[SearchResult]]

    # Combined results
    candidates: Optional[List[SearchResult]]
    chunks: Optional[List[Chunk]]
    context_text: Optional[str]

    # Generation
    answer: Optional[str]

    # Conversation History (operator.add appends messages)
    messages: Annotated[List[BaseMessage], operator.add]

    # Configuration
    top_k: int
    num_candidates: int
    use_reranking: bool
    dual_collection: bool
    memory_k: int  # Window size for conversation history