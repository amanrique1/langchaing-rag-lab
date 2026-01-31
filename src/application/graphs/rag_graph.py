from langgraph.graph import StateGraph, END
from src.application.graphs.states.rag_state import RAGState
from src.application.graphs.nodes.rag_nodes import RAGNodes


def create_rag_graph(nodes: RAGNodes, mode: str = "qa", checkpointer=None):
    """
    Creates a flexible RAG graph that can be used for search-only or full Q&A.

    Modes:
    - "search": Validation → Expansion → Retrieval → Fusion → Reranking → END
                Returns chunks only, no generation
    - "qa": Full pipeline including generation (for TalkUseCase)
    - "chat": Full pipeline with conversation memory (for ChatUseCase)

    Args:
        nodes: RAGNodes instance with all node implementations
        mode: Graph mode ("search", "qa", or "chat")
        checkpointer: Optional checkpointer (for "chat" mode only)

    Returns:
        Compiled graph ready for execution
    """
    workflow = StateGraph(RAGState)

    # --- Add Common Nodes ---
    workflow.add_node("fast_validation", nodes.fast_validation)
    workflow.add_node("semantic_validation", nodes.semantic_validation)
    workflow.add_node("expand_query", nodes.expand_query)
    workflow.add_node("content_search", nodes.content_search)
    workflow.add_node("metadata_search", nodes.metadata_search)
    workflow.add_node("fusion", nodes.fusion)
    workflow.add_node("reranking", nodes.reranking)

    # --- Add Generation Node (only for qa/chat modes) ---
    if mode in ["qa", "chat"]:
        workflow.add_node("generation", nodes.generation)

    # --- Entry Point ---
    workflow.set_entry_point("fast_validation")

    # --- Conditional Edges ---
    workflow.add_conditional_edges(
        "fast_validation",
        lambda state: state.get("is_safe_fast", True),
        {
            True: "semantic_validation",
            False: END
        }
    )

    workflow.add_conditional_edges(
        "semantic_validation",
        lambda state: state.get("is_safe_semantic", True),
        {
            True: "expand_query",
            False: END
        }
    )

    # --- Parallel Execution ---
    workflow.add_edge("expand_query", "content_search")
    workflow.add_edge("expand_query", "metadata_search")

    # --- Sync/Fan-in ---
    workflow.add_edge("content_search", "fusion")
    workflow.add_edge("metadata_search", "fusion")

    # --- Sequential Flow ---
    workflow.add_edge("fusion", "reranking")

    # --- Mode-specific ending ---
    if mode == "search":
        # Search mode: Stop after reranking, return chunks
        workflow.add_edge("reranking", END)
    else:
        # Q&A/Chat mode: Continue to generation
        workflow.add_edge("reranking", "generation")
        workflow.add_edge("generation", END)

    # Compile with optional checkpointer
    # Only chat mode uses checkpointer for conversation history
    if mode == "chat" and checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


# Convenience factories for backward compatibility
def create_search_graph(nodes: RAGNodes):
    """Create a search-only graph (no generation)."""
    return create_rag_graph(nodes, mode="search")


def create_qa_graph(nodes: RAGNodes):
    """Create a Q&A graph (stateless with generation)."""
    return create_rag_graph(nodes, mode="qa")


def create_conversational_rag_graph(nodes: RAGNodes, checkpointer=None):
    """Create a conversational graph (with memory and generation)."""
    return create_rag_graph(nodes, mode="chat", checkpointer=checkpointer)