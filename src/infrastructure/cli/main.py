import json
import argparse
import sys
from typing import Optional
from dotenv import load_dotenv

from src.application.dependency_container import DependencyContainer
from src.domain.models.enums import (
    LengthBasedChunkingMode, 
    SemanticChunkingThresholdType,
    QueryExpansionStrategy
)
from src.domain.models.config_classes import ChunkingConfig, QueryConfig, StorageConfig

# --- Runner Functions ---

def run_chunking(chunk_config: ChunkingConfig, container: DependencyContainer, storage_config: StorageConfig):
    """Loads documents, chunks them according to a strategy, and saves them."""
    strategy_params = chunk_config.strategy_config

    # Enum conversion logic
    if chunk_config.strategy == "length_based" and "mode" in strategy_params:
        strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])
    if chunk_config.strategy == "semantic" and "threshold_mode" in strategy_params:
        strategy_params["threshold_mode"] = SemanticChunkingThresholdType(strategy_params["threshold_mode"])

    print(f"Running chunking strategy '{chunk_config.strategy}' on '{chunk_config.source_path}'...")
    
    chunking_use_case = container.get_chunking_use_case()
    storage_use_case = container.get_storage_use_case(storage_config)
    
    chunks = chunking_use_case.execute(
        chunk_config.source_path, chunk_config.strategy, strategy_params
    )
    storage_use_case.save(chunks)
    print(f"Successfully processed and saved {len(chunks)} chunks.")


def run_talk(
    talk_config: QueryConfig, 
    container: DependencyContainer, 
    storage_config: StorageConfig, 
    use_llm_reranking: bool,
    expansion_strategy: Optional[QueryExpansionStrategy] = None
):
    """Searches for relevant chunks and generates an answer."""
    print(f"Question: {talk_config.query}")
    if expansion_strategy:
        print(f"Using {expansion_strategy.value.upper()} query expansion")
    
    talk_use_case = container.get_talk_use_case(storage_config, use_llm_reranking, expansion_strategy)
    answer = talk_use_case.execute(
        talk_config.query,
        talk_config.top_k,
        talk_config.num_candidates,
        use_reranking=talk_config.use_reranking
    )
    print(f"\nAnswer: {answer}")


def run_search(
    search_config: QueryConfig, 
    container: DependencyContainer, 
    storage_config: StorageConfig, 
    use_llm_reranking: bool,
    expansion_strategy: Optional[QueryExpansionStrategy] = None
):
    """Performs a search for relevant chunks."""
    if expansion_strategy:
        print(f"Using {expansion_strategy.value.upper()} query expansion")
        
    search_use_case = container.get_search_use_case(storage_config, use_llm_reranking, expansion_strategy)
    chunks = search_use_case.execute(
        query=search_config.query,
        top_k=search_config.top_k,
        num_candidates=search_config.num_candidates,
        use_reranking=search_config.use_reranking
    )

    if chunks:
        print(f"Found {len(chunks)} relevant chunks for query: '{search_config.query}'")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Content: {chunk.content[:200]}...")
            print(f"Metadata: {chunk.metadata}")
    else:
        print("No relevant chunks found.")


def clean_storage(container: DependencyContainer, storage_config: StorageConfig):
    """Clears all data from the specified storage location."""
    print("Clearing storage...")
    storage_use_case = container.get_storage_use_case(storage_config)
    storage_use_case.clear()
    print("Storage cleared successfully.")


# --- CLI Setup ---

def _add_storage_args(parser: argparse.ArgumentParser):
    """Helper to add standard storage arguments to a parser."""
    parser.add_argument(
        "--single-collection", dest="dual_collection", action="store_false", default=True,
        help="Use single collection mode instead of dual (dual enabled by default)."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chroma-collection", help="ChromaDB collection name.", default=None)
    group.add_argument("--local-dir", help="FileSystem directory path.", default=None)

def _add_ranking_args(parser: argparse.ArgumentParser):
    """Helper to add standard ranking arguments."""
    parser.add_argument("--top-k", type=int, default=5, help="Top K chunks.")
    parser.add_argument("--candidates", type=int, default=20, help="Candidates before reranking.")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", default=True, help="Disable reranking.")
    parser.add_argument("--llm-reranking", action="store_true", default=False, help="Use LLM reranking.")

def _add_expansion_args(parser: argparse.ArgumentParser):
    """Helper to add query expansion arguments."""
    parser.add_argument(
        "--expand", 
        choices=[s.value for s in QueryExpansionStrategy], 
        default=None,
        help="Enable query expansion strategy."
    )

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Chunk documents and interact with them.")
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task to perform")

    # SAVE
    p_save = subparsers.add_parser("save", help="Chunk and save documents.")
    p_save.add_argument("source", help="Path to source folder.")
    p_save.add_argument("strategy", choices=["length_based", "structure_based", "semantic", "full_doc"])
    p_save.add_argument("--config", default="{}", help="JSON config for strategy.")
    p_save.add_argument("--clean", action="store_true", help="Clean before saving.")
    _add_storage_args(p_save)

    # TALK
    p_talk = subparsers.add_parser("talk", help="Ask a question.")
    p_talk.add_argument("query", help="Query string.")
    _add_ranking_args(p_talk)
    _add_expansion_args(p_talk)
    _add_storage_args(p_talk)

    # SEARCH
    p_search = subparsers.add_parser("search", help="Search chunks.")
    p_search.add_argument("query", help="Query string.")
    _add_ranking_args(p_search)
    _add_expansion_args(p_search)
    _add_storage_args(p_search)

    # CLEAN
    p_clean = subparsers.add_parser("clean", help="Clean storage.")
    _add_storage_args(p_clean)

    # DELETE
    subparsers.add_parser("delete", help="Delete documents (not implemented).")

    return parser


def main():
    load_dotenv()
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    container = DependencyContainer()
    
    try:
        # Resolve Storage Configuration
        storage_config = StorageConfig.resolve(
            chroma_collection=getattr(args, 'chroma_collection', None),
            local_dir=getattr(args, 'local_dir', None),
            dual_collection=getattr(args, 'dual_collection', True)
        )

        if args.task == "save":
            if args.clean:
                clean_storage(container, storage_config)
            
            try:
                strategy_params = json.loads(args.config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --config: {e}")

            chunk_config = ChunkingConfig(args.source, args.strategy, strategy_params)
            run_chunking(chunk_config, container, storage_config)
        
        elif args.task in ["search", "talk"]:
            query_config = QueryConfig(
                query=args.query,
                top_k=args.top_k,
                num_candidates=args.candidates,
                use_reranking=args.rerank
            )
            
            # Convert string to enum if provided
            expansion_strategy = None
            if hasattr(args, 'expand') and args.expand:
                expansion_strategy = QueryExpansionStrategy(args.expand)
            
            if args.task == "search":
                run_search(query_config, container, storage_config, args.llm_reranking, expansion_strategy)
            else:
                run_talk(query_config, container, storage_config, args.llm_reranking, expansion_strategy)
        
        elif args.task == "clean":
            clean_storage(container, storage_config)
            
        elif args.task == "delete":
            print("Delete functionality not implemented.")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()