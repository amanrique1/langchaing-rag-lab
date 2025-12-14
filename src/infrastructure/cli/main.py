import json
import argparse
import sys

from dotenv import load_dotenv

from src.application.dependency_container import DependencyContainer
from src.domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType
)
from src.domain.models.cli_config_classes import ChunkingConfig, QueryConfig

def run_chunking(
    chunk_config: ChunkingConfig,
    container: DependencyContainer,
    collection_name: str = None,
    local_dir: str = None,
    dual_collection: bool = True,
):
    """
    Loads documents, chunks them according to a strategy, and saves them.
    """
    strategy_params = chunk_config.strategy_config

    # Safely convert string representations to Enum members
    if chunk_config.strategy == "length_based" and "mode" in strategy_params:
        try:
            strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])
        except ValueError as e:
            raise ValueError(f"Invalid 'mode' for length_based strategy: {e}") from e

    if chunk_config.strategy == "semantic" and "threshold_mode" in strategy_params:
        try:
            strategy_params["threshold_mode"] = SemanticChunkingThresholdType(
                strategy_params["threshold_mode"]
            )
        except ValueError as e:
            raise ValueError(f"Invalid 'threshold_mode' for semantic strategy: {e}") from e

    print(f"Running chunking strategy '{chunk_config.strategy}' on '{chunk_config.source_path}'...")
    
    # Get use cases from container
    chunking_use_case = container.get_chunking_use_case()
    storage_use_case = container.get_storage_use_case(
        collection_name=collection_name,
        local_dir=local_dir,
        dual_collection=dual_collection
    )
    
    chunks = chunking_use_case.execute(
        chunk_config.source_path, chunk_config.strategy, strategy_params
    )
    storage_use_case.save(chunks)

    print(f"Successfully processed and saved {len(chunks)} chunks to content and metadata stores.")


def run_talk(talk_config: QueryConfig, container: DependencyContainer, collection_name: str = None, local_dir: str = None, dual_collection: bool = True, use_llm_reranking: bool = False):
    """
    Searches for relevant chunks and generates an answer based on a query.
    """
    print(f"Question: {talk_config.query}")
    
    talk_use_case = container.get_talk_use_case(
        collection_name=collection_name,
        local_dir=local_dir,
        dual_collection=dual_collection,
        use_llm_reranking=use_llm_reranking
    )

    answer = talk_use_case.execute(
        talk_config.query,
        talk_config.top_k,
        talk_config.num_candidates,
        use_reranking=talk_config.use_reranking
    )

    print(f"\nAnswer: {answer}")


def run_search(search_config: QueryConfig, container: DependencyContainer, collection_name: str = None, local_dir: str = None, dual_collection: bool = True, use_llm_reranking: bool = False):
    """
    Performs a search for relevant chunks and displays them.
    """
    search_use_case = container.get_search_use_case(
        collection_name=collection_name,
        local_dir=local_dir,
        dual_collection=dual_collection,
        use_llm_reranking=use_llm_reranking
    )
    
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


def clean_storage(container: DependencyContainer, collection_name: str = None, local_dir: str = None, dual_collection: bool = True):
    """Clears all data from the specified storage location."""
    print("Clearing storage (content and metadata)...")
    
    storage_use_case = container.get_storage_use_case(
        collection_name=collection_name,
        local_dir=local_dir,
        dual_collection=dual_collection
    )
    storage_use_case.clear()
    print("Storage cleared successfully.")


# --- CLI Argument Handling & Validation ---
# This section is now solely responsible for parsing CLI arguments, validating
# them, and calling the core logic functions with the correct parameters.


def setup_arg_parser():
    """Configures the argument parser for the command-line interface."""
    parser = argparse.ArgumentParser(description="Chunk documents and interact with them.")

    # Task Subparsers for better command structure (e.g., `app.py save ...`, `app.py talk ...`)
    subparsers = parser.add_subparsers(dest="task", required=True, help="Task to perform")

    # --- 'save' command ---
    parser_save = subparsers.add_parser("save", help="Chunk and save documents.")
    parser_save.add_argument("source", help="Path to the folder with markdown files.")
    parser_save.add_argument(
        "strategy",
        choices=["length_based", "structure_based", "semantic","full_doc"],
        help="Chunking strategy.",
    )
    parser_save.add_argument("--config", default="{}", help="JSON string with strategy configuration.")
    parser_save.add_argument("--clean", action="store_true", help="Clean the destination before saving.")
    parser_save.add_argument(
        "--single-collection", dest="dual_collection", action="store_false", default=True,
        help="Use single collection mode instead of dual collection (dual collection enabled by default)."
    )

    # --- 'talk' command ---
    parser_talk = subparsers.add_parser("talk", help="Ask a question about the documents.")
    parser_talk.add_argument("query", help="Query string for searching.")
    parser_talk.add_argument(
        "--top-k", type=int, default=5, help="Number of top relevant chunks to use for answer generation."
    )
    parser_talk.add_argument(
        "--candidates", type=int, default=20, help="Number of candidates to retrieve before reranking."
    )
    parser_talk.add_argument(
        "--single-collection", dest="dual_collection", action="store_false", default=True,
        help="Use single collection mode instead of dual collection (dual collection enabled by default)."
    )
    parser_talk.add_argument(
        "--no-rerank", dest="rerank", action="store_false", default=True,
        help="Disable LLM-based reranking (enabled by default)."
    )
    parser_talk.add_argument(
        "--llm-reranking", dest="llm_reranking", action="store_true", default=False,
        help="Use LLM-based reranking instead of default Encoder-based reranking."
    )

    # --- 'search' command ---
    parser_search = subparsers.add_parser("search", help="Search for relevant chunks.")
    parser_search.add_argument("query", help="Query string for searching.")
    parser_search.add_argument(
        "--top-k", type=int, default=5, help="Number of top relevant chunks to retrieve."
    )
    parser_search.add_argument(
        "--candidates", type=int, default=20, help="Number of candidates to retrieve before reranking."
    )
    parser_search.add_argument(
        "--single-collection", dest="dual_collection", action="store_false", default=True,
        help="Use single collection mode instead of dual collection (dual collection enabled by default)."
    )
    parser_search.add_argument(
        "--no-rerank", dest="rerank", action="store_false", default=True,
        help="Disable LLM-based reranking (enabled by default)."
    )
    parser_search.add_argument(
        "--llm-reranking", dest="llm_reranking", action="store_true", default=False,
        help="Use LLM-based reranking instead of default Encoder-based reranking."
    )

    # --- 'clean' command ---
    parser_clean = subparsers.add_parser("clean", help="Clean the storage location.")
    parser_clean.add_argument(
        "--single-collection", dest="dual_collection", action="store_false", default=True,
        help="Use single collection mode instead of dual collection (dual collection enabled by default)."
    )

    # --- 'delete' command (placeholder) ---
    subparsers.add_parser("delete", help="Delete specific documents (not implemented).")

    # --- Common arguments for all subparsers ---
    for sub_parser in [parser_save, parser_talk, parser_search, parser_clean]:
        storage_group = sub_parser.add_mutually_exclusive_group()
        storage_group.add_argument(
            "--chroma-collection",
            help="Use ChromaDB collection with this name.",
            default=None,
        )
        storage_group.add_argument(
            "--local-dir",
            help="Use FileSystem storage with this directory path.",
            default=None,
        )

    return parser


def main():
    """Main entry point for the script - uses DI Container for all dependencies."""
    load_dotenv()
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Default to ChromaDB if neither storage option is specified
    if not args.chroma_collection and not args.local_dir:
        args.chroma_collection = "default_collection"
    
    # Create dependency container (manages all component lifecycle)
    container = DependencyContainer()
    
    try:
        # --- Task Dispatching ---
        if args.task == "save":
            if args.clean:
                clean_storage(container, args.chroma_collection, args.local_dir, args.dual_collection)
            
            # Validate JSON config
            try:
                strategy_config_dict = json.loads(args.config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error: Invalid JSON in --config string. Details: {e}") from e
            
            # Safely convert string representations to Enum members
            strategy_params = strategy_config_dict
            if args.strategy == "length_based" and "mode" in strategy_params:
                try:
                    strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])
                except ValueError as e:
                    raise ValueError(f"Invalid 'mode' for length_based strategy: {e}") from e
            
            if args.strategy == "semantic" and "threshold_mode" in strategy_params:
                try:
                    strategy_params["threshold_mode"] = SemanticChunkingThresholdType(
                        strategy_params["threshold_mode"]
                    )
                except ValueError as e:
                    raise ValueError(f"Invalid 'threshold_mode' for semantic strategy: {e}") from e
            
            # Execute chunking
            chunk_config = ChunkingConfig(
                source_path=args.source,
                strategy=args.strategy,
                strategy_config=strategy_params,
            )
            run_chunking(chunk_config, container, args.chroma_collection, args.local_dir, args.dual_collection)
        
        elif args.task == "search":
            # Create config with reranking flag
            search_config = QueryConfig(
                query=args.query,
                top_k=args.top_k,
                num_candidates=args.candidates
            )
            search_config.use_reranking = args.rerank
            
            run_search(search_config, container, args.chroma_collection, args.local_dir, args.dual_collection, args.llm_reranking)
        
        elif args.task == "talk":
            # Create config with reranking flag
            talk_config = QueryConfig(
                query=args.query,
                top_k=args.top_k,
                num_candidates=args.candidates
            )
            talk_config.use_reranking = args.rerank
            
            run_talk(talk_config, container, args.chroma_collection, args.local_dir, args.dual_collection, args.llm_reranking)
        
        elif args.task == "clean":
            clean_storage(container, args.chroma_collection, args.local_dir, args.dual_collection)
        
        elif args.task == "delete":
            print("Delete functionality is not yet implemented.")
    
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()