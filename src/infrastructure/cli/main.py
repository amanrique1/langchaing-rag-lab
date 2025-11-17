import json
import argparse
import sys

from dotenv import load_dotenv

from src.application.use_cases.chunking_use_case import ChunkingUseCase
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import (
    ChromaChunkStore,
)
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import (
    FileSystemChunkStore,
)
from src.infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)
from src.infrastructure.adapters.language_models.google_genai_embedding_model import (
    GoogleGenAIEmbeddingModel,
)
from src.infrastructure.adapters.language_models.google_genai_language_model import (
    GoogleGenAILanguageModel,
)
from src.domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType
)
from src.domain.models.cli_config_classes import ChunkingConfig, TalkConfig


def run_chunking(
    chunk_config: ChunkingConfig,
    chunking_use_case: ChunkingUseCase,
    storage_use_case: StorageUseCase,
):
    """
    Loads documents, chunks them according to a strategy, and saves them.
    """
    # Make a copy to avoid mutating the original dictionary
    strategy_params = chunk_config.strategy_config.copy()

    # Safely convert string representations to Enum members
    if chunk_config.strategy == "length_based" and "mode" in strategy_params:
        try:
            strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])
        except ValueError as e:
            raise ValueError(f"Invalid 'mode' for length_based strategy: {e}") from e

    if chunk_config.strategy == "semantic" and "breakpoint_threshold_type" in strategy_params:
        try:
            strategy_params["breakpoint_threshold_type"] = SemanticChunkingThresholdType(
                strategy_params["breakpoint_threshold_type"]
            )
        except ValueError as e:
            raise ValueError(f"Invalid 'breakpoint_threshold_type' for semantic strategy: {e}") from e

    print(f"Running chunking strategy '{chunk_config.strategy}' on '{chunk_config.source_path}'...")
    chunks = chunking_use_case.execute(
        chunk_config.source_path, chunk_config.strategy, strategy_params
    )
    storage_use_case.save(chunks)

    print(f"Successfully processed and saved {len(chunks)} chunks.")


def run_talk(talk_config: TalkConfig, talk_use_case: TalkUseCase):
    """
    Searches for relevant chunks and generates an answer based on a query.
    """
    print(f"Question: {talk_config.query}")

    answer = talk_use_case.execute(talk_config.query, talk_config.top_k)

    print(f"\nAnswer: {answer}")


def run_search(talk_config: TalkConfig, storage_use_case: StorageUseCase):
    """
    Performs a search for relevant chunks and displays them.
    """
    relevant_chunks = storage_use_case.search(talk_config.query, talk_config.top_k)

    if relevant_chunks:
        print(f"Found {len(relevant_chunks)} relevant chunks for query: '{talk_config.query}'")
        for i, chunk in enumerate(relevant_chunks):
            print(f"\n--- Chunk {i+1} (Score: {chunk.score if hasattr(chunk, 'score') else 'N/A'}) ---")
            print(f"Content: {chunk.content}")
            print(f"Metadata: {chunk.metadata}")
    else:
        print("No relevant chunks found.")


def clean_storage(storage_use_case: StorageUseCase):
    """Clears all data from the specified storage location."""
    print("Clearing storage...")
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
        choices=["length_based", "structure_based", "semantic"],
        help="Chunking strategy.",
    )
    parser_save.add_argument("--config", default="{}", help="JSON string with strategy configuration.")
    parser_save.add_argument("--clean", action="store_true", help="Clean the destination before saving.")

    # --- 'talk' command ---
    parser_talk = subparsers.add_parser("talk", help="Ask a question about the documents.")
    parser_talk.add_argument("query", help="Query string for searching.")
    parser_talk.add_argument(
        "--top-k", type=int, default=5, help="Number of top relevant chunks to retrieve."
    )

    # --- 'search' command ---
    parser_search = subparsers.add_parser("search", help="Search for relevant chunks.")
    parser_search.add_argument("query", help="Query string for searching.")
    parser_search.add_argument(
        "--top-k", type=int, default=5, help="Number of top relevant chunks to retrieve."
    )

    # --- 'clean' command ---
    subparsers.add_parser("clean", help="Clean the storage location.")

    # --- 'delete' command (placeholder) ---
    subparsers.add_parser("delete", help="Delete specific documents (not implemented).")

    # --- Common arguments for all subparsers ---
    for sub_parser in [parser_save, parser_talk, parser_search, subparsers.choices["clean"]]:
        storage_group = sub_parser.add_mutually_exclusive_group()
        storage_group.add_argument(
            "--local-dir", help="Use local file system storage at this directory.", default="output_chunks"
        )
        storage_group.add_argument(
            "--chroma-collection",
            help="Use ChromaDB collection with this name.",
            default="default_collection",
        )

    return parser


def main():
    """Main entry point for the script."""
    load_dotenv()
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Determine storage configuration from common arguments
    # The mutually exclusive group ensures only one is chosen.
    use_local = "local_dir" in args and args.local_dir != "output_chunks"
    if use_local:
        chunk_store = FileSystemChunkStore(args.local_dir)
    else:
        embedding_model = GoogleGenAIEmbeddingModel()
        chunk_store = ChromaChunkStore(args.chroma_collection, embedding_model)

    # Instantiate adapters
    document_loader = MarkdownDocumentLoader()
    language_model = GoogleGenAILanguageModel()

    # Instantiate use cases and inject dependencies
    chunking_use_case = ChunkingUseCase(document_loader)
    storage_use_case = StorageUseCase(chunk_store)
    talk_use_case = TalkUseCase(language_model, chunk_store)

    try:
        # --- Task Dispatching ---
        if args.task == "save":
            if args.clean:
                clean_storage(storage_use_case)

            # Validate JSON config
            try:
                strategy_config_dict = json.loads(args.config)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error: Invalid JSON in --config string. Details: {e}") from e

            chunk_config = ChunkingConfig(
                source_path=args.source,
                strategy=args.strategy,
                strategy_config=strategy_config_dict,
            )
            run_chunking(chunk_config, chunking_use_case, storage_use_case)

        elif args.task in ["talk", "search"]:
            talk_config = TalkConfig(query=args.query, top_k=args.top_k)
            if args.task == "talk":
                run_talk(talk_config, talk_use_case)
            else:
                run_search(talk_config, storage_use_case)

        elif args.task == "clean":
            clean_storage(storage_use_case)

        elif args.task == "delete":
            print("Delete functionality is not yet implemented.")

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()