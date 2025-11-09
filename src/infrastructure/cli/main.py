import json
import argparse
from dotenv import load_dotenv
from application.use_cases.chunking_use_case import ChunkingUseCase
from application.use_cases.storage_use_case import StorageUseCase
from application.use_cases.talk_use_case import TalkUseCase
from infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)

from domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType,
    StorageType,
)


def setup_common_arguments(parser):
    parser.add_argument(
        "source", nargs="?", help="Path to the folder with markdown files."
    )
    parser.add_argument(
        "strategy",
        nargs="?",
        help="Chunking strategy to use.",
        choices=["length_based", "structure_based", "semantic"],
    )
    parser.add_argument(
        "--config",
        help="JSON string with the strategy configuration.",
        default="{}",
    )
    parser.add_argument(
        "--query",
        help="Query string for searching chunks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of top relevant chunks to retrieve.",
        default=5,
    )


def run_chunking(args, store_type: StorageType):
    document_loader = MarkdownDocumentLoader()
    chunking_use_case = ChunkingUseCase(document_loader)
    output = args.output_dir if args.local else args.collection_name
    storage_use_case = StorageUseCase(store_type, output)
    strategy_config = json.loads(args.config)

    if args.strategy == "length_based" and "mode" in strategy_config:
        strategy_config["mode"] = LengthBasedChunkingMode(strategy_config["mode"])

    if args.strategy == "semantic" and "breakpoint_threshold_type" in strategy_config:
        strategy_config["breakpoint_threshold_type"] = SemanticChunkingThresholdType(
            strategy_config["breakpoint_threshold_type"]
        )

    chunks = chunking_use_case.execute(
        args.source, args.strategy, strategy_config)
    storage_use_case.save(chunks)

    print(f"Successfully processed {len(chunks)} documents.")


def run_talk(args, store_type: StorageType):
    storage_use_case = StorageUseCase(store_type, args.collection_name)
    talk_use_case = TalkUseCase()

    print(f"Question: {args.query}")

    relevant_chunks = storage_use_case.search(args.query, args.top_k)
    answer = talk_use_case.execute(args.query, relevant_chunks)

    print(f"\nAnswer: {answer}")

def clean(storage_type: StorageType, output_loc: str):
    storage = StorageUseCase(storage_type, output_loc)
    storage.clear()

def validate_args(args):
    if args.task == "save":
        if not args.source or not args.strategy:
            raise argparse.ArgumentError(
                None,
                "Error: 'source' and 'strategy' arguments are required for the 'save' task.",
            )
        if args.strategy == "length_based":
            config = json.loads(args.config)
            if "chunk_size" not in config or "chunk_overlap" not in config:
                raise argparse.ArgumentError(
                    None,
                    "Error: 'chunk_size' and 'chunk_overlap' are required in the config for the 'length_based' strategy.",
                )
    elif args.task in ["search", "talk"]:
        if not args.query:
            raise argparse.ArgumentError(
                None,
                f"Error: '--query' argument is required for the '{args.task}' task.",
            )

    if args.local and args.collection_name != "default_collection":
        raise argparse.ArgumentError(
            None,
            "Error: '--collection-name' should not be used with '--local'. Use '--output-dir' to specify the output directory.",
        )

    if not args.local and args.output_dir != "output_chunks":
        raise argparse.ArgumentError(
            None,
            "Error: '--output-dir' should only be used with '--local'. Use '--collection-name' for ChromaDB collections.",
        )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Chunk documents using different strategies."
    )
    # Add the new 'task' argument
    parser.add_argument(
        "task",
        help="Task to perform.",
        choices=["save", "search", "delete", "clean", "talk"],
    )

    setup_common_arguments(parser)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Store chunks locally in the file system.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to save the chunks (used with --local).",
        default="output_chunks",
    )
    parser.add_argument(
        "--collection-name",
        help="Name of the ChromaDB collection.",
        default="default_collection",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the collection before saving new chunks.",
    )

    args = parser.parse_args()
    validate_args(args)

    store_type = StorageType.LOCAL if args.local else StorageType.CHROMA
    if args.task == "clean":
        output = args.output_dir if args.local else args.collection_name
        clean(store_type, output)
    elif args.task == "save":
        if args.clean:
            output = args.output_dir if args.local else args.collection_name
            clean(store_type, output)
            if not args.local:
                print(f"Cleared collection '{args.collection_name}'.")
        storage_type = StorageType.LOCAL if args.local else StorageType.CHROMA
        run_chunking(args, storage_type)
    elif args.task == "search":
        storage_use_case = StorageUseCase(store_type, args.collection_name)
        relevant_chunks = storage_use_case.search(args.query, args.top_k)
        if relevant_chunks:
            print(f"Found {len(relevant_chunks)} relevant chunks:")
            for i, chunk in enumerate(relevant_chunks):
                print(f"\n--- Chunk {i+1} ---")
                print(f"Content: {chunk.content}")
                print(f"Metadata: {chunk.metadata}")
        else:
            print("No relevant chunks found.")
    elif args.task == "talk":
        run_talk(args, store_type)
    elif args.task == "delete":
        print("Delete functionality not yet implemented.")


if __name__ == "__main__":
    main()
