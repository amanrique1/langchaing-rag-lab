import json
import argparse
from dotenv import load_dotenv
from application.use_cases.chunking_use_case import ChunkingUseCase
from infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)
from infrastructure.adapters.chunk_stores.file_system_chunk_store import (
    FileSystemChunkStore,
)
from application.ports.chunk_store import ChunkStore

from domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType,
    DocumentLoaderMode,
)
from infrastructure.adapters.chunk_stores.chroma_chunk_store import (
    ChromaChunkStore,
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
        "--loader-mode",
        help="Document loader mode.",
        choices=[mode.value for mode in DocumentLoaderMode],
        default=DocumentLoaderMode.SINGLE.value,
    )


def run_chunking(args, chunk_store: ChunkStore):
    document_loader = MarkdownDocumentLoader()
    use_case = ChunkingUseCase(document_loader, chunk_store)
    strategy_config = json.loads(args.config)

    if args.strategy == "length_based" and "mode" in strategy_config:
        strategy_config["mode"] = LengthBasedChunkingMode(strategy_config["mode"])

    if args.strategy == "semantic" and "breakpoint_threshold_type" in strategy_config:
        strategy_config["breakpoint_threshold_type"] = SemanticChunkingThresholdType(
            strategy_config["breakpoint_threshold_type"]
        )

    chunks = use_case.execute(
        args.source, args.strategy, strategy_config, loader_mode=args.loader_mode
    )

    if isinstance(chunk_store, FileSystemChunkStore):
        chunk_store.save(chunks)
    elif isinstance(chunk_store, ChromaChunkStore):
        for chunk in chunks:
            chunk_store.add(chunk)

    print(f"Successfully processed {len(chunks)} documents.")


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Chunk documents using different strategies."
    )
    setup_common_arguments(parser)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Store chunks locally in the file system.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean the chunk store before processing.",
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

    args = parser.parse_args()

    if args.local:
        chunk_store = FileSystemChunkStore(args.output_dir)
    else:
        chunk_store = ChromaChunkStore(args.collection_name)

    if args.clean:
        chunk_store.clear()
        if isinstance(chunk_store, ChromaChunkStore):
            print(f"Successfully cleared collection: {args.collection_name}")
        else:
            print(f"Successfully cleared output directory: {args.output_dir}")

    if args.source:
        run_chunking(args, chunk_store)


if __name__ == "__main__":
    main()
