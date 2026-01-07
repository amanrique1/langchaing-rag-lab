import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated
from dotenv import load_dotenv
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.syntax import Syntax

from src.application.dependency_container import DependencyContainer
from src.domain.models.enums import (
    LengthBasedChunkingMode,
    SemanticChunkingThresholdType,
    QueryExpansionStrategy,
    StorageType
)
from src.domain.models.config_classes import ChunkingConfig, QueryConfig, StorageConfig

# Constants
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 20
MAX_CHUNK_PREVIEW_LENGTH = 200
EXIT_CODE_ERROR = 1

# Initialize Typer App and Rich Console
app = typer.Typer(
    help="Document Chunking and Retrieval CLI - Chunk documents and interact with them via natural language.",
    add_completion=False
)
console = Console()

# Configure logging
def setup_logging(verbose: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

logger = logging.getLogger(__name__)

# --- Helpers ---

def log_command_params(command_name: str, params: Dict[str, Any], verbose: bool = False) -> None:
    """Log command parameters for debugging."""
    filtered_params = {k: v for k, v in params.items() if v is not None and v is not False}

    logger.debug(f"Command: {command_name}")
    logger.debug(f"Parameters: {json.dumps(filtered_params, indent=2, default=str)}")

    if verbose:
        param_json = json.dumps(filtered_params, indent=2, default=str)
        syntax = Syntax(param_json, "json", theme="monokai", line_numbers=False)
        console.print(Panel(
            syntax,
            title=f"[bold cyan]Command: {command_name}[/bold cyan]",
            subtitle="[dim]Parameters[/dim]",
            border_style="blue"
        ))


def get_container() -> DependencyContainer:
    """Lazy load the dependency container."""
    try:
        logger.debug("Initializing dependency container")
        return DependencyContainer()
    except Exception as e:
        logger.error(f"Failed to initialize dependency container: {e}")
        console.print(f"[red]Error:[/red] Failed to initialize application: {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def validate_json(value: str) -> Dict[str, Any]:
    """Validate and parse JSON input strings."""
    if not value or value == "{}":
        logger.debug("No JSON config provided, using empty dict")
        return {}

    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise typer.BadParameter("JSON must be an object/dictionary")
        logger.debug(f"Parsed JSON config: {parsed}")
        return parsed
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {e}")
        raise typer.BadParameter(f"Invalid JSON format: {e}")


def validate_source_path(source: str) -> Path:
    """Validate that the source path exists."""
    path = Path(source)
    logger.debug(f"Validating source path: {path.absolute()}")

    if not path.exists():
        logger.error(f"Source path does not exist: {path.absolute()}")
        raise typer.BadParameter(f"Source path does not exist: {source}")

    logger.debug(f"Source path validated: {path.absolute()}")
    return path


def resolve_storage_config(
    collection: Optional[str],
    storage_path: Optional[str],
    lance: bool,
    chroma: bool,
    filesystem: bool,
    single_collection: bool
) -> StorageConfig:
    """
    Resolve storage configuration from CLI arguments.

    Args:
        collection: Collection/table name for vector stores
        storage_path: Custom storage directory path (None = use store default)
        lance: Use LanceDB explicitly
        chroma: Use ChromaDB
        filesystem: Use filesystem storage
        single_collection: Whether to use single collection mode

    Returns:
        StorageConfig: The resolved storage configuration
    """
    dual_collection = not single_collection

    # Determine storage type (priority: filesystem > chroma > lance)
    if filesystem:
        storage_type = StorageType.FILESYSTEM
        backend_name = "Local Filesystem"
    elif chroma:
        storage_type = StorageType.CHROMA
        backend_name = "ChromaDB"
    else:  # lance or default
        storage_type = StorageType.LANCE
        backend_name = "LanceDB"

    coll_name = collection or "default_collection"

    logger.debug(
        f"Resolved storage - type={storage_type.name}, "
        f"persist_dir={storage_path or 'default'}, "
        f"collection={coll_name}, dual_collection={dual_collection}"
    )

    try:
        config = StorageConfig(
            storage_type=storage_type,
            collection_name=coll_name,
            persist_directory=storage_path,
            dual_collection=dual_collection
        )
        logger.debug(f"Storage config created: {config}")

        # Display storage backend being used
        location_info = f" at {storage_path}" if storage_path else ""
        console.print(f"[dim]Storage: {backend_name}{location_info}[/dim]")

        return config
    except ValueError as e:
        logger.error(f"Storage configuration error: {e}")
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def convert_strategy_enums(strategy: str, strategy_params: Dict[str, Any]) -> None:
    """Convert string parameters to enum types for chunking strategies."""
    logger.debug(f"Converting enums for strategy '{strategy}' with params: {strategy_params}")

    if strategy == "length_based" and "mode" in strategy_params:
        try:
            original_value = strategy_params["mode"]
            strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])
            logger.debug(f"Converted mode: {original_value} -> {strategy_params['mode']}")
        except ValueError as e:
            logger.error(f"Invalid mode for length_based strategy: {e}")
            raise typer.BadParameter(f"Invalid mode value: {strategy_params['mode']}")

    if strategy == "semantic" and "threshold_mode" in strategy_params:
        try:
            original_value = strategy_params["threshold_mode"]
            strategy_params["threshold_mode"] = SemanticChunkingThresholdType(
                strategy_params["threshold_mode"]
            )
            logger.debug(f"Converted threshold_mode: {original_value} -> {strategy_params['threshold_mode']}")
        except ValueError as e:
            logger.error(f"Invalid threshold_mode for semantic strategy: {e}")
            raise typer.BadParameter(f"Invalid threshold_mode value: {strategy_params['threshold_mode']}")


# --- Reusable Argument Annotations ---

# Storage type flags
LanceArg = Annotated[
    bool,
    typer.Option("--lance", help="Use LanceDB storage (allows custom path with --storage-path).")
]
ChromaArg = Annotated[
    bool,
    typer.Option("--chroma", help="Use ChromaDB storage.")
]
FilesystemArg = Annotated[
    bool,
    typer.Option("--filesystem", help="Use local filesystem JSON storage.")
]

# Storage configuration
StoragePathArg = Annotated[
    Optional[str],
    typer.Option("--storage-path", "-p", help="Custom storage directory path (omit to use store default).")
]
CollectionArg = Annotated[
    Optional[str],
    typer.Option("--collection", "-c", help="Collection/table name (default: 'default_collection').")
]
SingleColArg = Annotated[
    bool,
    typer.Option("--single-collection", help="Use single collection mode (default: dual collection).")
]

# Query parameters
TopKArg = Annotated[
    int,
    typer.Option("--top-k", "-k", help="Number of top results to return.", min=1)
]
CandidatesArg = Annotated[
    int,
    typer.Option("--candidates", "-n", help="Number of candidates before reranking.", min=1)
]
NoRerankArg = Annotated[
    bool,
    typer.Option("--no-rerank", help="Disable reranking step.")
]
LlmRerankArg = Annotated[
    bool,
    typer.Option("--llm-rerank", help="Use LLM-based reranking.")
]
ExpandArg = Annotated[
    Optional[QueryExpansionStrategy],
    typer.Option("--expand", "-e", help="Query expansion strategy (hyde, stepback, subqueries, zero_shot).")
]
VerboseArg = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable verbose output.")
]

# --- Business Logic Runners ---

def _run_chunking(
    chunk_config: ChunkingConfig,
    storage_config: StorageConfig,
    verbose: bool = False
) -> None:
    """Execute chunking logic with progress indication."""
    logger.debug(f"Starting chunking process with config: {chunk_config}")
    container = get_container()
    strategy_params = chunk_config.strategy_config.copy()

    convert_strategy_enums(chunk_config.strategy, strategy_params)

    console.print(
        f"[blue]ℹ[/blue] Processing [bold]{chunk_config.source_path}[/bold] "
        f"using [bold]{chunk_config.strategy}[/bold] strategy..."
    )

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Chunking documents...", total=None)

            logger.debug("Executing chunking use case")
            chunking_use_case = container.get_chunking_use_case()
            chunks = chunking_use_case.execute(
                chunk_config.source_path,
                chunk_config.strategy,
                strategy_params
            )

            logger.debug(f"Generated {len(chunks)} chunks")
            progress.update(task, description="Saving chunks...")

            storage_use_case = container.get_storage_use_case(storage_config)
            storage_use_case.save(chunks)
            logger.debug("Chunks saved to storage")

        console.print(f"[green]✓[/green] Successfully processed and saved {len(chunks)} chunks.")
        logger.info(f"Chunked {len(chunks)} documents successfully")

    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] Chunking failed: {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def _clean_storage(storage_config: StorageConfig, force: bool = False) -> None:
    """Clean the storage with optional confirmation."""
    logger.debug(f"Cleaning storage with force={force}")

    if not force:
        confirm = typer.confirm("⚠️  This will delete all stored data. Continue?")
        if not confirm:
            logger.info("Storage cleaning cancelled by user")
            console.print("[yellow]Operation cancelled.[/yellow]")
            raise typer.Exit()

    try:
        container = get_container()
        console.print("[blue]Clearing storage...[/blue]")
        storage_use_case = container.get_storage_use_case(storage_config)
        storage_use_case.clear()
        console.print("[green]✓[/green] Storage cleared successfully.")
        logger.info("Storage cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear storage: {e}")
        console.print(f"[red]Error:[/red] Failed to clear storage: {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def _execute_query(
    query: str,
    storage_config: StorageConfig,
    top_k: int,
    candidates: int,
    use_reranking: bool,
    llm_reranking: bool,
    expand: Optional[QueryExpansionStrategy],
    use_case_getter: str,
    verbose: bool = False
) -> Any:
    """Execute a query operation (search or talk)."""
    logger.debug(
        f"Executing query - type={use_case_getter}, query='{query}', "
        f"top_k={top_k}, candidates={candidates}, use_reranking={use_reranking}, "
        f"llm_reranking={llm_reranking}, expand={expand}"
    )

    container = get_container()

    if expand:
        console.print(f"[blue]ℹ[/blue] Using {expand.value.upper()} query expansion")
        logger.debug(f"Query expansion enabled: {expand.value}")

    try:
        if use_case_getter == "talk":
            logger.debug("Getting talk use case")
            use_case = container.get_talk_use_case(storage_config, llm_reranking, expand)
        else:
            logger.debug("Getting search use case")
            use_case = container.get_search_use_case(storage_config, llm_reranking, expand)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Processing query...", total=None)

            logger.debug("Executing use case")
            result = use_case.execute(
                query=query,
                top_k=top_k,
                num_candidates=candidates,
                use_reranking=use_reranking
            )

        logger.debug(f"Query executed successfully, result type: {type(result).__name__}")
        return result

    except Exception as e:
        logger.error(f"Query execution failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] Query failed: {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def _display_chunks(chunks: List[Any], query: str) -> None:
    """Display search results in a formatted table."""
    logger.debug(f"Displaying {len(chunks)} chunks")

    if not chunks:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        logger.warning("No chunks found for query")
        return

    console.print(f"[green]✓[/green] Found {len(chunks)} relevant chunks for: [bold]{query}[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Content Preview", overflow="fold")
    table.add_column("Metadata", style="dim", overflow="fold")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.content[:MAX_CHUNK_PREVIEW_LENGTH]
        if len(chunk.content) > MAX_CHUNK_PREVIEW_LENGTH:
            preview += "..."

        metadata_str = ", ".join(f"{k}: {v}" for k, v in chunk.metadata.items())
        table.add_row(str(i), preview, metadata_str)

    console.print(table)
    logger.debug("Chunks displayed successfully")


# --- Commands ---

@app.command()
def save(
    source: Annotated[str, typer.Argument(help="Path to source document(s).")],
    strategy: Annotated[str, typer.Argument(help="Chunking strategy (e.g., length_based, semantic).")],
    config: Annotated[
        str,
        typer.Option("--config", callback=validate_json, help="JSON configuration for strategy.")
    ] = "{}",
    clean: Annotated[bool, typer.Option("--clean", help="Clean storage before saving.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompts.")] = False,
    verbose: VerboseArg = False,
    # Storage Type
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    # Storage Configuration
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """
    Chunk documents from SOURCE and save them to storage.

    Storage backends:
    - LanceDB (default): Fast hybrid search with vector + BM25
    - ChromaDB: Alternative vector store (use --chroma)
    - Filesystem: Local JSON storage (use --filesystem)

    Examples:
        # Default LanceDB with default location
        $ cli save ./docs.pdf length_based --config '{"chunk_size": 512}'

        # LanceDB with custom location
        $ cli save ./docs.pdf length_based --lance --storage-path ./custom_db

        # ChromaDB with custom collection
        $ cli save ./docs.pdf length_based --chroma --collection my_docs

        # Filesystem storage with custom path
        $ cli save ./docs.pdf length_based --filesystem --storage-path ./chunks
    """
    setup_logging(verbose)

    log_command_params("save", {
        "source": source,
        "strategy": strategy,
        "config": config,
        "clean": clean,
        "force": force,
        "lance": lance,
        "chroma": chroma,
        "filesystem": filesystem,
        "storage_path": storage_path,
        "collection": collection,
        "single_collection": single_collection,
    }, verbose)

    validate_source_path(source)
    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )

    if clean:
        _clean_storage(storage_config, force=force)

    chunk_config = ChunkingConfig(source, strategy, config)
    _run_chunking(chunk_config, storage_config, verbose)


@app.command()
def talk(
    query: Annotated[str, typer.Argument(help="Question to ask your documents.")],
    # Ranking Args
    top_k: TopKArg = DEFAULT_TOP_K,
    candidates: CandidatesArg = DEFAULT_CANDIDATES,
    no_rerank: NoRerankArg = False,
    llm_reranking: LlmRerankArg = False,
    # Expansion Args
    expand: ExpandArg = None,
    verbose: VerboseArg = False,
    # Storage Type
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    # Storage Configuration
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """
    Ask a question and get an AI-generated answer based on your documents.
    """
    setup_logging(verbose)

    log_command_params("talk", {
        "query": query,
        "top_k": top_k,
        "candidates": candidates,
        "no_rerank": no_rerank,
        "llm_reranking": llm_reranking,
        "expand": expand,
        "lance": lance,
        "chroma": chroma,
        "filesystem": filesystem,
        "storage_path": storage_path,
        "collection": collection,
        "single_collection": single_collection,
    }, verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )
    use_reranking = not no_rerank

    console.print(f"\n[bold]Question:[/bold] {query}\n")

    answer = _execute_query(
        query=query,
        storage_config=storage_config,
        top_k=top_k,
        candidates=candidates,
        use_reranking=use_reranking,
        llm_reranking=llm_reranking,
        expand=expand,
        use_case_getter="talk",
        verbose=verbose
    )

    console.print("\n[bold yellow]Answer:[/bold yellow]")
    console.print(answer)
    logger.debug("Talk command completed successfully")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    # Ranking Args
    top_k: TopKArg = DEFAULT_TOP_K,
    candidates: CandidatesArg = DEFAULT_CANDIDATES,
    no_rerank: NoRerankArg = False,
    llm_reranking: LlmRerankArg = False,
    # Expansion Args
    expand: ExpandArg = None,
    verbose: VerboseArg = False,
    # Storage Type
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    # Storage Configuration
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """
    Search for relevant document chunks without generating an answer.
    """
    setup_logging(verbose)

    log_command_params("search", {
        "query": query,
        "top_k": top_k,
        "candidates": candidates,
        "no_rerank": no_rerank,
        "llm_reranking": llm_reranking,
        "expand": expand,
        "lance": lance,
        "chroma": chroma,
        "filesystem": filesystem,
        "storage_path": storage_path,
        "collection": collection,
        "single_collection": single_collection,
    }, verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )
    use_reranking = not no_rerank

    chunks = _execute_query(
        query=query,
        storage_config=storage_config,
        top_k=top_k,
        candidates=candidates,
        use_reranking=use_reranking,
        llm_reranking=llm_reranking,
        expand=expand,
        use_case_getter="search",
        verbose=verbose
    )

    _display_chunks(chunks, query)
    logger.debug("Search command completed successfully")


@app.command()
def clean(
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation prompt.")] = False,
    verbose: VerboseArg = False,
    # Storage Type
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    # Storage Configuration
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """
    Clean/clear all data from storage.
    """
    setup_logging(verbose)

    log_command_params("clean", {
        "force": force,
        "lance": lance,
        "chroma": chroma,
        "filesystem": filesystem,
        "storage_path": storage_path,
        "collection": collection,
        "single_collection": single_collection,
    }, verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )
    _clean_storage(storage_config, force=force)
    logger.debug("Clean command completed successfully")


@app.command()
def info(
    verbose: VerboseArg = False,
    # Storage Type
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    # Storage Configuration
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """
    Display information about the current storage configuration.
    """
    setup_logging(verbose)

    log_command_params("info", {
        "lance": lance,
        "chroma": chroma,
        "filesystem": filesystem,
        "storage_path": storage_path,
        "collection": collection,
        "single_collection": single_collection,
    }, verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )

    # Get store defaults by importing the constants
    from src.infrastructure.adapters.chunk_stores.lance_chunk_store import DEFAULT_PERSIST_DIRECTORY as LANCE_DEFAULT
    from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import DEFAULT_PERSIST_DIRECTORY as CHROMA_DEFAULT
    from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import DEFAULT_PERSIST_DIRECTORY as FS_DEFAULT

    if storage_config.persist_directory:
        persist_dir = storage_config.persist_directory
        custom_location = " (custom)"
    else:
        if storage_config.storage_type.name == "LANCE":
            persist_dir = LANCE_DEFAULT
        elif storage_config.storage_type.name == "CHROMA":
            persist_dir = CHROMA_DEFAULT
        else:
            persist_dir = FS_DEFAULT
        custom_location = " (default)"

    console.print("\n[bold]Storage Configuration:[/bold]")
    console.print(f"  Backend: {storage_config.storage_type.name}")
    console.print(f"  Collection Mode: {'Single' if not storage_config.dual_collection else 'Dual'}")
    console.print(f"  Persist Directory: {persist_dir}{custom_location}")
    console.print(f"  Collection Name: {storage_config.collection_name}")

    # Display backend-specific info
    if storage_config.storage_type.name == "LANCE":
        console.print("  [dim]Using LanceDB with hybrid search (vector + BM25)[/dim]")
    elif storage_config.storage_type.name == "CHROMA":
        console.print("  [dim]Using ChromaDB with dual collection strategy[/dim]")
    elif storage_config.storage_type.name == "FILESYSTEM":
        console.print("  [dim]Using local filesystem JSON storage[/dim]")
    console.print()
    logger.debug("Info command completed successfully")


@app.callback()
def main_callback():
    """
    Document Chunking and Retrieval CLI

    A powerful tool for chunking documents and interacting with them using natural language.

    Storage Backends:
    - LanceDB (default): High-performance hybrid search
    - ChromaDB: Use --chroma flag
    - Filesystem: Use --filesystem flag

    Custom storage locations can be specified with --storage-path
    """
    load_dotenv()


if __name__ == "__main__":
    app()