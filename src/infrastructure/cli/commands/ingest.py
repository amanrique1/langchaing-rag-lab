import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.domain.models.config_classes import ChunkingConfig
from src.infrastructure.cli.dependencies import (
    LanceArg, ChromaArg, FilesystemArg, StoragePathArg,
    CollectionArg, SingleColArg, VerboseArg
)
from src.infrastructure.cli.utils import (
    console, logger, EXIT_CODE_ERROR,
    setup_logging, log_command_params,
    get_container, validate_json, validate_source_path,
    resolve_storage_config, convert_strategy_enums
)

def save(
    source: Annotated[str, typer.Argument(help="Path to source document(s).")],
    strategy: Annotated[str, typer.Argument(help="Chunking strategy (e.g., length_based, semantic).")],
    config: Annotated[str, typer.Option("--config", callback=validate_json, help="JSON config.")] = "{}",
    clean: Annotated[bool, typer.Option("--clean", help="Clean storage before saving.")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="Skip confirmation.")] = False,
    verbose: VerboseArg = False,
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """Chunk documents from SOURCE and save them to storage."""
    setup_logging(verbose)
    log_command_params("save", locals(), verbose)

    validate_source_path(source)
    storage_config = resolve_storage_config(collection, storage_path, lance, chroma, filesystem, single_collection)

    # Resolve dependencies once
    container = get_container()
    ingestion_use_case = container.get_ingestion_use_case(storage_config)

    if clean:
        _clean_storage(ingestion_use_case, force=force)

    chunk_config = ChunkingConfig(source, strategy, config)
    _run_ingestion(ingestion_use_case, chunk_config, verbose)

def clean_command(
    force: Annotated[bool, typer.Option("--force", "-f")] = False,
    verbose: VerboseArg = False,
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """Clean/clear all data from storage."""
    setup_logging(verbose)
    log_command_params("clean", locals(), verbose)

    storage_config = resolve_storage_config(collection, storage_path, lance, chroma, filesystem, single_collection)

    # Resolve dependencies
    container = get_container()
    ingestion_use_case = container.get_ingestion_use_case(storage_config)

    _clean_storage(ingestion_use_case, force=force)

# --- Internal Helpers ---

def _run_ingestion(ingestion_use_case, chunk_config, verbose):
    """
    Executes the ingestion pipeline (Load -> Chunk -> Save) using the
    provided IngestionUseCase.
    """
    strategy_params = chunk_config.strategy_config.copy()
    convert_strategy_enums(chunk_config.strategy, strategy_params)

    console.print(f"[blue]ℹ[/blue] Processing [bold]{chunk_config.source_path}[/bold]...")

    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Ingesting documents...", total=None)

            # The use case handles loading, chunking, and saving atomically
            chunks = ingestion_use_case.ingest(
                chunk_config.source_path,
                chunk_config.strategy,
                strategy_params
            )

            progress.update(task, completed=100)

        console.print(f"[green]✓[/green] Successfully processed and saved {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)

def _clean_storage(ingestion_use_case, force):
    """
    Clears storage using the provided IngestionUseCase.
    """
    if not force and not typer.confirm("⚠️  Delete all stored data?"):
        raise typer.Exit()

    try:
        console.print("[blue]Clearing storage...[/blue]")
        ingestion_use_case.clear_storage()
        console.print("[green]✓[/green] Storage cleared.")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)