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

    if clean:
        _clean_storage(storage_config, force=force)

    chunk_config = ChunkingConfig(source, strategy, config)
    _run_chunking(chunk_config, storage_config, verbose)

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
    _clean_storage(storage_config, force=force)

# --- Internal Helpers ---

def _run_chunking(chunk_config, storage_config, verbose):
    container = get_container()
    strategy_params = chunk_config.strategy_config.copy()
    convert_strategy_enums(chunk_config.strategy, strategy_params)

    console.print(f"[blue]ℹ[/blue] Processing [bold]{chunk_config.source_path}[/bold]...")

    try:
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("Chunking documents...", total=None)
            chunking_use_case = container.get_chunking_use_case()
            chunks = chunking_use_case.execute(chunk_config.source_path, chunk_config.strategy, strategy_params)

            progress.update(task, description="Saving chunks...")
            storage_use_case = container.get_storage_use_case(storage_config)
            storage_use_case.save(chunks)

        console.print(f"[green]✓[/green] Successfully processed {len(chunks)} chunks.")
    except Exception as e:
        logger.error(f"Chunking failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)

def _clean_storage(storage_config, force):
    if not force and not typer.confirm("⚠️  Delete all stored data?"):
        raise typer.Exit()

    try:
        container = get_container()
        console.print("[blue]Clearing storage...[/blue]")
        container.get_storage_use_case(storage_config).clear()
        console.print("[green]✓[/green] Storage cleared.")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)