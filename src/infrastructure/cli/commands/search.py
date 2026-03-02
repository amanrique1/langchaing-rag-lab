import typer
from typing import List, Any
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.infrastructure.cli.dependencies import (
    DEFAULT_TOP_K, DEFAULT_CANDIDATES, LanceArg, ChromaArg, FilesystemArg,
    StoragePathArg, CollectionArg, SingleColArg, VerboseArg,
    TopKArg, CandidatesArg, NoRerankArg, LlmRerankArg, ExpandArg
)
from src.infrastructure.cli.utils import (
    get_container, resolve_storage_config, console, setup_logging,
    log_command_params, logger, EXIT_CODE_ERROR
)

MAX_CHUNK_PREVIEW_LENGTH = 200


def search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    top_k: TopKArg = DEFAULT_TOP_K,
    candidates: CandidatesArg = DEFAULT_CANDIDATES,
    no_rerank: NoRerankArg = False,
    llm_reranking: LlmRerankArg = False,
    expand: ExpandArg = None,
    verbose: VerboseArg = False,
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """Search for relevant document chunks."""
    setup_logging(verbose)
    log_command_params("search", locals(), verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )

    if expand:
        console.print(f"[blue]ℹ[/blue] Using {expand.value.upper()} query expansion")

    container = get_container()
    use_reranking = not no_rerank

    search_use_case = container.get_search_use_case(storage_config, llm_reranking, expand)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Searching...", total=None)
        result = search_use_case.execute(query, top_k, candidates, use_reranking)

    if result.is_failure:
        logger.error(f"Search failed: {result.error}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {result.error}")
        raise typer.Exit(code=EXIT_CODE_ERROR)

    _display_chunks(result.value, query)


def _display_chunks(chunks: List[Any], query: str) -> None:
    """Display search results in a formatted table."""
    if not chunks:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(f"[green]✓[/green] Found {len(chunks)} relevant chunks.\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Content Preview")
    table.add_column("Metadata", style="dim")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.content[:MAX_CHUNK_PREVIEW_LENGTH] + (
            "..." if len(chunk.content) > MAX_CHUNK_PREVIEW_LENGTH else ""
        )
        metadata_str = ", ".join(f"{k}: {v}" for k, v in chunk.metadata.items())
        table.add_row(str(i), preview, metadata_str)

    console.print(table)