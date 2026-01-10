import typer
from typing import Any, List
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

def talk(
    query: Annotated[str, typer.Argument(help="Question to ask.")],
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
    """Ask a question and get an AI-generated answer."""
    setup_logging(verbose)
    log_command_params("talk", locals(), verbose)

    storage_config = resolve_storage_config(collection, storage_path, lance, chroma, filesystem, single_collection)
    console.print(f"\n[bold]Question:[/bold] {query}\n")

    answer = _execute_query(
        query, storage_config, top_k, candidates,
        not no_rerank, llm_reranking, expand, "talk", verbose
    )
    console.print("\n[bold yellow]Answer:[/bold yellow]")
    console.print(answer)

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

    storage_config = resolve_storage_config(collection, storage_path, lance, chroma, filesystem, single_collection)

    chunks = _execute_query(
        query, storage_config, top_k, candidates,
        not no_rerank, llm_reranking, expand, "search", verbose
    )
    _display_chunks(chunks, query)

# --- Internal Helpers ---

def _execute_query(
    query: str, storage_config, top_k: int, candidates: int,
    use_reranking: bool, llm_rerank: bool, expand, use_case_type: str, verbose: bool
) -> Any:
    container = get_container()
    if expand:
        console.print(f"[blue]ℹ[/blue] Using {expand.value.upper()} query expansion")

    try:
        method = container.get_talk_use_case if use_case_type == "talk" else container.get_search_use_case
        use_case = method(storage_config, llm_rerank, expand)

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            p.add_task("Processing query...", total=None)
            return use_case.execute(query, top_k, candidates, use_reranking)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)

def _display_chunks(chunks: List[Any], query: str) -> None:
    if not chunks:
        console.print("[yellow]No relevant chunks found.[/yellow]")
        return

    console.print(f"[green]✓[/green] Found {len(chunks)} relevant chunks.\n")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Content Preview")
    table.add_column("Metadata", style="dim")

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.content[:MAX_CHUNK_PREVIEW_LENGTH] + ("..." if len(chunk.content) > MAX_CHUNK_PREVIEW_LENGTH else "")
        metadata_str = ", ".join(f"{k}: {v}" for k, v in chunk.metadata.items())
        table.add_row(str(i), preview, metadata_str)

    console.print(table)