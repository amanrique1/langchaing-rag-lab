import typer
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.infrastructure.cli.dependencies import (
    DEFAULT_TOP_K, DEFAULT_CANDIDATES, LanceArg, ChromaArg, FilesystemArg,
    StoragePathArg, CollectionArg, SingleColArg, VerboseArg,
    TopKArg, CandidatesArg, NoRerankArg, LlmRerankArg, ExpandArg
)
from src.infrastructure.cli.utils import (
    get_container, resolve_storage_config, console, setup_logging,
    log_command_params, logger, EXIT_CODE_ERROR
)


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
    """Ask a single question and get an AI-generated answer."""
    setup_logging(verbose)
    log_command_params("talk", locals(), verbose)

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )

    console.print(f"\n[bold]Question:[/bold] {query}\n")

    if expand:
        console.print(f"[blue]ℹ[/blue] Using {expand.value.upper()} query expansion")

    container = get_container()
    use_reranking = not no_rerank

    try:
        talk_use_case = container.get_talk_use_case(storage_config, llm_reranking, expand)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Processing query...", total=None)
            answer = talk_use_case.execute(query, top_k, candidates, use_reranking)

        console.print("\n[bold yellow]Answer:[/bold yellow]")
        console.print(answer)

    except Exception as e:
        logger.error(f"Talk query failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)