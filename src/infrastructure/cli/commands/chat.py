import typer
from typing import Optional
from typing_extensions import Annotated
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
import uuid

from src.infrastructure.cli.dependencies import (
    DEFAULT_TOP_K, DEFAULT_CANDIDATES, LanceArg, ChromaArg, FilesystemArg,
    StoragePathArg, CollectionArg, SingleColArg, VerboseArg,
    TopKArg, CandidatesArg, NoRerankArg, LlmRerankArg, ExpandArg
)
from src.infrastructure.cli.utils import (
    get_container, resolve_storage_config, console, setup_logging,
    log_command_params, logger, EXIT_CODE_ERROR
)


def chat(
    user_id: Annotated[
        Optional[str],
        typer.Option("--user-id", "-u", help="User ID for session context")
    ] = None,
    session_id: Annotated[
        Optional[str],
        typer.Option("--session-id", "-s", help="Session ID (auto-generated if not provided)")
    ] = None,
    memory_window: Annotated[
        int,
        typer.Option("--window", "-w", help="Number of conversation turns to keep in memory")
    ] = 5,
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
    """
    Start an interactive chat session with conversation memory.

    The chat maintains context using a sliding window memory (default 5 turns).

    Available Commands:
    - /exit, /quit: End the session
    - /clear: Clear conversation history
    - /history: Show recent conversation
    - /stats: Show session statistics
    - /help: Show available commands

    Example:
        $ cli chat --user-id john --memory-window 10
        $ cli chat --session-id abc123 --llm-reranking
    """
    setup_logging(verbose)
    log_command_params("chat", locals(), verbose)

    # Auto-generate session ID if not provided
    if session_id is None:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    storage_config = resolve_storage_config(
        collection, storage_path, lance, chroma, filesystem, single_collection
    )
    container = get_container()

    use_reranking = not no_rerank

    try:
        # Get or create chat session with memory
        chat_use_case = container.get_chat_use_case(
            config=storage_config,
            use_llm_reranking=llm_reranking,
            expansion_strategy=expand,
            user_id=user_id,
            session_id=session_id,
            memory_k=memory_window
        )

        # Display welcome panel
        _display_welcome_panel(chat_use_case, expand, llm_reranking)

        # Main chat loop
        while True:
            try:
                query = console.input("\n[bold green]You:[/bold green] ").strip()

                if not query:
                    continue

                # Handle commands
                if query.startswith("/"):
                    should_exit = _handle_command(query, chat_use_case, container)
                    if should_exit:
                        break
                    continue

                # Process regular query
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    progress.add_task("🤔 Thinking...", total=None)

                    answer = chat_use_case.execute(
                        query=query,
                        top_k=top_k,
                        num_candidates=candidates,
                        use_reranking=use_reranking
                    )

                console.print(f"\n[bold cyan]Assistant:[/bold cyan] {answer}")

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠ Use /exit or /quit to end the session.[/yellow]")
                continue

            except EOFError:
                console.print("\n[yellow]👋 Ending chat session.[/yellow]")
                break

    except Exception as e:
        logger.error(f"Chat session failed: {e}", exc_info=verbose)
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)


def _display_welcome_panel(chat_use_case, expand, llm_reranking):
    """Display welcome information panel."""
    stats = chat_use_case.get_memory_stats()

    config_info = []
    config_info.append(f"[cyan]User ID:[/cyan] {stats['user_id']}")
    config_info.append(f"[cyan]Session ID:[/cyan] {stats['session_id']}")
    config_info.append(f"[cyan]Memory Window:[/cyan] {stats['memory_window_k']} exchanges")

    if expand:
        config_info.append(f"[cyan]Query Expansion:[/cyan] {expand.value.upper()}")
    if llm_reranking:
        config_info.append(f"[cyan]Reranking:[/cyan] LLM-based")

    welcome_text = "\n".join([
        "[bold]Welcome to Chat Mode! 💬[/bold]",
        "",
        *config_info,
        "",
        "[dim]Commands: /exit, /quit, /clear, /history, /stats, /help[/dim]"
    ])

    console.print(Panel(welcome_text, title="Chat Session", border_style="cyan"))


def _handle_command(command: str, chat_use_case, container) -> bool:
    """
    Handle chat commands.

    Returns:
        bool: True if should exit, False otherwise
    """
    command = command.lower().strip()

    if command in ["/exit", "/quit"]:
        console.print("\n[yellow]👋 Ending chat session. Goodbye![/yellow]")
        return True

    elif command == "/clear":
        chat_use_case.clear_memory()
        console.print("[dim]✓ Conversation history cleared.[/dim]")
        return False

    elif command == "/history":
        _display_conversation_history(chat_use_case)
        return False

    elif command == "/stats":
        _display_session_stats(chat_use_case)
        return False

    elif command == "/help":
        _display_help()
        return False

    elif command == "/sessions":
        _display_active_sessions(container)
        return False

    else:
        console.print(f"[yellow]Unknown command: {command}[/yellow]")
        console.print("[dim]Type /help for available commands[/dim]")
        return False


def _display_conversation_history(chat_use_case) -> None:
    """Display conversation history using the new API."""
    history = chat_use_case.get_conversation_history()

    if not history:
        console.print("[dim]No conversation history yet.[/dim]")
        return

    table = Table(
        show_header=True,
        header_style="bold magenta",
        title="📜 Conversation History",
        title_style="bold cyan"
    )
    table.add_column("Turn", style="dim", width=6)
    table.add_column("Role", style="bold", width=12)
    table.add_column("Message")

    turn = 1
    for i, msg in enumerate(history):
        role = msg["role"]
        content = msg["content"]

        # Truncate long messages
        if len(content) > 200:
            content = content[:197] + "..."

        if role == "human":
            role_display = "You"
            style = "green"
        else:
            role_display = "Assistant"
            style = "cyan"

        # Show turn number for pairs
        turn_display = str(turn) if role == "human" else ""
        if role == "assistant":
            turn += 1

        table.add_row(turn_display, f"[{style}]{role_display}[/{style}]", content)

    console.print(table)


def _display_session_stats(chat_use_case) -> None:
    """Display session statistics."""
    stats = chat_use_case.get_memory_stats()

    stats_table = Table(
        show_header=False,
        box=None,
        title="📊 Session Statistics",
        title_style="bold cyan"
    )
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="bold")

    stats_table.add_row("User ID", stats["user_id"])
    stats_table.add_row("Session ID", stats["session_id"])
    stats_table.add_row("Total Messages", str(stats["total_messages"]))
    stats_table.add_row("Exchanges", str(stats["exchanges"]))
    stats_table.add_row("Memory Window", f"{stats['memory_window_k']} exchanges")

    console.print(stats_table)


def _display_active_sessions(container) -> None:
    """Display all active sessions in the container."""
    sessions = container.get_active_sessions()

    if not sessions:
        console.print("[dim]No active sessions.[/dim]")
        return

    sessions_table = Table(
        show_header=True,
        header_style="bold magenta",
        title="🔄 Active Sessions",
        title_style="bold cyan"
    )
    sessions_table.add_column("User ID", style="green")
    sessions_table.add_column("Session ID", style="cyan")
    sessions_table.add_column("Messages", justify="right")
    sessions_table.add_column("Exchanges", justify="right")

    for session in sessions:
        sessions_table.add_row(
            session["user_id"],
            session["session_id"],
            str(session["total_messages"]),
            str(session["exchanges"])
        )

    console.print(sessions_table)


def _display_help() -> None:
    """Display available commands."""
    help_text = """
[bold cyan]Available Commands:[/bold cyan]

[green]/exit, /quit[/green]     - End the chat session
[green]/clear[/green]           - Clear conversation history
[green]/history[/green]         - Show recent conversation
[green]/stats[/green]           - Show session statistics
[green]/sessions[/green]        - Show all active sessions
[green]/help[/green]            - Show this help message

[bold]Tips:[/bold]
• The assistant remembers recent conversation context
• Use specific questions for best results
• Reference previous messages naturally
    """
    console.print(Panel(help_text, title="Help", border_style="cyan"))