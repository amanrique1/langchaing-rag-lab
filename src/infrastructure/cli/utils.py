import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from src.application.dependency_container import DependencyContainer
from src.domain.models.config_classes import StorageConfig
from src.domain.models.enums import (
    StorageType, 
    LengthBasedChunkingMode, 
    SemanticChunkingThresholdType
)

# --- Constants & Globals ---
EXIT_CODE_ERROR = 1
console = Console()
logger = logging.getLogger(__name__)

# --- UI & Logging Helpers ---

def setup_logging(verbose: bool = False) -> None:
    """Configure application logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

def log_command_params(command_name: str, params: Dict[str, Any], verbose: bool = False) -> None:
    """Log command parameters for debugging and verbose output."""
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

# --- Validation & Configuration Helpers ---

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
        return {}
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise typer.BadParameter("JSON must be an object/dictionary")
        return parsed
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON format: {e}")

def validate_source_path(source: str) -> Path:
    """Validate that the source path exists."""
    path = Path(source)
    if not path.exists():
        raise typer.BadParameter(f"Source path does not exist: {source}")
    return path

def resolve_storage_config(
    collection: Optional[str],
    storage_path: Optional[str],
    lance: bool,
    chroma: bool,
    filesystem: bool,
    single_collection: bool
) -> StorageConfig:
    """Resolve storage configuration from CLI arguments."""
    dual_collection = not single_collection
    
    if filesystem:
        storage_type = StorageType.FILESYSTEM
        backend_name = "Local Filesystem"
    elif chroma:
        storage_type = StorageType.CHROMA
        backend_name = "ChromaDB"
    else:
        storage_type = StorageType.LANCE
        backend_name = "LanceDB"

    coll_name = collection or "default_collection"

    try:
        config = StorageConfig(
            storage_type=storage_type,
            collection_name=coll_name,
            persist_directory=storage_path,
            dual_collection=dual_collection
        )
        location_info = f" at {storage_path}" if storage_path else ""
        console.print(f"[dim]Storage: {backend_name}{location_info}[/dim]")
        return config
    except ValueError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        raise typer.Exit(code=EXIT_CODE_ERROR)

def convert_strategy_enums(strategy: str, strategy_params: Dict[str, Any]) -> None:
    """Convert string parameters to enum types in place."""
    if strategy == "length_based" and "mode" in strategy_params:
        strategy_params["mode"] = LengthBasedChunkingMode(strategy_params["mode"])

    if strategy == "semantic" and "threshold_mode" in strategy_params:
        strategy_params["threshold_mode"] = SemanticChunkingThresholdType(
            strategy_params["threshold_mode"]
        )