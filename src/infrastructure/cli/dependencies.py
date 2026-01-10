from typing import Optional
from typing_extensions import Annotated
import typer
from src.domain.models.enums import QueryExpansionStrategy

# Constants
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 20

# --- Storage Type Flags ---
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

# --- Storage Configuration ---
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

# --- Query Parameters ---
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