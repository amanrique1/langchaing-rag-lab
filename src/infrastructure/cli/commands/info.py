from src.infrastructure.cli.dependencies import (
    LanceArg, ChromaArg, FilesystemArg, StoragePathArg,
    CollectionArg, SingleColArg, VerboseArg
)
from src.infrastructure.cli.utils import resolve_storage_config, console, setup_logging, log_command_params

# Import defaults purely for display purposes
from src.infrastructure.adapters.chunk_stores.lance_chunk_store import DEFAULT_PERSIST_DIRECTORY as LANCE_DEFAULT
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import DEFAULT_PERSIST_DIRECTORY as CHROMA_DEFAULT
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import DEFAULT_PERSIST_DIRECTORY as FS_DEFAULT

def info(
    verbose: VerboseArg = False,
    lance: LanceArg = False,
    chroma: ChromaArg = False,
    filesystem: FilesystemArg = False,
    storage_path: StoragePathArg = None,
    collection: CollectionArg = None,
    single_collection: SingleColArg = False,
):
    """Display information about the current storage configuration."""
    setup_logging(verbose)
    log_command_params("info", locals(), verbose)

    storage_config = resolve_storage_config(collection, storage_path, lance, chroma, filesystem, single_collection)

    if storage_config.persist_directory:
        persist_dir = f"{storage_config.persist_directory} (custom)"
    else:
        mapping = {"LANCE": LANCE_DEFAULT, "CHROMA": CHROMA_DEFAULT, "FILESYSTEM": FS_DEFAULT}
        persist_dir = f"{mapping.get(storage_config.storage_type.name)} (default)"

    console.print("\n[bold]Storage Configuration:[/bold]")
    console.print(f"  Backend: {storage_config.storage_type.name}")
    console.print(f"  Collection Mode: {'Dual' if storage_config.dual_collection else 'Single'}")
    console.print(f"  Persist Directory: {persist_dir}")
    console.print(f"  Collection Name: {storage_config.collection_name}")