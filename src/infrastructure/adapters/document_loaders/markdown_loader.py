import concurrent.futures
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from pathlib import Path
from application.ports.document_loader import DocumentLoader
from domain.models.document import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from markitdown import MarkItDown


class MarkdownDocumentLoader(DocumentLoader):
    """
    A document loader that processes files in a directory in parallel.
    It uses lazy, thread-safe initialization for expensive resources like OCR.
    """

    def __init__(self):
        """Initializes the loader and the locks for thread-safe lazy loading."""
        # Placeholders for the lazily initialized objects
        self._markdown_converter_instance: Optional[MarkItDown] = None

        # Locks to ensure that initialization happens only once in a multi-threaded context
        self._markdown_lock = threading.Lock()

    @property
    def markdown_converter(self) -> MarkItDown:
        """
        Provides a lazily-initialized, thread-safe singleton instance of MarkItDown.
        """
        if self._markdown_converter_instance is None:
            with self._markdown_lock:
                if self._markdown_converter_instance is None:
                    print("Initializing MarkItDown for the first time...")
                    self._markdown_converter_instance = MarkItDown()
        return self._markdown_converter_instance

    def load(self, source: str, max_workers: Optional[int] = None) -> List[Document]:
        data_path = Path(source)

        if not data_path.exists():
            raise FileNotFoundError(f"The folder '{source}' does not exist")

        all_files = [p for p in data_path.rglob("*") if p.is_file()]

        if not all_files:
            raise FileNotFoundError(f"No files found in '{source}'")

        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_file, file_path) for file_path in all_files]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result_docs = future.result()
                    if result_docs:
                        documents.extend(result_docs)
                except Exception as exc:
                    print(f"[ERROR] {exc}")

        return documents

    def _process_file(self, file_path: Path) -> List[Document]:
        """
        Processes a single file. This function is executed by worker threads.

        Note: This function does not handle its own exceptions. It allows them
        to propagate up to the main 'load' method, which has the context
        to log them properly.
        """
        try:
            file_name = file_path.name
            file_suffix = file_path.suffix.lower()
            file_path_str = str(file_path)
            metadata = {"source": file_path_str, "file_name": file_name}

            # 1. Process markdown files
            if file_suffix == ".md":
                loader = UnstructuredMarkdownLoader(file_path_str, mode="single", strategy="fast")
                loaded_docs = loader.load()
                return [Document(content=doc.page_content, metadata={**doc.metadata, **metadata}) for doc in loaded_docs]

            # 2. Convert other formats
            else:
                conversion_result = self.markdown_converter.convert(file_path_str)
                if conversion_result and conversion_result.markdown:
                    return [Document(content=conversion_result.markdown, metadata=metadata)]
                else:
                    # Return an empty list for empty conversions.
                    return []
        except Exception as e:
            raise Exception(f"Failed to process file '{file_path}': {e}") from e