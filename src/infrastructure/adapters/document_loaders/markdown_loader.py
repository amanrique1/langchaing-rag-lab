import concurrent.futures
import threading
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
from pathlib import Path
from application.ports.document_loader import DocumentLoader
from langchain_core.documents import Document
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

class MarkdownDocumentLoader(DocumentLoader):
    """
    Standardizes all input files into Markdown format.

    - .md files are read as-is to preserve exact structure for headers.
    - .pdf, .docx, etc., are converted to Markdown via MarkItDown.
    """

    def __init__(self):
        self._markdown_converter_instance: Optional[MarkItDown] = None
        self._markdown_lock = threading.Lock()

    @property
    def markdown_converter(self) -> MarkItDown:
        if self._markdown_converter_instance is None:
            with self._markdown_lock:
                if self._markdown_converter_instance is None:
                    logger.info("Initializing MarkItDown...")
                    self._markdown_converter_instance = MarkItDown()
        return self._markdown_converter_instance

    def load(self, source: str, max_workers: Optional[int] = None) -> List[Document]:
        data_path = Path(source)
        if not data_path.exists():
            raise FileNotFoundError(f"Folder '{source}' does not exist")

        all_files = [p for p in data_path.rglob("*") if p.is_file() and not p.name.startswith(".")]

        documents = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_file, p): p for p in all_files}

            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if res: documents.extend(res)
                except Exception as e:
                    logger.error(f"Error loading {futures[future]}: {e}")

        return documents

    def _process_file(self, file_path: Path) -> List[Document]:
        try:
            file_str = str(file_path)
            metadata = {"source": file_str, "filename": file_path.name}

            # FAST PATH: It's already Markdown. Don't process, just read.
            # This ensures structure (# headers) is perfectly preserved.
            if file_path.suffix.lower() == ".md":
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                return [Document(page_content=content, metadata=metadata)]

            # CONVERSION PATH: Convert PDF/Docx to Markdown
            else:
                result = self.markdown_converter.convert(file_str)
                if result and result.markdown:
                    return [Document(page_content=result.markdown, metadata=metadata)]
                return []
        except Exception as e:
            raise Exception(f"Processing failed: {e}")