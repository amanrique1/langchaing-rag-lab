import pytest

import threading
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

from src.domain.models.document import Document
from src.infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)

# --- Fixtures ---

@pytest.fixture
def loader() -> MarkdownDocumentLoader:
    """Provides a fresh instance of the MarkdownDocumentLoader for each test."""
    return MarkdownDocumentLoader()

# --- Test Cases ---

def test_load_source_not_found(loader: MarkdownDocumentLoader):
    """Tests that FileNotFoundError is raised if the source path does not exist."""
    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="The folder 'invalid_path' does not exist"):
            loader.load("invalid_path")


def test_load_no_files_in_source(loader: MarkdownDocumentLoader):
    """Tests that FileNotFoundError is raised if the source path is empty."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.rglob", return_value=[]):
            with pytest.raises(FileNotFoundError, match="No files found in 'empty_dir'"):
                loader.load("empty_dir")


def test_process_markdown_file_successfully(loader: MarkdownDocumentLoader):
    """Tests the successful processing of a .md file."""
    mock_md_path = MagicMock(spec=Path)
    mock_md_path.is_file.return_value = True
    mock_md_path.suffix = ".md"
    mock_md_path.name = "test.md"
    mock_md_path.__str__.return_value = "/fake/path/test.md"

    mock_langchain_doc = MagicMock()
    mock_langchain_doc.page_content = "# Markdown Content"
    mock_langchain_doc.metadata = {"langchain_meta": "value"}

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.rglob", return_value=[mock_md_path]):
            with patch(
                "src.infrastructure.adapters.document_loaders.markdown_loader.UnstructuredMarkdownLoader"
            ) as mock_unstructured_loader:
                mock_unstructured_loader.return_value.load.return_value = [mock_langchain_doc]

                # FIXED: Patch the Document class to resolve the import conflict
                with patch("src.infrastructure.adapters.document_loaders.markdown_loader.Document", new=Document):
                    documents = loader.load("fake_path")

                    assert len(documents) == 1
                    doc = documents[0]
                    assert isinstance(doc, Document)
                    assert doc.content == "# Markdown Content"
                    assert doc.metadata["langchain_meta"] == "value"
                    assert doc.metadata["source"] == "/fake/path/test.md"


def test_process_other_file_successfully(loader: MarkdownDocumentLoader):
    """Tests the successful conversion of a non-markdown file."""
    mock_other_path = MagicMock(spec=Path)
    mock_other_path.is_file.return_value = True
    mock_other_path.suffix = ".pdf"
    mock_other_path.name = "report.pdf"
    mock_other_path.__str__.return_value = "/fake/path/report.pdf"

    mock_conversion_result = MagicMock()
    mock_conversion_result.markdown = "Converted content from PDF"

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.rglob", return_value=[mock_other_path]):
            with patch.object(
                MarkdownDocumentLoader, "markdown_converter", new_callable=PropertyMock
            ) as mock_converter_prop:
                mock_converter_instance = MagicMock()
                mock_converter_instance.convert.return_value = mock_conversion_result
                mock_converter_prop.return_value = mock_converter_instance

                # FIXED: Patch the Document class to resolve the import conflict
                with patch("src.infrastructure.adapters.document_loaders.markdown_loader.Document", new=Document):
                    documents = loader.load("fake_path")

                    assert len(documents) == 1
                    doc = documents[0]
                    assert isinstance(doc, Document)
                    assert doc.content == "Converted content from PDF"
                    assert doc.metadata["source"] == "/fake/path/report.pdf"
                    mock_converter_instance.convert.assert_called_once_with(str(mock_other_path))


def test_process_other_file_with_empty_conversion(loader: MarkdownDocumentLoader):
    """Tests when a conversion results in no markdown content."""
    mock_other_path = MagicMock(spec=Path, name="empty.doc")
    mock_other_path.is_file.return_value = True
    mock_other_path.suffix = ".doc"
    mock_other_path.__str__.return_value = "/fake/path/empty.doc"

    mock_conversion_result = MagicMock()
    mock_conversion_result.markdown = ""

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.rglob", return_value=[mock_other_path]):
            with patch.object(
                MarkdownDocumentLoader, "markdown_converter", new_callable=PropertyMock
            ) as mock_converter_prop:
                mock_converter_instance = MagicMock()
                mock_converter_instance.convert.return_value = mock_conversion_result
                mock_converter_prop.return_value = mock_converter_instance

                documents = loader.load("fake_path")
                assert len(documents) == 0


def test_load_handles_file_processing_exception(loader: MarkdownDocumentLoader, capsys):
    """Tests that `load` correctly handles exceptions from worker threads."""
    mock_path = MagicMock(spec=Path, name="bad.file")
    mock_path.is_file.return_value = True
    mock_path.suffix = ".bad"
    mock_path.__str__.return_value = "/fake/path/bad.file"

    error_message = f"Failed to process file '{mock_path}': Mock processing error"

    with patch("pathlib.Path.exists", return_value=True):
        with patch("pathlib.Path.rglob", return_value=[mock_path]):
            with patch.object(loader, "_process_file", side_effect=Exception(error_message)):
                documents = loader.load("fake_path", max_workers=1)

                assert documents == []
                captured = capsys.readouterr()
                assert f"[ERROR] {error_message}" in captured.out


def test_lazy_initialization_of_markdown_converter(loader: MarkdownDocumentLoader, capsys):
    """Tests the lazy initialization of the markdown_converter property."""
    with patch("src.infrastructure.adapters.document_loaders.markdown_loader.MarkItDown") as mock_markitdown_class:
        assert loader._markdown_converter_instance is None

        # First access
        converter_instance_1 = loader.markdown_converter
        captured = capsys.readouterr()
        assert "Initializing MarkItDown for the first time..." in captured.out
        mock_markitdown_class.assert_called_once()
        assert converter_instance_1 is not None

        # Second access
        converter_instance_2 = loader.markdown_converter
        captured = capsys.readouterr()
        assert "Initializing MarkItDown for the first time..." not in captured.out
        mock_markitdown_class.assert_called_once()
        assert converter_instance_1 is converter_instance_2


def test_thread_safe_initialization_of_markdown_converter():
    """Tests that markdown_converter initialization is thread-safe."""
    loader = MarkdownDocumentLoader()
    with patch("src.infrastructure.adapters.document_loaders.markdown_loader.MarkItDown") as mock_markitdown_class:
        num_threads = 5
        barrier = threading.Barrier(num_threads)

        def access_converter():
            barrier.wait()
            _ = loader.markdown_converter

        threads = [threading.Thread(target=access_converter) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mock_markitdown_class.assert_called_once()


def test_process_file_exception_propagation(loader: MarkdownDocumentLoader):
    """Tests that _process_file raises exceptions with proper error messages."""
    mock_path = MagicMock(spec=Path)
    mock_path.is_file.return_value = True
    mock_path.suffix = ".md"
    mock_path.name = "test.md"
    mock_path.__str__.return_value = "/fake/path/test.md"

    with patch(
        "src.infrastructure.adapters.document_loaders.markdown_loader.UnstructuredMarkdownLoader"
    ) as mock_loader:
        mock_loader.return_value.load.side_effect = RuntimeError("Original error")

        with pytest.raises(Exception, match="Failed to process file '/fake/path/test.md': Original error"):
            loader._process_file(mock_path)