
import pytest
import tempfile
import os
from pathlib import Path
from src.infrastructure.adapters.document_loaders.markdown_loader import MarkdownDocumentLoader
from src.domain.models.enums import DocumentLoaderMode

@pytest.fixture
def temp_markdown_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test.md"
        with open(file_path, "w") as f:
            f.write("# Test\n\nThis is a test.")
        yield temp_dir

def test_markdown_document_loader(temp_markdown_file):
    loader = MarkdownDocumentLoader()
    documents = loader.load(source=temp_markdown_file, mode=DocumentLoaderMode.SINGLE)
    assert len(documents) == 1
    assert documents[0].metadata["file_name"] == "test.md"

def test_markdown_document_loader_file_not_found():
    loader = MarkdownDocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load(source="non_existent_dir")
