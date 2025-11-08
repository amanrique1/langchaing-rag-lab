
import pytest
from src.application.ports.document_loader import DocumentLoader
from src.domain.models.document import Document

class ConcreteDocumentLoader(DocumentLoader):
    def load(self, file_path: str) -> Document:
        return Document(metadata={"path": file_path}, content="test content")

def test_document_loader_load_raises_not_implemented_error():
    class AbstractDocumentLoader(DocumentLoader):
        pass

    with pytest.raises(TypeError):
        AbstractDocumentLoader()

    class IncompleteDocumentLoader(DocumentLoader):
        def some_other_method(self):
            pass

    with pytest.raises(TypeError):
        IncompleteDocumentLoader()

def test_document_loader_load():
    loader = ConcreteDocumentLoader()
    document = loader.load("test_path")
    assert isinstance(document, Document)
    assert document.metadata["path"] == "test_path"
    assert document.content == "test content"
