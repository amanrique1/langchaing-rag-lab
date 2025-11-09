from typing import List
from pathlib import Path
from application.ports.document_loader import DocumentLoader
from domain.models.document import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader


class MarkdownDocumentLoader(DocumentLoader):
    def load(
        self, source: str
    ) -> List[Document]:
        documents = []
        data_path = Path(source)

        if not data_path.exists():
            raise FileNotFoundError(f"The folder '{source}' does not exist")

        markdown_files = sorted(list(data_path.glob("*.md")))

        if not markdown_files:
            raise FileNotFoundError(f"No .md files found in '{source}'")

        for file_path in markdown_files:
            loader = UnstructuredMarkdownLoader(
                str(file_path),
                mode="single",
                strategy="fast",
                include_page_breaks=False,
                chunking_strategy=None,
            )
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata.update(
                    {"source": str(file_path), "file_name": Path(file_path).name}
                )
                documents.append(
                    Document(
                        content=doc.page_content,
                        metadata=doc.metadata,
                    )
                )
        return documents
