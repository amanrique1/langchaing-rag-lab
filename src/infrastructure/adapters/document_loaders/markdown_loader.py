from typing import List
from pathlib import Path
from application.ports.document_loader import DocumentLoader
from domain.models.document import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from markitdown import MarkItDown


class MarkdownDocumentLoader(DocumentLoader):
    def load(
        self, source: str
    ) -> List[Document]:
        documents = []
        data_path = Path(source)

        if not data_path.exists():
            raise FileNotFoundError(f"The folder '{source}' does not exist")

        # Instantiate the converter once
        markdown_converter = MarkItDown()

        # Recursively find all files in the source directory
        all_files = [p for p in data_path.rglob("*") if p.is_file()]

        if not all_files:
            raise FileNotFoundError(f"No files found in '{source}'")

        for file_path in sorted(all_files):
            file_name = file_path.name

            # 1. Process original markdown files using the existing loader
            if file_path.suffix.lower() == ".md":
                try:
                    loader = UnstructuredMarkdownLoader(
                        str(file_path),
                        mode="single",
                        strategy="fast",
                    )
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata.update(
                            {"source": str(file_path), "file_name": file_name}
                        )
                        documents.append(
                            Document(
                                content=doc.page_content,
                                metadata=doc.metadata,
                            )
                        )
                except Exception as e:
                    print(f"Skipping markdown file '{file_name}' due to an error: {e}")

            # 2. For all other formats, use MarkItDown to convert to text
            else:
                try:
                    # Convert the file directly to a markdown string
                    result = markdown_converter.convert(str(file_path))

                    if result and result.markdown:
                        # Directly create the Document from the converted content
                        metadata = {"source": str(file_path), "file_name": file_name}
                        documents.append(
                            Document(
                                content=result.markdown,
                                metadata=metadata,
                            )
                        )
                    else:
                        print(f"Skipping file '{file_name}': Conversion resulted in empty content.")
                except Exception as e:
                    # Gracefully handle files that cannot be converted
                    print(f"Skipping file '{file_name}' due to conversion error: {e}")

        return documents