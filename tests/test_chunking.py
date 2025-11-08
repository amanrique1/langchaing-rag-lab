import os
import unittest
from unittest.mock import patch
from src.application.use_cases.chunking_use_case import ChunkingUseCase
from src.infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.domain.models.chunk import Chunk


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.test_data_path = "data"
        self.collection_name = "test_collection"
        self.persist_directory = "./test_chroma_db"
        self.chunk_store = ChromaChunkStore(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
        self.chunk_store.clear()
        self.document_loader = MarkdownDocumentLoader()
        self.use_case = ChunkingUseCase(self.document_loader, self.chunk_store)

    def tearDown(self):
        if os.path.exists(self.persist_directory):
            import shutil

            shutil.rmtree(self.persist_directory)

    @patch(
        "src.infrastructure.adapters.chunk_stores.chroma_chunk_store.GoogleGenerativeAIEmbeddings"
    )
    def test_chunking_use_case(self, mock_embeddings):
        mock_embeddings.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3]
        ] * 100
        strategy_name = "length_based"
        strategy_config = {"mode": "character", "chunk_size": 100, "chunk_overlap": 20}

        chunks = self.use_case.execute(
            source=self.test_data_path,
            strategy_name=strategy_name,
            strategy_config=strategy_config,
            loader_mode="single",
        )

        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], Chunk)

        retrieved_chunk = self.chunk_store.get(
            f"{chunks[0].metadata.get('source', 'doc')}_{chunks[0].metadata.get('chunk_index', 0)}"
        )
        self.assertIsNotNone(retrieved_chunk)
        self.assertEqual(retrieved_chunk.content, chunks[0].content)


if __name__ == "__main__":
    unittest.main()
