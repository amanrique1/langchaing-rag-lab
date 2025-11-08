import os
import numpy as np
from typing import List, Any
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.domain.models.document import Document
from src.domain.models.chunk import Chunk
from src.domain.strategies.chunking_strategy import ChunkingStrategy
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from src.domain.models.enums import SemanticChunkingThresholdType


class SemanticChunkingStrategy(ChunkingStrategy):
    def __init__(
        self,
        embedding_model: Any = None,
        breakpoint_threshold_type: SemanticChunkingThresholdType = SemanticChunkingThresholdType.PERCENTILE,
        breakpoint_threshold_amount: float = 95.0,
        min_chunk_size: int = 1,
        max_chunk_size: int = None,
    ):
        if embedding_model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
        else:
            self.embedding_model = embedding_model

        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _calculate_threshold(self, similarities: List[float]) -> float:
        if self.breakpoint_threshold_type == SemanticChunkingThresholdType.PERCENTILE:
            return np.percentile(similarities, self.breakpoint_threshold_amount)
        elif (
            self.breakpoint_threshold_type
            == SemanticChunkingThresholdType.STANDARD_DEVIATION
        ):
            mean = np.mean(similarities)
            std = np.std(similarities)
            return mean - (self.breakpoint_threshold_amount * std)
        elif (
            self.breakpoint_threshold_type
            == SemanticChunkingThresholdType.INTERQUARTILE
        ):
            q1 = np.percentile(similarities, 25)
            q3 = np.percentile(similarities, 75)
            iqr = q3 - q1
            return q1 - (self.breakpoint_threshold_amount * iqr)
        elif self.breakpoint_threshold_type == SemanticChunkingThresholdType.ABSOLUTE:
            return self.breakpoint_threshold_amount
        else:
            raise ValueError(
                f"Unsupported threshold type: {self.breakpoint_threshold_type}"
            )

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc_index, doc in enumerate(documents):
            sentences = sent_tokenize(doc.content)
            if not sentences:
                continue

            embeddings = self.embedding_model.embed_documents(sentences)

            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][
                    0
                ]
                similarities.append(similarity)

            if not similarities:
                all_chunks.append(Chunk(content=doc.content, metadata=doc.metadata))
                continue

            breakpoint_threshold = self._calculate_threshold(similarities)

            start_index = 0
            for i, similarity in enumerate(similarities):
                chunk_size = i - start_index + 1

                should_break = similarity < breakpoint_threshold

                if self.max_chunk_size and chunk_size >= self.max_chunk_size:
                    should_break = True

                if should_break and chunk_size >= self.min_chunk_size:
                    chunk_content = " ".join(sentences[start_index : i + 1])
                    metadata = {
                        "source": doc.metadata.get("source", ""),
                        "doc_index": doc_index,
                        "chunk_index": len(all_chunks),
                        "start_sentence_index": start_index,
                        "end_sentence_index": i,
                    }
                    all_chunks.append(Chunk(content=chunk_content, metadata=metadata))
                    start_index = i + 1

            if start_index < len(sentences):
                chunk_content = " ".join(sentences[start_index:])
                metadata = {
                    "source": doc.metadata.get("source", ""),
                    "doc_index": doc_index,
                    "chunk_index": len(all_chunks),
                    "start_sentence_index": start_index,
                    "end_sentence_index": len(sentences) - 1,
                }
                all_chunks.append(Chunk(content=chunk_content, metadata=metadata))

        return all_chunks
