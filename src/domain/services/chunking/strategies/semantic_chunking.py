import os
import re
import numpy as np
from typing import List, Any, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from src.domain.models.chunk import Chunk
from src.application.ports.chunking_strategy import ChunkingStrategy
from sklearn.metrics.pairwise import cosine_similarity
from src.domain.models.enums import SemanticChunkingThresholdType
from src.domain.services.metadata_manager import MetadataManager


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    A chunking strategy that segments text based on semantic topic shifts using vector embeddings.
    """

    def __init__(
        self,
        embedding_model: Any = None,
        threshold_mode: SemanticChunkingThresholdType = SemanticChunkingThresholdType.PERCENTILE,
        threshold_value: float = 95.0,
        min_sentences: int = 1,
        max_sentences: Optional[int] = None,
    ):
        """
        Initialize the Semantic Chunking Strategy.

        Args:
            embedding_model (Any, optional): The embedding model instance. If None, tries to
                load `GoogleGenerativeAIEmbeddings` using the `EMBEDDING_MODEL` env var.
            threshold_mode (SemanticChunkingThresholdType): Logic for calculating the split threshold.
                Defaults to `PERCENTILE`.
            threshold_value (float): The value for the threshold calculation.
                Defaults to 95.0 (meaning split at the 5% biggest changes in similarity).
            min_sentences (int): Minimum sentences per chunk. Defaults to 1.
            max_sentences (int, optional): Maximum sentences per chunk (forced split).
                Defaults to None (unlimited).
        """

        if embedding_model is None:
            model_name = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
            self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)
        else:
            self.embedding_model = embedding_model

        self.threshold_mode = threshold_mode
        self.threshold_value = threshold_value
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    def _preprocess_markdown(self, text: str) -> str:
        """
        Prepares raw Markdown text for accurate sentence splitting preventing the
        merging of the header with the first sentence of the following paragraph.
        This degrades embedding quality as the topic title gets mashed into the content.

        Args:
            text (str): The raw input text containing Markdown formatting.

        Returns:
            str: The text with headers normalized to look like sentences (ending in periods).
        """
        # Regex explanation:
        # ^(#+\s+)  -> Match start of line, 1-6 hashes, and whitespace (Group 1)
        # (.*?)     -> Match the title text non-greedily (Group 2)
        # $         -> End of line (implied by re.MULTILINE logic in replacement)
        def add_period(match):
            header_prefix = match.group(1)
            title = match.group(2).strip()
            # If title exists and doesn't already end in . ? or !, add a period.
            if title and not title[-1] in ".?!":
                return f"{header_prefix}{title}.\n"
            return match.group(0)

        return re.sub(r'^(#+\s+)(.*?)$', add_period, text, flags=re.MULTILINE)

    def _split_sentences(self, text: str) -> List[str]:
        """
        Splits text into a list of sentences using regular expressions.

        This lightweight implementation avoids the heavy dependency of NLTK or SpaCy
        while being robust enough for pre-processed Markdown.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: A list of strings, where each string is a sentence.
        """
        # Split logic: Look for [.?!] followed by whitespace (Lookbehind assertion)
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_threshold(self, similarities: List[float]) -> float:
        """
        Calculates the numerical similarity threshold based on the configured statistical mode.

        Args:
            similarities (List[float]): A list of cosine similarity scores (0.0 to 1.0)
                between adjacent sentences.

        Returns:
            float: The threshold value. Any similarity score *lower* than this value
            is considered a "semantic dip" (a topic change).

        Raises:
            ValueError: If an unsupported `threshold_mode` is provided.
        """
        if self.threshold_mode == SemanticChunkingThresholdType.PERCENTILE:
            return np.percentile(similarities, self.threshold_value)
        elif self.threshold_mode == SemanticChunkingThresholdType.STANDARD_DEVIATION:
            mean = np.mean(similarities)
            std = np.std(similarities)
            return mean - (self.threshold_value * std)
        elif self.threshold_mode == SemanticChunkingThresholdType.INTERQUARTILE:
            q1 = np.percentile(similarities, 25)
            q3 = np.percentile(similarities, 75)
            iqr = q3 - q1
            return q1 - (self.threshold_value * iqr)
        elif self.threshold_mode == SemanticChunkingThresholdType.ABSOLUTE:
            return self.threshold_value
        else:
            raise ValueError(f"Unsupported threshold mode: {self.threshold_mode}")

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        """
        Orchestrates the semantic chunking process for a list of documents.

        Process Flow:
        1. **Pre-process:** Normalize Markdown headers to ensure clean splitting.
        2. **Tokenize:** Split the document into individual sentences.
        3. **Embed:** Generate vector representations for each sentence.
        4. **Compare:** Calculate Cosine Similarity between adjacent sentences ($S_i$ vs $S_{i+1}$).
        5. **Threshold:** Determine the cutoff value for what constitutes a "topic change".
        6. **Segment:** Group sentences into chunks, breaking where similarity drops below the threshold.

        Args:
            documents (List[Document]): The source documents to process.

        Returns:
            List[Chunk]: A list of semantically coherent chunks enriched with metadata.
        """
        all_chunks = []
        for doc in documents:
            # 1. Pre-process Markdown (Fix headers so they act like sentences)
            clean_content = self._preprocess_markdown(doc.page_content)

            # 2. Tokenize into sentences
            sentences = self._split_sentences(clean_content)
            if not sentences:
                continue

            # 3. Generate Embeddings
            embeddings = self.embedding_model.embed_documents(sentences)

            # 4. Calculate Cosine Similarity
            similarities = []
            for i in range(len(embeddings) - 1):
                # cosine_similarity returns [[score]], we extract the float
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)

            # Edge Case: Document is too short (1 sentence), no neighbors to compare.
            if not similarities:
                self._add_single_chunk(all_chunks, doc)
                continue

            # 5. Determine Splitting Threshold
            split_threshold = self._calculate_threshold(similarities)

            # 6. Group Sentences into Chunks
            start_index = 0
            for i, similarity in enumerate(similarities):
                current_chunk_length = i - start_index + 1

                # A split happens if similarity drops below threshold (Topic Change)
                should_break = similarity < split_threshold

                # Or if we hit the hard limit (Max Sentences)
                if self.max_sentences and current_chunk_length >= self.max_sentences:
                    should_break = True

                # But we enforce a minimum size (don't break if chunk is too small)
                if should_break and current_chunk_length >= self.min_sentences:
                    chunk_text = " ".join(sentences[start_index : i + 1])
                    self._create_and_append_chunk(
                        all_chunks, chunk_text, doc
                    )
                    start_index = i + 1

            # 7. Handle the final remaining sentences after the loop
            if start_index < len(sentences):
                chunk_text = " ".join(sentences[start_index:])
                self._create_and_append_chunk(
                    all_chunks, chunk_text, doc
                )

        return all_chunks

    def _add_single_chunk(self, all_chunks: List[Chunk], doc: Document) -> None:
        """
        Handles the edge case where a document contains only a single sentence.
        It wraps the entire document content into a single Chunk.

        Args:
            all_chunks (List[Chunk]): The destination list.
            doc (Document): The original source document.
        """
        source = doc.metadata.get("source", "")
        filename = os.path.basename(source) if source else "unknown"

        metadata = {
            "source": source,
            "filename": filename,
            "page": doc.metadata.get("page", None),
            "chunk_index": 0,
        }
        all_chunks.append(Chunk(content=doc.page_content, metadata=metadata))

    def _create_and_append_chunk(
        self,
        all_chunks: List[Chunk],
        content: str,
        doc: Document
    ) -> None:
        """
        Creates a Chunk object with enriched RAG metadata and appends it to the list.

        This method extracts **Section Context** from Markdown headers found within the chunk.
        This allows the LLM to understand where this chunk fits in the document hierarchy
        (e.g., "Pricing > Enterprise Plan") even if the chunk text itself is just "$50/mo".

        Args:
            all_chunks (List[Chunk]): The destination list.
            content (str): The text content of the chunk.
            doc (Document): The original source document.
        """

        # Parse headers from the chunk text itself
        headers = self._extract_headers(content)

        std_metadata = MetadataManager.normalize_metadata(
            doc_metadata=doc.metadata,
            chunk_content=content,
            chunk_index=len(all_chunks),
            total_chunks=0,
            hierarchy=headers
        )

        all_chunks.append(Chunk(content=content, metadata=std_metadata))

    def _extract_headers(self, content: str) -> List[str]:
        """
        Scans the chunk content for Markdown headers to build context.

        Args:
            content (str): The text to scan.

        Returns:
            List[str]: A list of header titles (e.g., ["Introduction", "Background"]).
        """
        headers = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('#'):
                header_text = line.lstrip('#').strip()
                if header_text:
                    headers.append(header_text)
        return headers