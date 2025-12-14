import os
from typing import List, Dict, Any, Optional, Set
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class MetadataManager:
    """
    Central authority for normalizing metadata and generating rich context strings.
    Handles English and Spanish stop words for keyword extraction.
    """

    _stop_words: Optional[Set[str]] = None

    @classmethod
    def _initialize_nltk(cls):
        """Lazy loader for NLTK resources."""
        if cls._stop_words is None:
            try:
                # Check if resources exist
                nltk.data.find('corpora/stopwords')
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                print("Downloading NLTK resources (stopwords, punkt)...")
                nltk.download('stopwords', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)

            # Combine English and Spanish stop words
            english_sw = set(stopwords.words('english'))
            spanish_sw = set(stopwords.words('spanish'))
            cls._stop_words = english_sw.union(spanish_sw)

    @classmethod
    def normalize_metadata(
        cls,
        doc_metadata: Dict[str, Any],
        chunk_content: str,
        chunk_index: int,
        total_chunks: int,
        hierarchy: List[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a standardized dictionary for the Content Collection.

        Args:
            doc_metadata: Metadata of the document
            chunk_content: Content of the chunk
            chunk_index: Index of the chunk
            total_chunks: Total number of chunks
            hierarchy: Hierarchy of the document
        
        Returns:
            Standardized dictionary for the Content Collection
        """
        cls._initialize_nltk()
        
        source = doc_metadata.get("source", "")
        filename = os.path.basename(source) if source else "unknown"
        clean_hierarchy = hierarchy if hierarchy else []
        
        # 1. Determine Immediate Context
        section_context = clean_hierarchy[-1] if clean_hierarchy else "General"

        # 2. Extract Keywords (Crucial for the Metadata Collection)
        keywords = cls._extract_top_keywords(chunk_content)

        return {
            # --- Identity ---
            "source": source,
            "filename": filename,
            "page": doc_metadata.get("page", 1),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            
            # --- Context / Hierarchy ---
            # 'breadcrumbs' string allows easy filtering in Chroma
            "breadcrumbs": " > ".join(clean_hierarchy), 
            "section_title": section_context,
            "root_doc_title": clean_hierarchy[0] if clean_hierarchy else filename,
            
            # --- Content Enrichment ---
            "extracted_keywords": ", ".join(keywords),
            "language_scope": "en_es"
        }

    @classmethod
    def create_searchable_string(cls, metadata: Dict[str, Any]) -> str:
        """
        Generates the text for the 'Metadata Collection' vector.
        Format: "File: X | Section: Y | Topics: Z"

        Args:
            metadata: Metadata dictionary
        
        Returns:
            Searchable string for the 'Metadata Collection' vector
        """
        components = []

        if metadata.get('filename'):
            components.append(f"File: {metadata['filename']}")

        if metadata.get('section_title'):
            components.append(f"Section: {metadata['section_title']}")
        
        # Breadcrumbs give the LLM hierarchical context
        if metadata.get('breadcrumbs'):
            components.append(f"Path: {metadata['breadcrumbs']}")

        # Keywords allow matching specific terms even if the header is generic
        if metadata.get('extracted_keywords'):
            components.append(f"Topics: {metadata['extracted_keywords']}")

        return " | ".join(components)

    @classmethod
    def _extract_top_keywords(cls, text: str, top_n: int = 8) -> List[str]:
        """
        Extracts significant terms, ignoring EN/ES stop words.

        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to extract
        
        Returns:
            List of top keywords
        """
        if not text:
            return []

        # Tokenize (handles punctuation better than regex)
        tokens = word_tokenize(text.lower())
        
        # Filter: Alphanumeric, Not Stopword, Length > 2
        filtered_words = [
            word for word in tokens 
            if word.isalnum() 
            and word not in cls._stop_words 
            and len(word) > 2
        ]

        count = Counter(filtered_words)
        return [word for word, _ in count.most_common(top_n)]