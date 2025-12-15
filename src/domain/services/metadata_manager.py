import os
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords
import yake

class MetadataManager:
    """
    Central authority for normalizing metadata and generating rich context strings.
    Uses YAKE for statistical keyword extraction with bilingual stopword support.
    """

    _stop_words: Optional[List[str]] = None

    @classmethod
    def _initialize_nltk(cls):
        """
        Lazy loader for NLTK resources.
        Combines English and Spanish stopwords for YAKE.
        """
        if cls._stop_words is None:
            try:
                # Check if stopwords exist
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)

            # Combine English and Spanish stop words
            english_sw = set(stopwords.words('english'))
            spanish_sw = set(stopwords.words('spanish'))
            
            # Convert to list for YAKE compatibility
            cls._stop_words = list(english_sw.union(spanish_sw))

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
            hierarchy: Hierarchy of the document (breadcrumbs)
        
        Returns:
            Standardized dictionary for the Content Collection
        """
        cls._initialize_nltk()
        
        source = doc_metadata.get("source", "")
        # Extracts 'report.pdf' from '/data/uploads/report.pdf'
        filename = os.path.basename(source) if source else "unknown"
        
        clean_hierarchy = hierarchy if hierarchy else []
        
        # 1. Determine Immediate Context
        section_context = clean_hierarchy[-1] if clean_hierarchy else "General"

        keywords = cls._extract_top_keywords(chunk_content)

        return {
            # --- Identity ---
            "source": source,
            "filename": filename,
            "page": doc_metadata.get("page", None),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            
            # --- Context / Hierarchy ---
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
        Extracts significant terms using YAKE.
        Uses the combined EN/ES stopword list to filter noise in both languages.

        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to extract
        
        Returns:
            List of top keywords
        """
        if not text:
            return []

        # Ensure stopwords are loaded
        cls._initialize_nltk()

        # Initialize YAKE Extractor
        # lan="en": Default language (less relevant since we provide custom stopwords)
        # n=1: Max ngram size (1 = single words). Set to 2 or 3 for phrases.
        # dedupLim=0.9: Deduplication threshold to avoid similar words
        # stopwords: combined English + Spanish list
        kw_extractor = yake.KeywordExtractor(
            lan="en", 
            n=1, 
            dedupLim=0.9, 
            top=top_n, 
            features=None,
            stopwords=cls._stop_words
        )
        
        # Extract keywords
        # Returns list of tuples: [('keyword', score), ...] where lower score is better
        keywords_with_scores = kw_extractor.extract_keywords(text)

        # Return just the keyword strings
        return [kw for kw, score in keywords_with_scores]