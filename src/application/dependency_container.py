from typing import Dict, Tuple, Optional
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.language_model import LanguageModel
from src.application.ports.document_loader import DocumentLoader
from src.application.ports.chunk_store import ChunkStore
from src.infrastructure.adapters.language_models.google_genai_embedding_model import (
    GoogleGenAIEmbeddingModel,
)
from src.infrastructure.adapters.language_models.google_genai_language_model import (
    GoogleGenAILanguageModel,
)
from src.infrastructure.adapters.rerankers.llm_reranker import LLMReranker
from src.infrastructure.adapters.rerankers.encoder_reranker import EncoderReranker
from src.infrastructure.adapters.document_loaders.markdown_loader import (
    MarkdownDocumentLoader,
)
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.domain.services.search_service import SearchService
from src.domain.services.answer_generation_service import AnswerGenerationService
from src.domain.services.storage_service import StorageService
from src.application.use_cases.search_use_case import SearchUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.chunking_use_case import ChunkingUseCase


class DependencyContainer:
    """
    Dependency Injection Container for managing component lifecycle.
    
    Manages 4 tiers of dependencies:
    - Tier 1: Singleton models (expensive, stateless)
    - Tier 2: Cached chunk stores (stateful, config-based)
    - Tier 3: Cached services (stateless, dependency-based)
    - Tier 4: Use case factories (request-scoped)
    """
    
    def __init__(self):
        # Tier 1: Singleton instances (expensive, stateless)
        self._embedding_model: Optional[EmbeddingModel] = None
        self._language_model: Optional[LanguageModel] = None
        self._llm_reranker: Optional[LLMReranker] = None
        self._encoder_reranker: Optional[EncoderReranker] = None
        self._document_loader: Optional[DocumentLoader] = None
        
        # Tier 2: Chunk stores cached by configuration
        # Key: (collection_name, dual_collection)
        self._chunk_stores: Dict[Tuple[str, bool], ChunkStore] = {}
        
        # Tier 3: Services cached by chunk store configuration
        # Key: (storage_type, identifier, dual_collection)
        self._search_services: Dict[Tuple[str, str, bool], SearchService] = {}
        self._answer_services: Dict[Tuple[str, str, bool], AnswerGenerationService] = {}
        self._storage_services: Dict[Tuple[str, str, bool], StorageService] = {}
        
        # Tier 4: Use cases cached by configuration
        # Key: (storage_type, identifier, dual_collection)
        self._search_use_cases: Dict[Tuple[str, str, bool], SearchUseCase] = {}
        self._talk_use_cases: Dict[Tuple[str, str, bool], TalkUseCase] = {}
        self._storage_use_cases: Dict[Tuple[str, str, bool], StorageUseCase] = {}
        self._chunking_use_case: Optional[ChunkingUseCase] = None
    
    # ========== Tier 1: Singleton Models ==========
    
    def get_embedding_model(self) -> EmbeddingModel:
        """Get singleton embedding model instance."""
        if self._embedding_model is None:
            self._embedding_model = GoogleGenAIEmbeddingModel()
        return self._embedding_model
    
    def get_language_model(self) -> LanguageModel:
        """Get singleton language model instance."""
        if self._language_model is None:
            self._language_model = GoogleGenAILanguageModel()
        return self._language_model
    
    def get_llm_reranker(self) -> LLMReranker:
        """Get singleton LLM reranker instance."""
        if self._llm_reranker is None:
            self._llm_reranker = LLMReranker(self.get_language_model())
        return self._llm_reranker
    
    def get_encoder_reranker(self) -> EncoderReranker:
        """Get singleton Encoder reranker instance."""
        if self._encoder_reranker is None:
            self._encoder_reranker = EncoderReranker()
        return self._encoder_reranker
    
    def get_document_loader(self) -> DocumentLoader:
        """Get singleton document loader instance."""
        if self._document_loader is None:
            self._document_loader = MarkdownDocumentLoader()
        return self._document_loader
    
    # ========== Tier 2: Cached Chunk Stores ==========
    
    def get_chunk_store(
        self,
        collection_name: str,
        dual_collection: bool = True
    ) -> ChunkStore:
        """
        Get cached chunk store instance by configuration.
        
        Args:
            collection_name: Name of the Chroma collection
            dual_collection: Whether to enable dual collection storage
            
        Returns:
            Cached or newly created ChromaChunkStore instance
        """
        cache_key = (collection_name, dual_collection)
        
        if cache_key not in self._chunk_stores:
            self._chunk_stores[cache_key] = ChromaChunkStore(
                collection_name=collection_name,
                embedding_model=self.get_embedding_model(),
                dual_collection=dual_collection
            )
        
        return self._chunk_stores[cache_key]
    
    # ========== Tier 3: Cached Services ==========
    
    def get_search_service(
        self,
        collection_name: str,
        dual_collection: bool = True
    ) -> SearchService:
        """
        Get cached search service instance.
        
        Args:
            collection_name: Name of the Chroma collection
            dual_collection: Whether to enable dual collection storage
            
        Returns:
            Cached or newly created SearchService instance
        """
        # Assuming 'chroma' as storage_type for existing methods
        cache_key = ('chroma', collection_name, dual_collection)
        
        if cache_key not in self._search_services:
            self._search_services[cache_key] = SearchService(
                chunk_store=self.get_chunk_store(collection_name, dual_collection),
                language_model=self.get_language_model()
            )
        
        return self._search_services[cache_key]
    
    def get_answer_service(
        self,
        collection_name: str,
        dual_collection: bool = True
    ) -> AnswerGenerationService:
        """
        Get cached answer generation service instance.
        
        Args:
            collection_name: Name of the Chroma collection
            dual_collection: Whether to enable dual collection storage
            
        Returns:
            Cached or newly created AnswerGenerationService instance
        """
        # Assuming 'chroma' as storage_type for existing methods
        cache_key = ('chroma', collection_name, dual_collection)
        
        if cache_key not in self._answer_services:
            self._answer_services[cache_key] = AnswerGenerationService(
                language_model=self.get_language_model(),
                chunk_store=self.get_chunk_store(collection_name, dual_collection)
            )
        
        return self._answer_services[cache_key]
    
    def get_storage_service(
        self,
        collection_name: str,
        dual_collection: bool = True
    ) -> StorageService:
        """
        Get cached storage service instance.
        
        Args:
            collection_name: Name of the Chroma collection
            dual_collection: Whether to enable dual collection storage
            
        Returns:
            Cached or newly created StorageService instance
        """
        # Assuming 'chroma' as storage_type for existing methods
        cache_key = ('chroma', collection_name, dual_collection)
        
        if cache_key not in self._storage_services:
            self._storage_services[cache_key] = StorageService(
                chunk_store=self.get_chunk_store(collection_name, dual_collection)
            )
        
        return self._storage_services[cache_key]
    
    # ========== Tier 4: Cached Use Cases ==========
    
    def get_search_use_case(
        self,
        collection_name: Optional[str] = None,
        local_dir: Optional[str] = None,
        dual_collection: bool = True,
        use_llm_reranking: bool = False
    ) -> SearchUseCase:
        """
        Get cached SearchUseCase instance.
        
        Args:
            collection_name: Name of the Chroma collection (for ChromaDB)
            local_dir: Directory path (for FileSystem storage)
            dual_collection: Whether to enable dual collection storage
            use_llm_reranking: Whether to use LLM-based reranking
            
        Returns:
            Cached or newly created SearchUseCase instance
        """
        # Determine storage type and identifier
        if collection_name:
            storage_type, identifier = 'chroma', collection_name
        else:
            storage_type, identifier = 'filesystem', local_dir
        
        cache_key = (storage_type, identifier, dual_collection, use_llm_reranking)
        
        if cache_key not in self._search_use_cases:
            reranker = self.get_llm_reranker() if use_llm_reranking else self.get_encoder_reranker()
            
            self._search_use_cases[cache_key] = SearchUseCase(
                embedding_model=self.get_embedding_model(),
                collection_name=collection_name,
                local_dir=local_dir,
                dual_collection=dual_collection,
                reranker=reranker
            )
        
        return self._search_use_cases[cache_key]
    
    def get_talk_use_case(
        self,
        collection_name: Optional[str] = None,
        local_dir: Optional[str] = None,
        dual_collection: bool = True,
        use_llm_reranking: bool = False
    ) -> TalkUseCase:
        """
        Get cached TalkUseCase instance.
        
        Args:
            collection_name: Name of the Chroma collection (for ChromaDB)
            local_dir: Directory path (for FileSystem storage)
            dual_collection: Whether to enable dual collection storage
            use_llm_reranking: Whether to use LLM-based reranking (default is Encoder-based)
            
        Returns:
            Cached or newly created TalkUseCase instance
        """
        # Determine storage type and identifier
        if collection_name:
            storage_type, identifier = 'chroma', collection_name
        else:
            storage_type, identifier = 'filesystem', local_dir
        
        cache_key = (storage_type, identifier, dual_collection, use_llm_reranking)
        
        if cache_key not in self._talk_use_cases:
            reranker = self.get_llm_reranker() if use_llm_reranking else self.get_encoder_reranker()
            
            self._talk_use_cases[cache_key] = TalkUseCase(
                embedding_model=self.get_embedding_model(),
                language_model=self.get_language_model(),
                collection_name=collection_name,
                local_dir=local_dir,
                dual_collection=dual_collection,
                reranker=reranker
            )
        
        return self._talk_use_cases[cache_key]
    
    def get_storage_use_case(
        self,
        collection_name: Optional[str] = None,
        local_dir: Optional[str] = None,
        dual_collection: bool = True
    ) -> StorageUseCase:
        """
        Get cached StorageUseCase instance.
        
        Args:
            collection_name: Name of the Chroma collection (for ChromaDB)
            local_dir: Directory path (for FileSystem storage)
            dual_collection: Whether to enable dual collection storage
            
        Returns:
            Cached or newly created StorageUseCase instance
        """
        # Determine storage type and identifier
        if collection_name:
            storage_type, identifier = 'chroma', collection_name
        else:
            storage_type, identifier = 'filesystem', local_dir
        
        cache_key = (storage_type, identifier, dual_collection)
        
        if cache_key not in self._storage_use_cases:
            self._storage_use_cases[cache_key] = StorageUseCase(
                embedding_model=self.get_embedding_model(),
                collection_name=collection_name,
                local_dir=local_dir,
                dual_collection=dual_collection
            )
        
        return self._storage_use_cases[cache_key]
    
    def get_chunking_use_case(self) -> ChunkingUseCase:
        """
        Get cached ChunkingUseCase instance.
        
        Returns:
            Cached or newly created ChunkingUseCase instance
        """
        if self._chunking_use_case is None:
            self._chunking_use_case = ChunkingUseCase(
                document_loader=self.get_document_loader()
            )
        return self._chunking_use_case
