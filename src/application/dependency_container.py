from typing import Dict, Tuple, Optional, Any

# Ports & Domain
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.language_model import LanguageModel
from src.application.ports.document_loader import DocumentLoader
from src.application.ports.chunk_store import ChunkStore
from src.domain.models.enums import StorageType
from src.domain.models.config_classes import StorageConfig

# Adapters
from src.infrastructure.adapters.language_models.google_genai_embedding_model import GoogleGenAIEmbeddingModel
from src.infrastructure.adapters.language_models.google_genai_language_model import GoogleGenAILanguageModel
from src.infrastructure.adapters.rerankers.llm_reranker import LLMReranker
from src.infrastructure.adapters.rerankers.encoder_reranker import EncoderReranker
from src.infrastructure.adapters.document_loaders.markdown_loader import MarkdownDocumentLoader
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore

# Use Cases
from src.application.use_cases.search_use_case import SearchUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.chunking_use_case import ChunkingUseCase

class DependencyContainer:
    """
    Dependency Injection Container for managing component lifecycle.
    """
    
    def __init__(self):
        # Tier 1: Singletons
        self._embedding_model: Optional[EmbeddingModel] = None
        self._language_model: Optional[LanguageModel] = None
        self._llm_reranker: Optional[LLMReranker] = None
        self._encoder_reranker: Optional[EncoderReranker] = None
        self._document_loader: Optional[DocumentLoader] = None
        
        # Tier 2: Cached Stores (Key: StorageConfig)
        self._chunk_stores: Dict[StorageConfig, ChunkStore] = {}
        
        # Tier 3: Cached Use Cases
        # Keys include StorageConfig + specific params (like use_llm_reranking)
        self._search_use_cases: Dict[Tuple[StorageConfig, bool], SearchUseCase] = {}
        self._talk_use_cases: Dict[Tuple[StorageConfig, bool], TalkUseCase] = {}
        self._storage_use_cases: Dict[StorageConfig, StorageUseCase] = {}
        self._chunking_use_case: Optional[ChunkingUseCase] = None
    
    # ========== Tier 1: Singleton Models ==========
    
    def get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = GoogleGenAIEmbeddingModel()
        return self._embedding_model
    
    def get_language_model(self) -> LanguageModel:
        if self._language_model is None:
            self._language_model = GoogleGenAILanguageModel()
        return self._language_model
    
    def get_llm_reranker(self) -> LLMReranker:
        if self._llm_reranker is None:
            self._llm_reranker = LLMReranker(self.get_language_model())
        return self._llm_reranker
    
    def get_encoder_reranker(self) -> EncoderReranker:
        if self._encoder_reranker is None:
            self._encoder_reranker = EncoderReranker()
        return self._encoder_reranker
    
    def get_document_loader(self) -> DocumentLoader:
        if self._document_loader is None:
            self._document_loader = MarkdownDocumentLoader()
        return self._document_loader
    
    # ========== Tier 2: Cached Chunk Stores ==========
    
    def get_chunk_store(self, config: StorageConfig) -> ChunkStore:
        """Get cached chunk store based on StorageConfig."""
        if config not in self._chunk_stores:
            if config.storage_type == StorageType.CHROMA:
                store = ChromaChunkStore(
                    collection_name=config.output_loc,
                    embedding_model=self.get_embedding_model(),
                    dual_collection=config.dual_collection
                )
            elif config.storage_type == StorageType.FILESYSTEM:
                store = FileSystemChunkStore(
                    local_dir=config.output_loc,
                    embedding_model=self.get_embedding_model(),
                    dual_collection=config.dual_collection
                )
            else:
                raise ValueError(f"Unknown storage type: {config.storage_type}")
            
            self._chunk_stores[config] = store
        
        return self._chunk_stores[config]
    
    # ========== Tier 3: Cached Use Cases ==========

    def get_chunking_use_case(self) -> ChunkingUseCase:
        if self._chunking_use_case is None:
            self._chunking_use_case = ChunkingUseCase(
                document_loader=self.get_document_loader()
            )
        return self._chunking_use_case

    def get_storage_use_case(self, config: StorageConfig) -> StorageUseCase:
        if config not in self._storage_use_cases:
            self._storage_use_cases[config] = StorageUseCase(
                chunk_store=self.get_chunk_store(config)
            )
        return self._storage_use_cases[config]

    def get_search_use_case(self, config: StorageConfig, use_llm_reranking: bool = False) -> SearchUseCase:
        cache_key = (config, use_llm_reranking)
        if cache_key not in self._search_use_cases:
            reranker = self.get_llm_reranker() if use_llm_reranking else self.get_encoder_reranker()
            self._search_use_cases[cache_key] = SearchUseCase(
                chunk_store=self.get_chunk_store(config),
                reranker=reranker
            )
        return self._search_use_cases[cache_key]
    
    def get_talk_use_case(self, config: StorageConfig, use_llm_reranking: bool = False) -> TalkUseCase:
        cache_key = (config, use_llm_reranking)
        if cache_key not in self._talk_use_cases:
            reranker = self.get_llm_reranker() if use_llm_reranking else self.get_encoder_reranker()
            self._talk_use_cases[cache_key] = TalkUseCase(
                language_model=self.get_language_model(),
                chunk_store=self.get_chunk_store(config),
                reranker=reranker
            )
        return self._talk_use_cases[cache_key]