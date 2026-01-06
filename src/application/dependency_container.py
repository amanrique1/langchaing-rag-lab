from typing import Dict, Tuple, Optional

# Ports & Domain
from src.application.ports.embedding_model import EmbeddingModel
from src.application.ports.language_model import LanguageModel
from src.application.ports.document_loader import DocumentLoader
from src.application.ports.chunk_store import ChunkStore
from src.application.ports.retriever import Retriever
from src.domain.models.enums import StorageType, QueryExpansionStrategy
from src.domain.models.config_classes import StorageConfig
from src.domain.guardrails.input_guard import InputGuard

# Adapters
from src.infrastructure.adapters.models.google_genai_embedding_model import GoogleGenAIEmbeddingModel
from src.infrastructure.adapters.models.google_genai_language_model import GoogleGenAILanguageModel
from src.infrastructure.adapters.document_loaders.markdown_loader import MarkdownDocumentLoader
from src.infrastructure.adapters.chunk_stores.lance_chunk_store import LanceChunkStore
from src.infrastructure.adapters.chunk_stores.chroma_chunk_store import ChromaChunkStore
from src.infrastructure.adapters.chunk_stores.file_system_chunk_store import FileSystemChunkStore
from src.infrastructure.adapters.models.llama_guard_model import LlamaGuard
from src.infrastructure.adapters.rerankers.llm_reranker import LLMReranker
from src.infrastructure.adapters.rerankers.encoder_reranker import EncoderReranker

# Domain Services
from src.domain.services.retrieval.simple_retriever import SimpleRetriever
from src.domain.services.retrieval.ensemble_retriever import EnsembleRetriever
from src.domain.services.retrieval.query_expander import QueryExpander

# Use Cases
from src.application.use_cases.search_use_case import SearchUseCase
from src.application.use_cases.talk_use_case import TalkUseCase
from src.application.use_cases.storage_use_case import StorageUseCase
from src.application.use_cases.chunking_use_case import ChunkingUseCase


class DependencyContainer:
    """
    Dependency Injection Container for managing component lifecycle.

    Storage Backends:
    - LanceDB: Default, high-performance hybrid search
    - ChromaDB: Alternative, use with --chroma flag
    - FileSystem: Local JSON storage, use with --filesystem flag

    All stores support:
    - collection_name: For organizing data
    - persist_directory: For custom storage locations
    """

    def __init__(self):
        # Tier 1: Singletons
        self._embedding_model: Optional[EmbeddingModel] = None
        self._input_guard: Optional[InputGuard] = None
        self._language_model: Optional[LanguageModel] = None
        self._llm_reranker: Optional[LLMReranker] = None
        self._encoder_reranker: Optional[EncoderReranker] = None
        self._document_loader: Optional[DocumentLoader] = None

        # Tier 1.5: Query Generators (Cached by type)
        self._query_expanders: Dict[str, QueryExpander] = {}

        # Tier 2: Cached Stores (Key: StorageConfig)
        self._chunk_stores: Dict[StorageConfig, ChunkStore] = {}

        # Tier 3: Cached Use Cases
        self._search_use_cases: Dict[Tuple[StorageConfig, bool, Optional[str]], SearchUseCase] = {}
        self._talk_use_cases: Dict[Tuple[StorageConfig, bool, Optional[str]], TalkUseCase] = {}
        self._storage_use_cases: Dict[StorageConfig, StorageUseCase] = {}
        self._chunking_use_case: Optional[ChunkingUseCase] = None

    # ========== Tier 0: Retriever Factory ==========

    def get_retriever(
        self,
        config: StorageConfig,
        expansion_strategy: Optional[QueryExpansionStrategy] = None
    ) -> Retriever:
        """
        Factory method - creates new retriever with optional query expansion.

        Args:
            config (StorageConfig): Storage configuration.
            expansion_strategy (Optional[QueryExpansionStrategy]): 'hyde', 'stepback', or None.

        Returns:
            Retriever: Configured retriever instance.
        """
        chunk_store = self.get_chunk_store(config)
        query_expander = self.get_query_expander(expansion_strategy) if expansion_strategy else None

        if config.dual_collection:
            return EnsembleRetriever(
                chunk_store=chunk_store,
                rrf_k=60,
                content_weight=1.0,
                metadata_weight=1.0,
                query_expander=query_expander
            )
        else:
            return SimpleRetriever(
                chunk_store=chunk_store,
                query_expander=query_expander
            )

    # ========== Tier 1: Singleton Models ==========

    def get_embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = GoogleGenAIEmbeddingModel()
        return self._embedding_model

    def get_input_guard(self) -> InputGuard:
        if self._input_guard is None:
            guard_model = LlamaGuard()
            self._input_guard = InputGuard(guard_model)
        return self._input_guard

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

    # ========== Tier 1.5: Query Generators (Cached by type) ==========

    def get_query_expander(self, strategy: QueryExpansionStrategy) -> Optional[QueryExpander]:
        """
        Get or create a query expander based on strategy.

        Args:
            strategy (QueryExpansionStrategy): 'hyde' or 'stepback'.

        Returns:
            Optional[QueryExpander]: The query expander instance.
        """
        if not strategy:
            return None

        if strategy not in self._query_expanders:
            self._query_expanders[strategy] = QueryExpander(self.get_language_model(), strategy)

        return self._query_expanders[strategy]

    # ========== Tier 2: Cached Chunk Stores ==========

    def get_chunk_store(self, config: StorageConfig) -> ChunkStore:
        """
        Get cached chunk store based on StorageConfig.

        All stores support:
        - collection_name: For organizing/naming the data
        - persist_directory: For custom storage locations (None = use store default)
        - dual_collection: For ensemble retrieval strategies

        Args:
            config (StorageConfig): The storage configuration.

        Returns:
            ChunkStore: The initialized chunk store.

        Raises:
            ValueError: If storage type is unknown or initialization fails.
        """
        if config not in self._chunk_stores:
            storage_type = config.storage_type.value if hasattr(config.storage_type, 'value') else str(config.storage_type)

            try:
                if config.storage_type == StorageType.FILESYSTEM:
                    store = FileSystemChunkStore(
                        collection_name=config.collection_name,
                        persist_directory=config.persist_directory,
                        embedding_model=self.get_embedding_model(),
                        dual_collection=config.dual_collection
                    )
                elif config.storage_type == StorageType.CHROMA:
                    store = ChromaChunkStore(
                        collection_name=config.collection_name,
                        persist_directory=config.persist_directory,
                        embedding_model=self.get_embedding_model(),
                        dual_collection=config.dual_collection
                    )
                elif config.storage_type == StorageType.LANCE:
                    store = LanceChunkStore(
                        collection_name=config.collection_name,
                        persist_directory=config.persist_directory,
                        embedding_model=self.get_embedding_model()
                    )
                else:
                    raise ValueError(f"Unknown storage type: {config.storage_type}")

                self._chunk_stores[config] = store

            except Exception as e:
                raise ValueError(
                    f"Failed to create chunk store for {storage_type}: {str(e)}"
                ) from e

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

    def get_search_use_case(
        self,
        config: StorageConfig,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None
    ) -> SearchUseCase:
        """
        Get or create a search use case.

        Args:
            config (StorageConfig): Storage configuration.
            use_llm_reranking (bool): Whether to use LLM reranking.
            expansion_strategy (Optional[str]): 'hyde', 'stepback', or None.

        Returns:
            SearchUseCase: Configured search use case.
        """
        cache_key = (config, use_llm_reranking, expansion_strategy)

        if cache_key not in self._search_use_cases:
            retriever = self.get_retriever(config, expansion_strategy)
            reranker = (self.get_llm_reranker() if use_llm_reranking 
                       else self.get_encoder_reranker())

            self._search_use_cases[cache_key] = SearchUseCase(
                retriever=retriever,
                reranker=reranker
            )

        return self._search_use_cases[cache_key]

    def get_talk_use_case(
        self,
        config: StorageConfig,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None
    ) -> TalkUseCase:
        """
        Get or create a talk use case.

        Args:
            config (StorageConfig): Storage configuration.
            use_llm_reranking (bool): Whether to use LLM reranking.
            expansion_strategy (Optional[str]): 'hyde', 'stepback', or None.

        Returns:
            TalkUseCase: Configured talk use case.
        """
        cache_key = (config, use_llm_reranking, expansion_strategy)

        if cache_key not in self._talk_use_cases:
            search_uc = self.get_search_use_case(config, use_llm_reranking, expansion_strategy)

            self._talk_use_cases[cache_key] = TalkUseCase(
                language_model=self.get_language_model(),
                search_use_case=search_uc,
                input_guard=self.get_input_guard()
            )

        return self._talk_use_cases[cache_key]