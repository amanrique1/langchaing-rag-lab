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
from src.infrastructure.adapters.models.huggingface_embedding_model import HuggingFaceEmbeddingModel
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
from src.application.use_cases.IngestionUseCase import IngestionUseCase
from src.application.use_cases.chat_use_case import ChatUseCase


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
        self._ingestion_use_cases: Dict[StorageConfig, IngestionUseCase] = {}

        # Chat sessions (stateful) - Key: (user_id, session_id, config, use_llm_reranking, expansion_strategy)
        self._chat_sessions: Dict[Tuple[str, str, StorageConfig, bool, Optional[str]], ChatUseCase] = {}

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
            #self._embedding_model = GoogleGenAIEmbeddingModel()
            self._embedding_model = HuggingFaceEmbeddingModel()
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

    def get_ingestion_use_case(self, config: StorageConfig) -> IngestionUseCase:
        """
        Get or create an IngestionUseCase for a specific storage configuration.
        """
        if config not in self._ingestion_use_cases:
            self._ingestion_use_cases[config] = IngestionUseCase(
                document_loader=self.get_document_loader(),
                chunk_store=self.get_chunk_store(config),
                embedding_model=self.get_embedding_model()
            )
        return self._ingestion_use_cases[config]

    def get_search_use_case(
        self,
        config: StorageConfig,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None
    ) -> SearchUseCase:
        """
        Get or create a LangGraph-based search use case.

        SearchUseCase is stateless and returns chunks only (no generation).
        It includes security validation, query expansion, retrieval, and reranking.

        Args:
            config: Storage configuration
            use_llm_reranking: Whether to use LLM reranking
            expansion_strategy: Query expansion strategy ('hyde', 'stepback', or None)

        Returns:
            SearchUseCase: Search use case instance
        """
        cache_key = (config, use_llm_reranking, expansion_strategy)

        if cache_key not in self._search_use_cases:
            # Get dependencies
            chunk_store = self.get_chunk_store(config)
            reranker = (self.get_llm_reranker() if use_llm_reranking
                       else self.get_encoder_reranker())
            query_expander = self.get_query_expander(expansion_strategy) if expansion_strategy else None

            # Create SearchUseCase with LangGraph components
            self._search_use_cases[cache_key] = SearchUseCase(
                chunk_store=chunk_store,
                reranker=reranker,
                input_guard=self.get_input_guard(),
                query_expander=query_expander
            )

        return self._search_use_cases[cache_key]

    def get_talk_use_case(
        self,
        config: StorageConfig,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None
    ) -> TalkUseCase:
        """
        Get or create a LangGraph-based talk use case (stateless Q&A).

        TalkUseCase is stateless - each query is independent with no conversation memory.
        For multi-turn conversations, use get_chat_use_case() instead.

        Args:
            config: Storage configuration
            use_llm_reranking: Whether to use LLM reranking
            expansion_strategy: Query expansion strategy ('hyde', 'stepback', or None)

        Returns:
            TalkUseCase: Stateless Q&A use case instance
        """
        cache_key = (config, use_llm_reranking, expansion_strategy)

        if cache_key not in self._talk_use_cases:
            # Get dependencies
            chunk_store = self.get_chunk_store(config)
            reranker = (self.get_llm_reranker() if use_llm_reranking
                       else self.get_encoder_reranker())
            query_expander = self.get_query_expander(expansion_strategy) if expansion_strategy else None

            # Create TalkUseCase with LangGraph components
            self._talk_use_cases[cache_key] = TalkUseCase(
                language_model=self.get_language_model(),
                chunk_store=chunk_store,
                reranker=reranker,
                input_guard=self.get_input_guard(),
                query_expander=query_expander
            )

        return self._talk_use_cases[cache_key]

    def get_chat_use_case(
        self,
        config: StorageConfig,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        memory_k: int = 5
    ) -> ChatUseCase:
        """
        Get or create a LangGraph-based chat use case with session management.

        ChatUseCase is stateful (holds conversation memory via LangGraph state),
        so instances are cached per (user_id, session_id, config) tuple to
        maintain conversation context across multiple queries.

        Args:
            config: Storage configuration
            use_llm_reranking: Whether to use LLM reranking
            expansion_strategy: Query expansion strategy ('hyde', 'stepback', or None)
            user_id: User identifier (defaults to "default_user")
            session_id: Session identifier (defaults to "default_session")
            memory_k: Number of conversation exchanges to retain in memory window

        Returns:
            ChatUseCase: Chat use case instance with conversation history
        """
        # Normalize identifiers
        user_id = user_id or "default_user"
        session_id = session_id or "default_session"

        # Cache key includes configuration to support different setups per session
        session_key = (user_id, session_id, config, use_llm_reranking, expansion_strategy)

        # Return existing session if available
        if session_key in self._chat_sessions:
            return self._chat_sessions[session_key]

        # Create new chat session with LangGraph components
        chunk_store = self.get_chunk_store(config)
        reranker = (self.get_llm_reranker() if use_llm_reranking
                   else self.get_encoder_reranker())
        query_expander = self.get_query_expander(expansion_strategy) if expansion_strategy else None

        chat_use_case = ChatUseCase(
            language_model=self.get_language_model(),
            chunk_store=chunk_store,
            reranker=reranker,
            input_guard=self.get_input_guard(),
            query_expander=query_expander,
            user_id=user_id,
            session_id=session_id,
            memory_k=memory_k
        )

        # Cache the session
        self._chat_sessions[session_key] = chat_use_case

        return chat_use_case

    def clear_chat_session(
        self,
        user_id: str,
        session_id: str,
        config: Optional[StorageConfig] = None,
        use_llm_reranking: bool = False,
        expansion_strategy: Optional[str] = None
    ) -> bool:
        """
        Clear a specific chat session from the container.

        Args:
            user_id: User identifier
            session_id: Session identifier
            config: Optional storage config (if None, clears all matching user/session)
            use_llm_reranking: Reranking configuration
            expansion_strategy: Expansion strategy configuration

        Returns:
            bool: True if session(s) were found and cleared, False otherwise
        """
        if config is not None:
            # Clear specific configuration
            session_key = (user_id, session_id, config, use_llm_reranking, expansion_strategy)
            if session_key in self._chat_sessions:
                self._chat_sessions[session_key].clear_memory()
                del self._chat_sessions[session_key]
                return True
            return False
        else:
            # Clear all sessions matching user_id and session_id
            keys_to_remove = [
                key for key in self._chat_sessions.keys()
                if key[0] == user_id and key[1] == session_id
            ]
            for key in keys_to_remove:
                self._chat_sessions[key].clear_memory()
                del self._chat_sessions[key]
            return len(keys_to_remove) > 0

    def get_active_sessions(self) -> list[Dict[str, any]]:
        """
        Get list of active chat sessions with their statistics.

        Returns:
            List of dicts containing session information
        """
        sessions = []
        for (user_id, session_id, config, use_llm_reranking, expansion_strategy), chat in self._chat_sessions.items():
            session_info = {
                "user_id": user_id,
                "session_id": session_id,
                "config": {
                    "storage_type": config.storage_type.value if hasattr(config.storage_type, 'value') else str(config.storage_type),
                    "collection_name": config.collection_name,
                    "dual_collection": config.dual_collection
                },
                "use_llm_reranking": use_llm_reranking,
                "expansion_strategy": expansion_strategy,
                **chat.get_memory_stats()
            }
            sessions.append(session_info)
        return sessions

    def clear_all_chat_sessions(self) -> int:
        """
        Clear all active chat sessions.

        Returns:
            int: Number of sessions cleared
        """
        count = len(self._chat_sessions)
        for chat in self._chat_sessions.values():
            chat.clear_memory()
        self._chat_sessions.clear()
        return count