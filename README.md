# LangChain RAG Lab

This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.

> **📖 Looking to run the system?** See [USAGE.md](USAGE.md) for installation, CLI reference, and practical examples.

## Features

### Core Features
*   **Multiple Chunking Strategies**: Supports Length-Based, Structure-Based, Semantic, and Full Document Chunking.
*   **Hexagonal Architecture**: Clean separation of concerns for robust and testable code.
*   **Pluggable Storage Backends**:
    *   **LanceDB** (Default): High-performance hybrid search with vector + BM25
    *   **ChromaDB**: Alternative vector store with dual collection support
    *   **FileSystem**: Local JSON storage for development/testing
    *   **All stores support**: `collection_name` and `persist_directory` for flexible configuration
*   **Modern CLI**: Easy-to-use subcommand-based interface (`save`, `talk`, `search`, `chat`, `clean`, `info`).
*   **Conversational Memory**: Modern LangChain message-based conversation history with sliding window memory.
*   **Session Management**: Multi-session support with persistent conversation context per user/session.
*   **Google Gemini Integration**: Uses Google's embedding and language models.
*   **RAG Evaluation**: Built-in evaluation suite using the Ragas library to measure performance.

### 🚀 Enhanced RAG Architecture
*   **Query Expansion Strategies**: Transform queries for better retrieval using HyDE, Step-Back, Subqueries, Zero-Shot, or Few-Shot prompting.
*   **Metadata-Aware Search**: Dual collection support for content and metadata with separate embeddings.
*   **Ensemble Retrieval**: Combines multiple search strategies using **Reciprocal Rank Fusion (RRF)**.
*   **Dual Reranking Strategies**:
    *   **Encoder-Based** (Default): Local reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    *   **LLM-Based** (Optional): Intelligent reordering using Google Gemini (`--llm-rerank`).
*   **Security Guardrails**: Multi-layer protection using Regex Fast Rules and Semantic Guardrails (LlamaGuard).
*   **Strict Grounding**: System instructions enforced via prompt templates to prevent hallucinations and jailbreaks.
*   **Rich Metadata Extraction**: Automatically extracts headers, filenames, and section titles.

### 💬 Conversational Features
*   **Message-Based History**: Uses LangChain's modern `ChatMessageHistory` with proper message types (HumanMessage, AIMessage).
*   **Sliding Window Memory**: Configurable conversation window (default: 5 exchanges) to maintain context without overwhelming the LLM.
*   **Session Persistence**: Conversations cached by `(user_id, session_id)` for seamless multi-session support.
*   **Interactive Commands**: Built-in `/history`, `/stats`, `/sessions`, `/help`, `/clear` commands for session management.
*   **Auto-Session IDs**: Automatic generation of session identifiers for quick testing.
*   **Memory Introspection**: Real-time statistics about conversation state (message count, exchanges, window size).

## Technologies Used

*   **Python 3.11+**: The primary programming language.
*   **Poetry**: For dependency management and project packaging.
*   **LangChain**: Framework for LLM orchestration.
*   **LanceDB**: Default vector database with hybrid search (vector + BM25).
*   **ChromaDB**: Alternative vector database for document chunks.
*   **Google Gemini**: Used for Embeddings and Language Modeling.
*   **LlamaGuard**: AI-powered semantic guardrail for input validation.
*   **Sentence Transformers**: Local Cross-Encoders for fast reranking.
*   **Ragas**: Framework for evaluating RAG pipelines.

## Table of Contents

- [Core Concepts](#core-concepts)
  - [Chunking Strategies](#chunking-strategies)
- [Architecture](#architecture)
  - [Three-Layer Architecture](#three-layer-architecture)
  - [Layers](#layers)
  - [Storage Backend Architecture](#storage-backend-architecture)
  - [Conversation Memory Architecture](#conversation-memory-architecture)
  - [Data Flow Overview](#data-flow-overview)
- [Enhanced RAG Architecture Deep Dive](#enhanced-rag-architecture-deep-dive)
  - [Storage Backend Strategy](#storage-backend-strategy)
  - [Ensemble Retrieval with RRF](#ensemble-retrieval-with-rrf)
  - [Retriever Selection](#retriever-selection)
  - [Reranking System](#reranking-system)
  - [Metadata Extraction](#metadata-extraction)
- [Query Expansion Strategies](#query-expansion-strategies)
  - [Available Strategies](#available-strategies)
  - [Query Expansion Architecture](#query-expansion-architecture)
  - [Performance Considerations](#performance-considerations)
- [Conversation Memory System](#conversation-memory-system)
  - [Memory Architecture](#memory-architecture)
  - [Session Management](#session-management)
  - [Memory Window Strategy](#memory-window-strategy)
- [Security & Grounding](#security--grounding)
  - [Multi-Layer Security Workflow](#multi-layer-security-workflow)
  - [Prompt Grounding Strategy](#prompt-grounding-strategy)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Core Concepts

This project is designed around three core chunking strategies, each suited for different types of documents and use cases. Understanding these strategies will help you choose the best approach for your needs.

### Chunking Strategies

1.  **Length-Based Chunking**:
    This is the most straightforward method. It splits the text into chunks of a specified size, with an optional overlap between them. It's fast and simple but doesn't consider the content's structure or meaning.
    *   **Mode Options**:
        *   `character`: Splits by character count (faster, simpler)
        *   `token`: Splits by token count (more accurate for LLM context windows)

2.  **Structure-Based Chunking**:
    This method leverages the document's structure, such as Markdown headers, to create more meaningful chunks. It's ideal for well-structured documents like technical manuals or articles, as it keeps related content together.

3.  **Semantic Chunking**:
    This advanced technique uses NLP models to split the text based on semantic similarity. It identifies topic shifts and creates chunks that are contextually coherent. It's best for unstructured or semi-structured documents where preserving meaning is crucial.

4.  **Full Document Chunking**:
    This strategy treats the entire document as a single chunk. It is useful for smaller documents or when using LLMs with very large context windows where splitting is unnecessary.

---

## Architecture

This project is built using a **Hexagonal Architecture** (also known as Ports and Adapters) with a **three-layer design** for clean separation of concerns.

### Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Entry Points (CLI, API, etc)           │
│  - Parse arguments/requests             │
│  - Call use cases                       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Use Cases (Application Layer)          │
│  - Create dependencies                  │
│  - Inject into services                 │
│  - Coordinate service calls             │
│  - Manage conversation sessions         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Services (Domain Layer)                │
│  - Business logic                       │
│  - Reusable components                  │
└─────────────────────────────────────────┘
```

**Benefits:**
- **Entry Points** are thin adapters that only parse input and call use cases
- **Use Cases** orchestrate dependencies and coordinate services (easy to add new entry points like API)
- **Services** contain reusable business logic that can be shared across use cases
- **Session Management** lives at the container level for proper lifecycle control

### Layers

*   **Domain Layer** (`src/domain`): Contains the business logic, core entities, and service implementations.
    *   **Guardrails**: `InputGuard` (Multi-layer security gateway using Regex and LlamaGuard).
    *   **Models**: `Document`, `Chunk`, `SearchResult` (Core entities with relevance scoring).
    *   **Services**:
        *   `MetadataManager`: Centralized metadata normalization and keyword extraction (YAKE).
    *   **Enums**: `StorageType`, `LengthBasedChunkingMode`, `SemanticChunkingThresholdType`, `QueryExpansionStrategy`.

*   **Application Layer** (`src/application`): Orchestrates business logic via Use Cases and defines Port interfaces.
    *   **Use Cases**:
        *   `ChatUseCase`: Orchestrates conversational workflow with modern LangChain message-based memory.
        *   `TalkUseCase`: Manages the end-to-end "Chat with Data" pipeline and orchestrates security grounding.
        *   `SearchUseCase`: Orchestrates complex retrieval and reranking for search queries.
        *   `ChunkingUseCase`: Manages document decomposition strategies.
        *   `StorageUseCase`: Coordinates persistence and retrieval operations.
    *   **Ports (Interfaces)**: `ChunkStore`, `GuardrailModel`, `LanguageModel`, `EmbeddingModel`, `Reranker`, `Retriever`, `QueryExpander`, `DocumentLoader`, `ChunkingStrategy`.

*   **Infrastructure Layer** (`src/infrastructure`): Concrete adapters for external services and technologies.
    *   **Retrieval Services**:
        *   `SimpleRetriever`: Basic vector search with optional query expansion
        *   `EnsembleRetriever`: Advanced RRF-based retrieval combining content + metadata search
    *   **Storage Backends**:
        *   `LanceChunkStore` (Default): High-performance hybrid search (vector + BM25)
        *   `ChromaChunkStore`: Dual collection vector store
        *   `FileSystemChunkStore`: Local JSON storage
        *   **All stores support**: `collection_name` and `persist_directory` parameters
    *   **AI Models**:
        *   `GoogleGenAILanguageModel` (Gemini)
        *   `GoogleGenAIEmbeddingModel`
        *   `LlamaGuard` (Safety Adapter)
    *   **Query Expansion**:
        *   `HyDEGenerator`: Hypothetical document generation
        *   `StepBackGenerator`: Broader question generation
    *   **Rerankers**:
        *   `EncoderReranker` (Local MS-Marco, Default)
        *   `LLMReranker` (Gemini-based, Optional)
    *   **Data Ingestion**:
        *   `MarkdownDocumentLoader`: Markdown reader and parser
    *   **CLI**:
        *   `main.py`: Command-line interface with session-aware chat mode
    *   **Dependency Management**:
        *   `DependencyContainer`: Session-aware dependency injection with conversation caching

### Storage Backend Architecture

The system resolves storage backends based on a strict priority system defined in the CLI arguments:

```
┌──────────────────────────────────────────────┐
│            Storage Backend Priority          │
├──────────────────────────────────────────────┤
│  1. --filesystem       → FileSystem          │
│  2. --chroma           → ChromaDB            │
│  3. --lance            → LanceDB ⭐          │
│  4. Default (No flags) → LanceDB ⭐          │
└──────────────────────────────────────────────┘
```

**All storage backends support**:
- `--collection <name>`: Collection/table name for organizing data
- `--storage-path <path>`: Custom storage directory (None = use store's default)

**Default Locations** (when `--storage-path` is not specified):
- **LanceDB**: `./lancedb`
- **ChromaDB**: `./chroma_db`
- **FileSystem**: `./filesystem_db`

**LanceDB Advantages** (Default):
- Native hybrid search (vector + BM25)
- 10-100x faster than ChromaDB for large datasets
- Built-in full-text search without dual collections

**When to use ChromaDB**:
- Explicit compatibility requirements
- Existing ChromaDB infrastructure
- Dual collection strategy preference

**When to use FileSystem**:
- Development/testing
- Small datasets (<1000 chunks)
- No database dependencies needed

### Conversation Memory Architecture

The chat system uses modern LangChain patterns for conversation management:

```
┌──────────────────────────────────────────────┐
│         DependencyContainer                  │
│  ┌────────────────────────────────────────┐  │
│  │  Chat Session Cache                    │  │
│  │  Key: (user_id, session_id)            │  │
│  │  Value: ChatUseCase instance           │  │
│  │  ┌──────────────────────────────────┐  │  │
│  │  │ ChatUseCase                      │  │  │
│  │  │  ├─ ConversationBufferWindowMemory│  │  │
│  │  │  │   └─ ChatMessageHistory        │  │  │
│  │  │  │       ├─ HumanMessage         │  │  │
│  │  │  │       ├─ AIMessage            │  │  │
│  │  │  │       └─ (k exchanges max)    │  │  │
│  │  │  ├─ SearchUseCase                │  │  │
│  │  │  ├─ LanguageModel                │  │  │
│  │  │  └─ InputGuard                   │  │  │
│  │  └──────────────────────────────────┘  │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘
```

**Key Components**:

1. **ChatMessageHistory**: LangChain's modern message storage
   - Stores messages as proper `HumanMessage` and `AIMessage` objects
   - Type-safe message handling
   - Supports message metadata and additional attributes

2. **ConversationBufferWindowMemory**: Sliding window memory manager
   - Keeps last `k` conversation exchanges (default: 5)
   - Automatically manages message list size
   - Provides dictionary-based memory variable access

3. **Session Caching**: Container-level session management
   - Sessions cached by `(user_id, session_id)` tuple
   - Resumable conversations across CLI invocations
   - Memory persists until explicitly cleared or container reset

4. **Memory Introspection**: Real-time conversation state
   - `get_conversation_history()`: Returns formatted message list
   - `get_memory_stats()`: Provides session statistics
   - `clear_memory()`: Clears conversation while preserving session

### Data Flow Overview

#### Save Command Data Flow
```
┌─────────────────┐
│  Raw Documents  │
│  (data/*.md)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document Loader │ (MarkdownDocumentLoader)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Document     │ (Domain Model)
│    Objects      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Chunking Use    │
│     Case        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │ (Strategy Pattern)
│    Chunking     │ - Length-Based
│    Strategy     │ - Structure-Based
│                 │ - Semantic
│                 │ - Full Document
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Chunk       │ (Domain Model)
│    Objects      │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│         Storage Backend (Adapter)                  │
│  - LanceDB (Default, --lance flag, Hybrid Search)  │
│  - ChromaDB (--chroma flag)                        │
│  - FileSystem (--filesystem flag)                  │
│                                                    │
│  All support:                                      │
│  • collection_name (data organization)             │
│  • persist_directory (custom location)             │
└────────────────────────────────────────────────────┘
```

#### Enhanced Chat Command Data Flow
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────┐
│  DependencyContainer                 │
│  Get/Create ChatUseCase for session  │
│  Key: (user_id, session_id)          │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         ChatUseCase                  │
│  ┌────────────────────────────────┐  │
│  │ Load Conversation History      │  │
│  │ from ChatMessageHistory        │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │ Query Expansion (Optional)     │  │
│  │ - HyDE/StepBack/Subqueries...  │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │   Retrieval Strategy           │  │
│  │   (SearchUseCase)              │  │
│  │   - EnsembleRetriever          │  │
│  │   - SimpleRetriever            │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │  Reranking (Optional)          │  │
│  │  - Encoder-Based (Default)     │  │
│  │  - LLM-Based (--llm-rerank)    │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │  Build Context-Aware Prompt    │  │
│  │  - Recent conversation history │  │
│  │  - Retrieved RAG context       │  │
│  │  - Current query               │  │
│  │  - System instructions         │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │  Input Guard (Safety)          │  │
│  │  - Regex Fast Rules            │  │
│  │  - LlamaGuard Semantic         │  │
│  │  - Grounding via Template      │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │  Language Model (Gemini)       │  │
│  │  Generate Answer               │  │
│  └────────┬───────────────────────┘  │
│           │                           │
│           ▼                           │
│  ┌────────────────────────────────┐  │
│  │  Save to Memory                │  │
│  │  HumanMessage(query)           │  │
│  │  AIMessage(response)           │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ Return Response │
│ Display to User │
└─────────────────┘
```

---

## Enhanced RAG Architecture Deep Dive

The enhanced RAG system introduces several advanced components to improve retrieval accuracy and relevance.

### Storage Backend Strategy

The system supports three storage backends with automatic selection:

**1. LanceDB (Default)** ⭐
- **Native hybrid search**: Vector + BM25 in single collection
- **Performance**: 10-100x faster for large datasets
- **Zero-config**: No dual collection management needed
- **Customizable**: Supports both `collection_name` and `persist_directory`

**2. ChromaDB (--chroma flag)**
- **Dual collection strategy**: Separate content + metadata collections
- **Explicit opt-in**: Use when ChromaDB compatibility required
- **Customizable**: Supports both `collection_name` and `persist_directory`

**3. FileSystem (--filesystem flag)**
- **Local JSON storage**: Simple file-based persistence
- **No database**: Zero external dependencies
- **Use case**: Development, testing, small datasets
- **Customizable**: Supports both `collection_name` and `persist_directory`

### Ensemble Retrieval with RRF

The `EnsembleRetriever` combines multiple search strategies using **Reciprocal Rank Fusion (RRF)**:

**Algorithm**:
```python
for each chunk_id in results:
    rrf_score = 0
    for each retriever:
        if chunk found by retriever:
            rrf_score += weight / (k + rank)
    final_scores[chunk_id] = rrf_score
```

Where:
- `k = 60` (RRF constant, balances score distribution)
- `weight` = configurable weight for each retriever (default: 1.0)
- `rank` = position in that retriever's results (1-indexed)

**Benefits**:
- Combines evidence from multiple sources
- Reduces impact of outliers from any single retriever
- No need to normalize scores across different retrieval methods

### Retriever Selection

The system automatically selects the appropriate retriever based on configuration:

| Configuration | Retriever | Behavior |
|--------------|-----------|----------|
| Default | `EnsembleRetriever` | Dual collection search (content + metadata) with RRF fusion |
| `--single-collection` | `SimpleRetriever` | Single vector search (content only) |
| Query Expansion | Either retriever | Expands query → retrieves for each → deduplicates results |

### Reranking System

The system supports a multi-stage reranking process to maximize relevance:

1.  **Encoder-Based Reranking (DEFAULT)**:
    *   Uses a local Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`).
    *   **Pros**: Ultra-fast (milliseconds), runs locally on CPU/MPS, zero cost.
    *   **Behavior**: Pairs the query with each candidate and predicts a relevance score.
2.  **LLM-Based Reranking (OPTIONAL)**:
    *   Uses Google Gemini to analytically re-order results.
    *   **Pros**: Higher semantic understanding for complex queries.
    *   **Enable via**: `--llm-rerank` CLI flag.

### Metadata Extraction

Chunking strategies automatically extract rich metadata:

| Strategy | Extracted Metadata |
|----------|-------------------|
| **Semantic** | filename, headers (from content), section_title, doc_type |
| **Structure-Based** | filename, headers (from hierarchy), section_title, doc_type |
| **Length-Based** | filename, doc_type |

**Metadata Filtering**: Lists are converted to comma-separated strings for database compatibility.

---

## Query Expansion Strategies

Query expansion transforms the original user query to improve retrieval quality by generating alternative formulations that may match relevant documents more effectively.

### Available Strategies

#### 1. **HyDE (Hypothetical Document Embeddings)**

**How it works**:
- Generates a hypothetical answer to the user's question
- Searches using both the original query and the hypothetical answer
- The hypothesis often contains vocabulary and concepts similar to actual relevant documents

**When to use**:
- Complex questions requiring detailed answers
- Technical queries where you want to match against document-style content
- When documents contain answers rather than questions

**Example**:
```
Original: "What is RAG?"
HyDE Generated: "Retrieval Augmented Generation (RAG) is a technique that combines
information retrieval with language model generation. It works by first retrieving
relevant documents from a knowledge base, then using those documents as context..."
```

#### 2. **Step-Back Prompting**

**How it works**:
- Transforms specific questions into broader, more general questions
- Retrieves high-level concepts first, then uses them to answer the specific query
- Helps when the specific question is too narrow to match relevant documents

**When to use**:
- Very specific questions that might miss broader relevant content
- Multi-hop reasoning scenarios
- When you need conceptual understanding before specific details

**Example**:
```
Original: "What happens to the pressure of an ideal gas if temperature doubles
and volume increases by a factor of 8?"
Step-Back: "What are the physics principles behind the ideal gas law?"
```

#### 3. **Subqueries Decomposition**

**How it works**:
- Breaks complex multi-part questions into focused sub-questions
- Retrieves documents for each sub-question independently
- Combines results for comprehensive answer coverage

**When to use**:
- Complex questions with multiple aspects
- Multi-hop reasoning requirements
- When comprehensive coverage is more important than speed

#### 4. **Zero-Shot Expansion**

**How it works**:
- Uses prompt engineering to reformulate queries without examples
- Quick query understanding and expansion
- Minimal latency overhead

**When to use**:
- Need fast query reformulation
- Simple to moderate complexity queries
- When example-based learning isn't necessary

#### 5. **Few-Shot Expansion**

**How it works**:
- Leverages example-based learning for query understanding
- Provides context through examples
- Improves domain-specific query handling

**When to use**:
- Domain-specific technical queries
- When you have example query patterns
- Need guided query understanding

### Query Expansion Architecture

```
┌─────────────────┐
│  Original Query │
└────────┬────────┘
         │
         ▼
┌────────────────────────────┐
│  QueryExpander (Optional)  │
│  - HyDEGenerator           │
│  - StepBackGenerator       │
│  - SubqueriesGenerator     │
│  - ZeroShotExpander        │
│  - FewShotExpander         │
└────────┬───────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Expanded Queries       │
│  [Original, Expanded]   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Retrieval for Each     │
│  Query Variation        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Aggregate & Deduplicate│
│  Results by chunk_id    │
└─────────────────────────┘
```

### Performance Considerations

| Strategy | Latency Impact | Cost | Best For |
|----------|---------------|------|----------|
| **No Expansion** | None | Free | Simple queries, well-matched vocabulary |
| **HyDE** | +1 LLM call | Low | Complex questions, technical content |
| **Step-Back** | +1 LLM call | Low | Specific questions, multi-hop reasoning |
| **Subqueries** | +1 LLM call | Low | Multi-part complex questions |
| **Zero-Shot** | +1 LLM call | Low | Fast reformulation needs |
| **Few-Shot** | +1 LLM call | Low | Domain-specific queries |

**Key Design Principles**:
- **Single Parameter**: Retrievers accept one `query_expander` object (not multiple strategy flags)
- **Automatic Activation**: If a query expander is injected, it's used automatically
- **Type-Safe**: Uses `QueryExpansionStrategy` enum for validation
- **Cached**: Query expanders are singleton instances per strategy type

---

## Conversation Memory System

The chat system implements modern LangChain patterns for robust conversation management.

### Memory Architecture

**Component Hierarchy**:

```
ChatUseCase
  ├─ ConversationBufferWindowMemory (LangChain)
  │    ├─ memory_key: "chat_history"
  │    ├─ k: 5 (configurable window size)
  │    ├─ return_messages: True
  │    └─ chat_memory: ChatMessageHistory
  │         ├─ messages: List[BaseMessage]
  │         │    ├─ HumanMessage(content="...")
  │         │    ├─ AIMessage(content="...")
  │         │    └─ (repeating pattern)
  │         └─ Methods:
  │              ├─ add_message()
  │              ├─ clear()
  │              └─ messages property
  ├─ SearchUseCase (RAG retrieval)
  ├─ LanguageModel (Answer generation)
  └─ InputGuard (Security validation)
```

**Key Features**:

1. **Message Types**: Proper `HumanMessage` and `AIMessage` objects
   - Type-safe message handling
   - Metadata support for future extensions
   - Compatible with LangChain ecosystem

2. **Sliding Window**: Automatic context management
   - Keeps last `k` exchanges (default: 5)
   - Prevents context overflow
   - Maintains recent relevant history

3. **Memory Variables**: Dictionary-based access
   ```python
   memory.load_memory_variables({})
   # Returns: {"chat_history": [HumanMessage(...), AIMessage(...), ...]}
   ```

4. **Context Building**: Structured prompt assembly
   - System instructions
   - Conversation history
   - Retrieved RAG context
   - Current query
   - Grounding instructions

### Session Management

**Container-Level Caching**:

```python
# DependencyContainer maintains session cache
_chat_sessions: Dict[Tuple[str, str], ChatUseCase] = {}

# Session key structure
session_key = (user_id, session_id)

# Benefits:
# - Conversations persist across CLI calls
# - Memory survives individual command executions
# - Multiple concurrent sessions supported
# - Explicit session cleanup available
```

**Session Lifecycle**:

1. **Creation**: First call to `get_chat_use_case()`
   - Creates new `ChatUseCase` instance
   - Initializes empty `ChatMessageHistory`
   - Caches in container by session key

2. **Resumption**: Subsequent calls with same identifiers
   - Returns existing `ChatUseCase` instance
   - Conversation history intact
   - Memory window automatically maintained

3. **Termination**: Explicit cleanup
   - `/clear` command: Clears history, keeps session
   - `/exit` command: Ends interactive mode, session cached
   - `clear_chat_session()`: Removes from container

**Session Identification**:

| Identifier | Default | Purpose |
|-----------|---------|---------|
| `user_id` | `"default_user"` | User identity/tracking |
| `session_id` | Auto-generated UUID | Conversation grouping |

---

## Security & Grounding

This project implements a multi-layer security approach to ensure that the LLM provides safe, grounded, and accurate answers.

### Multi-Layer Security Workflow

```
┌───────────────────┐
│    User Query     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────┐
│   Layer 1: Fast Rules (Regex)     │  <-- Blocks common jailbreaks
│   Check: GuardrailConfig.PATTERNS │
└─────────┬─────────────────────────┘
          │ (Pass)
          ▼
┌───────────────────────────────────┐
│   Layer 2: Semantic (LlamaGuard)  │  <-- AI-powered intent detection
│   Check: LlamaGuard.validate()    │
└─────────┬─────────────────────────┘
          │ (Pass)
          ▼
┌───────────────────────────────────┐
│   Layer 3: Strict Grounding       │  <-- Prompt instructions
│   Template: assets/templates/query_template │
└─────────┬─────────────────────────┘
          │ (Enforce Safe Prompt)
          ▼
┌───────────────────────────────────┐
│     Language Model Execution      │
└───────────────────────────────────┘
```

### Prompt Grounding Strategy

The `InputGuard` uses a strict system prompt (`assets/templates/query_template.txt`) that forces the model to stay "grounded" in the provided context. Key rules include:
- **No Hallucinations**: Do not use outside knowledge.
- **Strict Evidence**: Only answer if the answer is explicitly in the context.
- **Safety**: Refuse to answer harmful or out-of-scope questions.
- **Conversation Awareness**: Consider chat history while staying grounded in documentation.

---

**🚀 Ready to start?** Head over to [USAGE.md](USAGE.md) for installation instructions and examples!