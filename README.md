# LangChain RAG Lab

This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.

## Features

### Core Features
*   **Multiple Chunking Strategies**: Supports Length-Based, Structure-Based, Semantic, and Full Document Chunking.
*   **Hexagonal Architecture**: Clean separation of concerns for robust and testable code.
*   **Pluggable Storage Backends**: 
    *   **LanceDB** (Default): High-performance hybrid search with vector + BM25
    *   **ChromaDB**: Alternative vector store with dual collection support
    *   **FileSystem**: Local JSON storage for development/testing
    *   **All stores support**: `collection_name` and `persist_directory` for flexible configuration
*   **Modern CLI**: Easy-to-use subcommand-based interface (`save`, `talk`, `search`, `clean`, `info`).
*   **Google Gemini Integration**: Uses Google's embedding and language models.
*   **RAG Evaluation**: Built-in evaluation suite using the Ragas library to measure performance.

### 🚀 Enhanced RAG Architecture
*   **Query Expansion Strategies**: Transform queries for better retrieval using HyDE or Step-Back prompting.
*   **Metadata-Aware Search**: Dual collection support for content and metadata with separate embeddings.
*   **Ensemble Retrieval**: Combines multiple search strategies using **Reciprocal Rank Fusion (RRF)**.
*   **Dual Reranking Strategies**:
    *   **Encoder-Based** (Default): Local reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    *   **LLM-Based** (Optional): Intelligent reordering using Google Gemini (`--llm-rerank`).
*   **Security Guardrails**: Multi-layer protection using Regex Fast Rules and Semantic Guardrails (LlamaGuard).
*   **Strict Grounding**: System instructions enforced via prompt templates to prevent hallucinations and jailbreaks.
*   **Rich Metadata Extraction**: Automatically extracts headers, filenames, and section titles.

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
- [Architecture](#architecture)
- [Enhanced RAG Architecture Deep Dive](#enhanced-rag-architecture-deep-dive)
- [Query Expansion Strategies](#query-expansion-strategies)
- [Security & Grounding](#security--grounding)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [RAG Evaluation](#rag-evaluation)
- [Configuration Details](#configuration-details)
- [Examples](#examples)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

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

### Layers

*   **Domain Layer** (`src/domain`): Contains the business logic, core entities, and service implementations.
    *   **Guardrails**: `InputGuard` (Multi-layer security gateway using Regex and LlamaGuard).
    *   **Models**: `Document`, `Chunk`, `SearchResult` (Core entities with relevance scoring).
    *   **Services**:
        *   `MetadataManager`: Centralized metadata normalization and keyword extraction (YAKE).
    *   **Enums**: `StorageType`, `LengthBasedChunkingMode`, `SemanticChunkingThresholdType`, `QueryExpansionStrategy`.

*   **Application Layer** (`src/application`): Orchestrates business logic via Use Cases and defines Port interfaces.
    *   **Use Cases**:
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
        *   `main.py`: Command-line interface

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

#### Enhanced Talk Command Data Flow
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Query Expansion (Optional)        │
│   - HyDE: Generate hypothetical doc │
│   - StepBack: Broader question      │
└────────┬────────────────────────────┘
         │ (Original + Expanded Query)
         ▼
┌────────────────────────────────────────┐
│      Retrieval Strategy Selection      │
│  - Dual Collection: EnsembleRetriever  │
│  - Single Collection: SimpleRetriever  │
└────────┬───────────────────────────────┘
         │
         ├──────────────────┐
         ▼                  ▼                  
 ┌─────────────────┐  ┌──────────────────┐     
 │  Content Search │  │  Metadata Search │     
 │   (Semantic)    │  │   (Semantic)     │     
 └────────┬────────┘  └────────┬─────────┘     
          │                    │               
          └──────────┬─────────┘               
                     ▼                         
          ┌─────────────────────┐              
          │ Reciprocal Rank     │              
          │ Fusion (RRF)        │              
          │ Score Merging       │              
          └──────────┬──────────┘              
                     ▼                         
          ┌─────────────────────┐              
          │   Deduplication     │              
          │   (by chunk_id)     │              
          └──────────┬──────────┘              
                     ▼                         
          ┌─────────────────────┐              
          │  Top N Candidates   │              
          │      (e.g., 20)     │              
          └──────────┬──────────┘              
                     ▼                         
          ┌───────────────────────────────────┐
          │         Reranking Selection       │
          │  - Encoder-Based (Default, Local) │
          │  - LLM-Based (--llm-rerank)       │
          └──────────┬────────────────────────┘
                     ▼
          ┌─────────────────────┐
          │   Top K Results     │
          │      (e.g., 5)      │
          └──────────┬──────────┘
                     │
                     ▼
┌─────────────────────────────────────┐
│        Input Guard (Safety)         │
│  - Layer 1: Regex Fast Rules        │
│  - Layer 2: LlamaGuard Semantic     │
│  - Layer 3: Grounding via Template  │
└────────┬────────────────────────────┘
         │ (Safe Prompt)
         ▼
┌─────────────────────────────────────┐
│      Language Model (Gemini)        │
│  - Answer Generation                │
└──────────┬──────────────────────────┘
           ▼
┌─────────────────────┐
│  Generated Answer   │
└─────────────────────┘
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

**2. ChromaDB (--chroma flag)**
- **Dual collection strategy**: Separate content + metadata collections
- **Explicit opt-in**: Use when ChromaDB compatibility required

**3. FileSystem (--filesystem flag)**
- **Local JSON storage**: Simple file-based persistence
- **No database**: Zero external dependencies
- **Use case**: Development, testing, small datasets

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

**CLI Usage**:
```bash
poetry run cli talk "What is RAG?" --expand hyde
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

**CLI Usage**:
```bash
poetry run cli search "specific technical question" --expand stepback
```

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

**Key Design Principles**:
- **Single Parameter**: Retrievers accept one `query_expander` object (not multiple strategy flags)
- **Automatic Activation**: If a query expander is injected, it's used automatically
- **Type-Safe**: Uses `QueryExpansionStrategy` enum for validation
- **Cached**: Query expanders are singleton instances per strategy type

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
│   Template: assets/query_template │
└─────────┬─────────────────────────┘
          │ (Enforce Safe Prompt)
          ▼
┌───────────────────────────────────┐
│     Language Model Execution      │
└───────────────────────────────────┘
```

### Prompt Grounding Strategy

The `InputGuard` uses a strict system prompt (`assets/query_template.txt`) that forces the model to stay "grounded" in the provided context. Key rules include:
- **No Hallucinations**: Do not use outside knowledge.
- **Strict Evidence**: Only answer if the answer is explicitly in the context.
- **Safety**: Refuse to answer harmful or out-of-scope questions.

---

## Installation

### Prerequisites

*   **Python 3.11+**: Ensure you have Python installed.
*   **Poetry**: Package manager for Python.

### Steps

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd langchain-rag-lab
    ```

2.  **Install Poetry**: If you don't have Poetry, follow the instructions on the [official website](https://python-poetry.org/docs/#installation).
    ```bash
    curl -sSL https://install.python-poetry.org | python3 - 
    ```

3.  **Install Dependencies**:
    ```bash
    poetry install
    ```

4.  **Download NLTK Data** (required for semantic chunking):
    ```bash
    poetry run python -c "import nltk; nltk.download('punkt')"
    ```

---

## Environment Setup

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Google API Key (required for semantic chunking and talk functionality)
GOOGLE_API_KEY=your_google_api_key_here
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and add it to your `.env` file

**Note**: The semantic chunking strategy, query expansion, and the `talk` command require a valid Google API key. Other strategies work without it.

---

## Command-Line Interface (CLI)

The project uses a modern, **subcommand-based CLI** for different operations.

### Command Structure

```bash
poetry run cli [SUBCOMMAND] [ARGUMENTS] [OPTIONS]
```

### Subcommands

| Subcommand | Description |
| :--- | :--- |
| `save` | Chunks and saves documents from a source folder using a specified strategy. |
| `talk` | Asks a question, retrieves relevant documents, and generates a conversational answer. |
| `search` | Searches for document chunks most relevant to a query and displays them. |
| `clean` | Clears all data from a specified storage location. |
| `info` | Displays information about the current storage configuration. |

### Storage Options (All Commands)

All subcommands that interact with storage accept the following options to configure the backend:

| Option | Description | Default |
| :--- | :--- | :--- |
| `--collection <name>` | Collection/table name for organizing data | `default_collection` |
| `--storage-path <path>` | Custom storage directory (omit to use store default) | None |
| `--lance` | Use LanceDB storage (default if no storage flag specified) | True (implicit) |
| `--chroma` | Use ChromaDB storage instead of LanceDB | False |
| `--filesystem` | Use local filesystem JSON storage | False |
| `--single-collection` | Use single collection mode (disable ensemble retrieval) | False |

**Storage Type Priority** (when multiple flags specified):
1. `--filesystem` → FileSystem (highest priority)
2. `--chroma` → ChromaDB
3. `--lance` or no flags → LanceDB (default)

**Default Storage Locations** (when `--storage-path` not specified):
- **LanceDB**: `./lancedb`
- **ChromaDB**: `./chroma_db`
- **FileSystem**: `./filesystem_db`

**Examples**:
```bash
# Default LanceDB with default location
poetry run cli save ./docs.pdf length_based

# LanceDB with custom location
poetry run cli save ./docs.pdf length_based --lance --storage-path ./my_custom_db

# ChromaDB with custom collection
poetry run cli save ./docs.pdf length_based --chroma --collection my_docs

# FileSystem with custom path
poetry run cli save ./docs.pdf length_based --filesystem --storage-path ./my_chunks

# LanceDB with custom collection and location
poetry run cli save ./docs.pdf length_based --collection prod_docs --storage-path /data/vectordb
```

### Arguments and Options

#### `save` Subcommand
`poetry run cli save <source> <strategy> [OPTIONS]`
*   **`source`**: (Required) Path to the folder with markdown files.
*   **`strategy`**: (Required) Chunking strategy to use (`length_based`, `structure_based`, `semantic`, `full_doc`).
*   **`--config '...'`**: Optional JSON string with strategy-specific configuration.
*   **`--clean`**: Optional flag to clean the destination before saving new chunks.
*   **`--force`**: Skip confirmation prompts.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

#### `talk` Subcommand
`poetry run cli talk <query> [OPTIONS]`
*   **`query`**: (Required) The question to ask or the topic to discuss.
*   **`--top-k <number>`**: Optional number of final chunks to use for answer generation. Default is `5`.
*   **`--candidates <number>`**: Optional number of candidates to retrieve before reranking. Default is `20`.
*   **`--expand <strategy>`**: Optional query expansion strategy (`hyde` or `stepback`).
*   **`--no-rerank`**: Disable reranking completely.
*   **`--llm-rerank`**: Use LLM-based reranking instead of the default Encoder-based reranking.
*   **`--verbose`**: Enable verbose output for debugging.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

#### `search` Subcommand
`poetry run cli search <query> [OPTIONS]`
*   **`query`**: (Required) The search term or phrase.
*   **`--top-k <number>`**: Optional number of relevant chunks to retrieve. Default is `5`.
*   **`--candidates <number>`**: Optional number of candidates to retrieve before reranking. Default is `20`.
*   **`--expand <strategy>`**: Optional query expansion strategy (`hyde` or `stepback`).
*   **`--no-rerank`**: Disable reranking completely.
*   **`--llm-rerank`**: Use LLM-based reranking instead of the default Encoder-based reranking.
*   **`--verbose`**: Enable verbose output for debugging.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

#### `clean` Subcommand
`poetry run cli clean [OPTIONS]`
*   **`--force`**: Skip confirmation prompt.
*   **`--verbose`**: Enable verbose output.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

#### `info` Subcommand
`poetry run cli info [OPTIONS]`
*   **`--verbose`**: Enable verbose output.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

### Configuration Options

The enhanced pipeline is highly configurable:

| Option | Effect | Default |
|--------|--------|--------|
| `--expand hyde` | Use HyDE query expansion | None |
| `--expand stepback` | Use Step-Back query expansion | None |
| `--single-collection` | Use only content search (Disable Ensemble) | False |
| `--no-rerank` | Skip reranking step | False |
| `--llm-rerank` | Enable Gemini-based reranking | False |
| `--candidates N` | Candidates before reranking | 20 |
| `--top-k K` | Final results for answer | 5 |
| `--lance` | Use LanceDB instead of ChromaDB | True |
| `--chroma` | Use ChromaDB instead of LanceDB | False |
| `--filesystem` | Use FileSystem instead of LanceDB | False |
| `--storage-path <path>` | Custom storage directory | None (uses default) |

**Performance Profiles**:
- **Maximum Accuracy**: `--expand hyde --llm-rerank --candidates 30 --top-k 5` (Best results, highest latency & cost)
- **High Accuracy**: `--expand stepback --llm-rerank --candidates 20 --top-k 5` (Excellent results, moderate latency)
- **Balanced (Default)**: `--candidates 20 --top-k 5` (Uses local Encoder reranking + LanceDB hybrid search, fast & free)
- **Fast**: `--single-collection --no-rerank --top-k 5` (Lowest latency, no ensemble, no reranking)

---

## RAG Evaluation

The project includes a comprehensive evaluation suite using the **Ragas** library to measure the performance of the RAG pipeline.

### How it Works

The evaluation process involves the following steps:
1.  **Dataset Generation**: A ground truth dataset is generated from the source documents.
2.  **RAG System Querying**: The RAG system is queried with the test questions to get the generated answers and the retrieved contexts.
3.  **RAGAS Evaluation**: The generated dataset is evaluated using Ragas on a set of key metrics.
4.  **Results Analysis**: The evaluation results are analyzed to provide a summary of the RAG system's performance.

### How to Run

To run the evaluation suite, use the following command:

```bash
poetry run ragas
```

The evaluation script will:
- Chunk the documents in the `data` directory using the semantic strategy.
- Store the chunks in LanceDB collection named `ragas_evaluation_store`.
- Run the full evaluation, a category-based evaluation, and a quick test.
- Display a summary of the results in the console.

---

## Configuration Details

The `--config` option accepts a JSON string to customize the behavior of each strategy.

### `length_based`

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `chunk_size` | int | **Required**. Max size of each chunk | `1000` |
| `chunk_overlap` | int | **Required**. Overlap between chunks | `200` |
| `mode` | string | Splitting mode: `character` or `token` | `character` |

### `structure_based`

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `chunk_size` | int | Max size after splitting by headers | `1000` |
| `chunk_overlap` | int | Overlap between sub-chunks | `200` |
| `strip_headers` | bool | Remove header text from chunks | `false` |
| `max_header_levels` | int | Max header level to consider (1-6) | `3` |

### `semantic`

| Parameter | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| `threshold_mode` | string | Threshold algorithm | `percentile` |
| `threshold_value` | float | Value for the threshold type | `95.0` |
| `min_sentences` | int | Min sentences per chunk | `1` |
| `max_sentences` | int | Max sentences per chunk | `null` |

### `full_doc`

No configuration parameters are required. The entire document content is used as a single chunk.

---

## Comprehensive Testing Examples

This section provides complete examples to test all features with different storage backends.

### Testing with LanceDB (Default) ⭐

#### 1. Save with Different Strategies
```bash
# Full Doc Chunking (default LanceDB, default location)
poetry run cli save data full_doc --clean

# Length-based with custom collection and location
poetry run cli save data length_based \
  --collection 'length_docs' \
  --storage-path ./my_vectordb \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Structure-based with custom collection only
poetry run cli save data structure_based \
  --collection 'structure_docs' \
  --config '{"chunk_size": 1500, "chunk_overlap": 100}'

# Semantic with default settings
poetry run cli save data semantic \
  --config '{"threshold_mode": "percentile", "threshold_value": 90.0}'
```

#### 2. Search with LanceDB (Native Hybrid Search)
```bash
# Default: LanceDB with ensemble retrieval + encoder reranking
poetry run cli search "What are the Server Error Codes?" \
  --collection 'structure_docs' \
  --top-k 5

# With custom storage path
poetry run cli search "What are the Server Error Codes?" \
  --collection 'structure_docs' \
  --storage-path ./lancedb \
  --top-k 5

# With HyDE expansion
poetry run cli search "What are the Server Error Codes?" \
  --expand hyde \
  --top-k 5 \
  --candidates 20

# With Step-Back expansion
poetry run cli search "What are the Server Error Codes?" \
  --expand stepback \
  --top-k 3 \
  --candidates 10

# Fast mode (no expansion, no reranking)
poetry run cli search "What are the Server Error Codes?" \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 3. Talk with LanceDB
```bash
# Default: LanceDB + Ensemble + Encoder Reranking (RECOMMENDED)
poetry run cli talk "What are the Server Error Codes?" \
  --collection 'structure_docs' \
  --top-k 5

# With custom storage path
poetry run cli talk "What are the Server Error Codes?" \
  --collection 'length_docs' \
  --storage-path ./my_vectordb \
  --top-k 5

# Maximum accuracy (HyDE + LLM Reranking)
poetry run cli talk "What are the Server Error Codes?" \
  --expand hyde \
  --llm-rerank \
  --top-k 5 \
  --candidates 20

# High accuracy (Step-Back + Encoder Reranking)
poetry run cli talk "What are the Server Error Codes?" \
  --expand stepback \
  --top-k 5 \
  --candidates 20

# Fast mode
poetry run cli talk "What are the Server Error Codes?" \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 4. Manage LanceDB Collections
```bash
# Info about default collection
poetry run cli info

# Info about specific collection with custom path
poetry run cli info --collection 'length_docs' --storage-path ./my_vectordb

# Clean specific collection
poetry run cli clean --collection 'structure_docs'

# Clean with custom path
poetry run cli clean --collection 'length_docs' --storage-path ./my_vectordb --force
```

---

### Testing with ChromaDB (Explicit Opt-in)

#### 1. Save with ChromaDB
```bash
# Save to ChromaDB with default location
poetry run cli save data structure_based \
  --chroma \
  --collection 'chroma_docs' \
  --clean

# Save to ChromaDB with custom location
poetry run cli save data structure_based \
  --chroma \
  --collection 'chroma_docs' \
  --storage-path ./my_chroma_db \
  --clean
```

#### 2. Search with ChromaDB
```bash
# ChromaDB with dual collection strategy (default location)
poetry run cli search "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand hyde \
  --top-k 5

# ChromaDB with custom location
poetry run cli search "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --storage-path ./my_chroma_db \
  --top-k 5
```

#### 3. Talk with ChromaDB
```bash
# ChromaDB with full enhancement (default location)
poetry run cli talk "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand hyde \
  --llm-rerank \
  --top-k 5

# ChromaDB with custom location
poetry run cli talk "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --storage-path ./my_chroma_db \
  --top-k 5
```

#### 4. Manage ChromaDB Collections
```bash
# Info (default location)
poetry run cli info --chroma --collection 'chroma_docs'

# Info (custom location)
poetry run cli info --chroma --collection 'chroma_docs' --storage-path ./my_chroma_db

# Clean (default location)
poetry run cli clean --chroma --collection 'chroma_docs'

# Clean (custom location)
poetry run cli clean --chroma --collection 'chroma_docs' --storage-path ./my_chroma_db --force
```

---

### Testing with FileSystem Storage

#### 1. Save with Different Strategies
```bash
# Length-based with default location
poetry run cli save data length_based \
  --filesystem \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Structure-based with custom location and collection
poetry run cli save data structure_based \
  --filesystem \
  --collection 'fs_docs' \
  --storage-path ./my_filesystem_db \
  --config '{"chunk_size": 1500, "chunk_overlap": 150}'

# Semantic with default location
poetry run cli save data semantic \
  --filesystem \
  --config '{"threshold_mode": "percentile", "threshold_value": 95.0}'
```

#### 2. Search with FileSystem
```bash
# With HyDE expansion (default location)
poetry run cli search "What are the Server Error Codes?" \
  --filesystem \
  --expand hyde \
  --top-k 5

# Basic search with custom location
poetry run cli search "What are the Server Error Codes?" \
  --filesystem \
  --collection 'fs_docs' \
  --storage-path ./my_filesystem_db \
  --top-k 3
```

#### 3. Talk with FileSystem
```bash
# Fast mode with custom location
poetry run cli talk "What are the Server Error Codes?" \
  --filesystem \
  --collection 'fs_docs' \
  --storage-path ./my_filesystem_db \
  --top-k 3 \
  --single-collection \
  --no-rerank

# Full enhancement (default location)
poetry run cli talk "What are the Server Error Codes?" \
  --filesystem \
  --expand hyde \
  --top-k 5

```

#### 4. Manage FileSystem Storage
```bash
# Info (default location)
poetry run cli info --filesystem

# Info (custom location)
poetry run cli info --filesystem --collection 'fs_docs' --storage-path ./my_filesystem_db

# Clean (default location)
poetry run cli clean --filesystem

# Clean (custom location)
poetry run cli clean --filesystem --collection 'fs_docs' --storage-path ./my_filesystem_db --force
```

---

### Full Workflow Example (LanceDB with Custom Paths)

Complete workflow from scratch using LanceDB with custom storage locations:

```bash
# 1. Clean any existing data (optional)
poetry run cli clean --storage-path /data/production/vectordb --force

# 2. Save documents with semantic chunking to custom location
poetry run cli save data semantic \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --config '{"threshold_mode": "percentile", "threshold_value": 95.0}'

# 3. Search with default settings (fast & accurate)
poetry run cli search "What are the Server Error Codes?" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --top-k 5

# 4. Ask questions with expansion
poetry run cli talk "What are the Server Error Codes?" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --expand hyde \
  --top-k 5

# 5. Maximum accuracy configuration
poetry run cli talk "Explain API versioning strategies" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --expand hyde \
  --llm-rerank \
  --top-k 5 \
  --candidates 30

# 6. View storage info
poetry run cli info \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb

# 7. Clean up
poetry run cli clean \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --force
```

---

### Backend Comparison Example

Test the same query across all three backends with custom locations:

```bash
# Setup: Save to all backends with custom paths
poetry run cli save data semantic \
  --collection 'lance_test' \
  --storage-path ./test_dbs/lance \
  --clean

poetry run cli save data semantic \
  --chroma \
  --collection 'chroma_test' \
  --storage-path ./test_dbs/chroma \
  --clean

poetry run cli save data semantic \
  --filesystem \
  --collection 'fs_test' \
  --storage-path ./test_dbs/filesystem \
  --clean

# Compare: Same query, different backends
echo "=== LanceDB ==="
poetry run cli search "API error handling" \
  --collection 'lance_test' \
  --storage-path ./test_dbs/lance \
  --top-k 3

echo "=== ChromaDB ==="
poetry run cli search "API error handling" \
  --chroma \
  --collection 'chroma_test' \
  --storage-path ./test_dbs/chroma \
  --top-k 3

echo "=== FileSystem ==="
poetry run cli search "API error handling" \
  --filesystem \
  --collection 'fs_test' \
  --storage-path ./test_dbs/filesystem \
  --top-k 3

# Cleanup
poetry run cli clean --collection 'lance_test' --storage-path ./test_dbs/lance --force
poetry run cli clean --chroma --collection 'chroma_test' --storage-path ./test_dbs/chroma --force
poetry run cli clean --filesystem --collection 'fs_test' --storage-path ./test_dbs/filesystem --force
```

---

## Running Tests

This project includes comprehensive unit tests with high coverage.

### Run All Tests

```bash
poetry run pytest
```

### Run Tests with Coverage Report

```bash
poetry run pytest --cov=src --cov-report=term-missing
```

---

## Linting and Formatting

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Check for issues

```bash
poetry run ruff check .
```

### Fix issues automatically

```bash
poetry run ruff check . --fix
```

### Format code

```bash
poetry run ruff format .
```

---

## Troubleshooting

### Common Issues

#### 1. **"GOOGLE_API_KEY not set" Error**

**Solution**:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
```

#### 2. **Query Expansion Takes Too Long**

**Cause**: Query expansion requires an additional LLM call per query.

**Solutions**:
- Use expansion only for complex queries where it provides value
- Consider caching expanded queries for frequently asked questions
- Use faster reranking (default Encoder) instead of LLM reranking
- Reduce candidate count if using expansion + LLM reranking together

#### 3. **Query Expansion Not Improving Results**

**Analysis**:
- HyDE works best for questions with detailed answers in documents
- Step-Back works best for overly specific questions
- For simple keyword searches, expansion may not help

**Recommendation**: A/B test with and without expansion for your use case.

#### 4. **Which Storage Backend Should I Use?**

**Recommendation by Use Case**:
- **Production RAG System**: Use **LanceDB** (default) for best performance and native hybrid search
- **Existing ChromaDB Infrastructure**: Use `--chroma` flag for compatibility
- **Development/Testing**: Use `--filesystem` for simple file-based storage
- **Large Document Sets (10k+ chunks)**: Use **LanceDB** for 10-100x faster search
- **Small Datasets (<1000 chunks)**: Any backend will work fine
- **Multiple Projects**: Use `--collection` with different names for organization
- **Custom Locations**: Use `--storage-path` to specify exact directory

#### 5. **LanceDB vs ChromaDB Performance**

**Benchmarks** (approximate, dataset-dependent):
- **Small datasets (<1000 chunks)**: Similar performance (~100-200ms)
- **Medium datasets (1k-10k chunks)**: LanceDB 2-5x faster
- **Large datasets (>10k chunks)**: LanceDB 10-100x faster
- **Hybrid search**: LanceDB native support, ChromaDB requires dual collections

#### 6. **Managing Multiple Collections**

**Best Practices**:
```bash
# Organize by project
poetry run cli save data semantic --collection 'project_a_docs'
poetry run cli save data semantic --collection 'project_b_docs'

# Organize by data type
poetry run cli save data semantic --collection 'technical_docs'
poetry run cli save data semantic --collection 'business_docs'

# Use custom paths for isolation
poetry run cli save data semantic \
  --collection 'prod_docs' \
  --storage-path /data/production/vectordb

poetry run cli save data semantic \
  --collection 'dev_docs' \
  --storage-path /data/development/vectordb
```

#### 7. **Storage Path Confusion**

**Understanding the behavior**:
- **Without `--storage-path`**: Uses store's default directory (e.g., `./lancedb`, `./chroma_db`, `./filesystem_db`)
- **With `--storage-path`**: Uses your specified directory for that store type
- **Collection name**: Always used for organization within the storage directory

**Example**:
```bash
# These use different locations
poetry run cli save data semantic --collection 'docs'  # → ./lancedb/docs
poetry run cli save data semantic --collection 'docs' --storage-path ./custom  # → ./custom/docs
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone <your-fork-url>
cd langchain-rag-lab

# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run linting
poetry run ruff check .

# Format code
poetry run ruff format .
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.