# LangChain RAG Lab

This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.

## Features

### Core Features
*   **Multiple Chunking Strategies**: Supports Length-Based, Structure-Based, and Semantic Chunking.
*   **Hexagonal Architecture**: Clean separation of concerns for robust and testable code.
*   **Pluggable Chunk Stores**: Stores processed chunks in either the local file system or ChromaDB.
*   **Modern CLI**: Easy-to-use subcommand-based interface (`save`, `talk`, `search`, `clean`).
*   **Google Gemini Integration**: Uses Google's embedding and language models.
*   **RAG Evaluation**: Built-in evaluation suite using the Ragas library to measure performance.

### 🚀 Enhanced RAG Architecture (NEW)
*   **Metadata-Aware Search**: Dual ChromaDB collections for content and metadata with separate embeddings.
*   **Ensemble Retrieval**: Combines multiple search strategies using **Reciprocal Rank Fusion (RRF)**.
*   **Dual Reranking Strategies**:
    *   **Encoder-Based**: Local reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    *   **LLM-Based**: Intelligent reordering using Google Gemini (`--llm-reranking`).
*   **Security Guardrails**: Multi-layer protection using Regex Fast Rules and Semantic Guardrails (LlamaGuard).
*   **Strict Grounding**: System instructions enforced via prompt templates to prevent hallucinations and jailbreaks.
*   **Rich Metadata Extraction**: Automatically extracts headers, filenames, and section titles.

## Technologies Used

*   **Python 3.11+**: The primary programming language.
*   **Poetry**: For dependency management and project packaging.
*   **LangChain**: Framework for LLM orchestration.
*   **ChromaDB**: Vector database for document chunks.
*   **Google Gemini**: Used for Embeddings and Language Modeling.
*   **LlamaGuard**: AI-powered semantic guardrail for input validation.
*   **Sentence Transformers**: Local Cross-Encoders for fast reranking.
*   **Ragas**: Framework for evaluating RAG pipelines.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Enhanced RAG Architecture Deep Dive](#enhanced-rag-architecture-deep-dive)
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
        *   **Retrieval**: `EnsembleRetriever` (RRF algorithm), `SimpleRetriever`.
        *   **Strategies**: Implementations for `LengthBasedChunking`, `StructureBasedChunking`, `SemanticChunking`.
    *   **Enums**: `StorageType`, `LengthBasedChunkingMode`, `SemanticChunkingThresholdType`.

*   **Application Layer** (`src/application`): Orchestrates business logic via Use Cases and defines Port interfaces.
    *   **Use Cases**:
        *   `TalkUseCase`: Manages the end-to-end "Chat with Data" pipeline and orchestrates security grounding.
        *   `SearchUseCase`: Orchestrates complex retrieval and reranking for search queries.
        *   `ChunkingUseCase`: Manages document decomposition strategies.
        *   `StorageUseCase`: Coordinates persistence and retrieval operations.
    *   **Ports (Interfaces)**: `ChunkStore`, `GuardrailModel`, `LanguageModel`, `EmbeddingModel`, `Reranker`, `Retriever`, `DocumentLoader`, `ChunkingStrategy`.

*   **Infrastructure Layer** (`src/infrastructure`): Concrete adapters for external services and technologies.
    *   **Retrieval & Storage**: 
      * `ChromaChunkStore` (Vector DB)
      * `FileSystemChunkStore` (Local)
    *   **AI Models**: 
      * `GoogleGenAILanguageModel` (Gemini)
      * `GoogleGenAIEmbeddingModel`
      * `LlamaGuard` (Safety Adapter)
    *   **Rerankers**: 
      * `EncoderReranker` (Local MS-Marco)
      * `LLMReranker` (LLM-based)
    *   **Data Ingestion**: 
      * `MarkdownDocumentLoader`: markdown reader, converter, and parser to document objects
    *   **CLI**: 
      * `main.py`: Command-line interface for orchestrating the end-to-end "Chat with Data" pipeline

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
┌─────────────────┐
│  Chunk Store    │ (Adapter Pattern)
│  - FileSystem   │
│  - ChromaDB     │
└─────────────────┘
```

#### Enhanced Talk Command Data Flow
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      Search Use Case (Retrieval)    │
└────────┬────────────────────────────┘
         │
         ├──────────────────┬─────────────────┐
         ▼                  ▼                 │
┌─────────────────┐  ┌──────────────────┐     │
│  Content Store  │  │  Metadata Store  │     │
│   (Semantic)    │  │   (Semantic)     │     │
└────────┬────────┘  └────────┬─────────┘     │
          │                    │               │
          └──────────┬─────────┘               │
                     ▼                         │
          ┌─────────────────────┐              │
          │ Reciprocal Rank     │              │
          │ Fusion (RRF)        │              │
          │ Score Merging       │              │
          └──────────┬──────────┘              │
                     ▼                         │
          ┌─────────────────────┐              │
          │   Deduplication     │              │
          │   (by chunk_id)     │              │
          └──────────┬──────────┘              │
                     ▼                         │
          ┌─────────────────────┐              │
          │  Top N Candidates   │              │
          │      (e.g., 20)     │              │
          └──────────┬──────────┘              │
                     ▼                         │
          ┌───────────────────────────────────┐│
          │         Reranking Selection       ││
          │  - Encoder-Based (Local MS-Marco) ││
          │  - LLM-Based (Google Gemini)      ││
          └──────────┬────────────────────────┘┘
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

#### Single-Collection Data Flow (Fallback)
```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Talk Use Case │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Chunk Store   │ (ChromaChunkStore)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Relevant Chunks │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        Input Guard (Safety)         │
│  - Layer 1: Regex Fast Rules        │
│  - Layer 2: LlamaGuard Semantic     │
│  - Layer 3: Grounding via Template  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        Language Model (Gemini)      │
│  - Pure Answer Generation           │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Generated      │
│    Answer       │
└─────────────────┘
```

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

## Enhanced RAG Architecture Deep Dive

The enhanced RAG system introduces several advanced components to improve retrieval accuracy and relevance.

### Unified Dual-Collection Strategy

The **ChromaChunkStore** manages **two separate ChromaDB collections** internally for each document set:

1. **Content Collection** (`{collection_name}_content`): Stores full chunk content with embeddings
2. **Metadata Collection** (`{collection_name}_metadata`): Stores searchable metadata strings with embeddings

**Metadata String Format**:
```
filename: api_development_standards.md | section: RESTful API Design | headers: API Standards > RESTful API Design | type: markdown
```

This unified approach:
- Automatically saves to both collections when storing chunks
- Hides the complexity of managing dual stores
- Allows searching either collection via `mode` parameter
- Combines content and metadata signals for better retrieval

### Ensemble Retrieval with RRF

The `EnsembleRetrieverService` combines multiple search strategies using **Reciprocal Rank Fusion (RRF)**:

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

### Reranking System

The system supports a multi-stage reranking process to maximize relevance:

1.  **Encoder-Based Reranking (DEFAULT)**:
    *   Uses a local Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`).
    *   **Pros**: Ultra-fast (milliseconds), runs locally on CPU/MPS, zero cost.
    *   **Behavior**: Pairs the query with each candidate and predicts a relevance score.
2.  **LLM-Based Reranking (OPTIONAL)**:
    *   Uses Google Gemini to analytically re-order results.
    *   **Pros**: Higher semantic understanding for complex queries.
    *   **Enable via**: `--llm-reranking` CLI flag.

### Metadata Extraction

Chunking strategies automatically extract rich metadata:

| Strategy | Extracted Metadata |
|----------|-------------------|
| **Semantic** | filename, headers (from content), section_title, doc_type |
| **Structure-Based** | filename, headers (from hierarchy), section_title, doc_type |
| **Length-Based** | filename, doc_type |

**Metadata Filtering**: Lists are converted to comma-separated strings for ChromaDB compatibility.

### Configuration Options

The enhanced pipeline is highly configurable:

| Option | Effect | Default |
|--------|--------|--------|
| `--single-collection` | Use only content search (Disable Dual/Ensemble) | False |
| `--no-rerank` | Skip reranking step | False |
| `--llm-reranking` | Enable Gemini-based reranking | False |
| `--candidates N` | Candidates before reranking | 20 |
| `--top-k K` | Final results for answer | 5 |

**Performance Profiles**:
- **High Accuracy**: `--llm-reranking --candidates 20 --top-k 5` (Best results, higher latency)
- **Balanced (Default)**: `--candidates 20 --top-k 5` (Uses local Encoder reranking, fast & free)
- **Fast**: `--single-collection --no-rerank --top-k 5` (Lowest latency, no ensemble, no reranking)

---

## Technical Deep Dive

### Length-Based Chunking

*   **How it works**: It can split by character count or by token count (using a tokenizer). The `chunk_overlap` parameter allows for a certain number of characters or tokens to be repeated at the beginning of the next chunk to maintain some context.
*   **Pros**: Simple, fast, and predictable. No external dependencies.
*   **Cons**: Ignores sentence boundaries, logical structure, and semantic meaning. This can lead to chunks that are not coherent.

### Structure-Based Chunking

*   **How it works**: It uses LangChain's `MarkdownHeaderTextSplitter` to split documents based on headers (`#`, `##`, `###`, etc.). Each section under a header is treated as a potential chunk. If a section exceeds `chunk_size`, it's further subdivided using a length-based approach.
*   **Pros**: Preserves the logical structure of the document, creating chunks that are more coherent and contextually relevant. Respects document hierarchy.
*   **Cons**: Only effective for documents with a well-defined structure. It will perform poorly on unstructured text.

### Semantic Chunking

*   **How it works**: 
    1. Text is split into sentences using NLTK's `sent_tokenize`
    2. Each sentence is converted to an embedding vector using Google Gemini's `models/embedding-001`
    3. Cosine similarity is calculated between consecutive sentence embeddings
    4. A threshold is applied to identify "breakpoints" (significant drops in similarity)
    5. Text is chunked at these breakpoints, indicating topic shifts
*   **Pros**: Creates the most coherent and contextually relevant chunks, as it is based on the meaning of the text. Best for RAG systems.
*   **Cons**: Computationally expensive and slower than other methods. Requires API calls to Google's embedding service. The quality depends on the embedding model.

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

**Note**: The semantic chunking strategy and the `talk` command require a valid Google API key. Other strategies work without it.

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
| `clean` | Clears all data from a specified storage location (local directory or ChromaDB collection). |

### Arguments and Options

#### `save` Subcommand
`poetry run cli save <source> <strategy> [OPTIONS]`
*   **`source`**: (Required) Path to the folder with markdown files.
*   **`strategy`**: (Required) Chunking strategy to use (`length_based`, `structure_based`, `semantic`).
*   **`--config '...'`**: Optional JSON string with strategy-specific configuration.
*   **`--clean`**: Optional flag to clean the destination before saving new chunks.

#### `talk` Subcommand
`poetry run cli talk <query> [OPTIONS]`
*   **`query`**: (Required) The question to ask or the topic to discuss.
*   **`--top-k <number>`**: Optional number of final chunks to use for answer generation. Default is `5`.
*   **`--candidates <number>`**: Optional number of candidates to retrieve before reranking. Default is `20`.
*   **`--ensemble`**: Enable ensemble retrieval (default: True).
*   **`--no-ensemble`**: Disable ensemble retrieval, use only content search.
*   **`--no-rerank`**: Disable reranking completely (reranking is enabled by default using Encoder).
*   **`--llm-reranking`**: Use LLM-based reranking instead of the default Encoder-based reranking.

#### `search` Subcommand
`poetry run cli search <query> [OPTIONS]`
*   **`query`**: (Required) The search term or phrase.
*   **`--top-k <number>`**: Optional number of relevant chunks to retrieve. Default is `5`.
*   **`--candidates <number>`**: Optional number of candidates to retrieve before reranking. Default is `20`.
*   **`--ensemble`**: Enable ensemble retrieval (default: True).
*   **`--no-ensemble`**: Disable ensemble retrieval, use only content search.
*   **`--no-rerank`**: Disable reranking completely (reranking is enabled by default using Encoder).
*   **`--llm-reranking`**: Use LLM-based reranking instead of the default Encoder-based reranking.

### Universal Storage Options
All subcommands that interact with storage (`save`, `talk`, `search`, `clean`) accept one of the following mutually exclusive options to specify the destination:

| Option | Description | Default |
| :--- | :--- | :--- |
| `--local-dir <path>` | Use the local file system for storage. Specifies the output directory. | `output_chunks` |
| `--chroma-collection <name>` | Use ChromaDB for storage. Specifies the collection name. | `default_collection` |

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
- Store the chunks in a ChromaDB collection named `ragas_evaluation_store`.
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

This section provides complete examples to test all features with both storage backends.

### Testing with ChromaDB Storage

#### 1. Save with Different Strategies
```bash
# Full Doc Chunking
poetry run cli save data full_doc \
  --chroma-collection 'full_doc_docs' \
  --clean

# Length-based
poetry run cli save data length_based \
  --chroma-collection 'length_docs' \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Structure-based
poetry run cli save data structure_based \
  --chroma-collection 'structure_docs' \
  --config '{"chunk_size": 1500, "chunk_overlap": 100}'

# Semantic
poetry run cli save data semantic \
  --chroma-collection 'semantic_docs' \
  --config '{"threshold_mode": "percentile", "threshold_value": 90.0}'
```

#### 2. Search with ChromaDB
```bash
# Full enhancement
poetry run cli search "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 5 \
  --candidates 20

# Medium enhancement
poetry run cli search "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 3 \
  --candidates 10 \
  --no-rerank

# Basic search
poetry run cli search "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 3. Talk with ChromaDB
```bash
# Standard mode (Local Encoder reranking - FAST & FREE)
poetry run cli talk "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 5 \
  --candidates 20

# High accuracy mode (LLM-based Gemini reranking - BEST RESULTS)
poetry run cli talk "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 5 \
  --candidates 20 \
  --llm-reranking

# Balanced mode (no reranking)
poetry run cli talk "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 3 \
  --candidates 15 \
  --no-rerank

# Fast mode (Content search only, no Ensemble, no Rerank)
poetry run cli talk "What are the Server Error Codes?" \
  --chroma-collection 'structure_docs' \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 4. Clean ChromaDB Collection
```bash
poetry run cli clean --chroma-collection 'structure_docs'
```

---

### Comparing Storage Backends

Test the same query with both backends to compare results:

```bash
# Save to both backends
poetry run cli save data structure_based \
  --local-dir 'fs_test' \
  --config '{"chunk_size": 1000, "chunk_overlap": 200}' \
  --clean

poetry run cli save data structure_based \
  --chroma-collection 'chroma_test' \
  --config '{"chunk_size": 1000, "chunk_overlap": 200}' \
  --clean

# Search with FileSystem
poetry run cli search "What are the Server Error Codes?" \
  --local-dir 'fs_test' \
  --top-k 3

# Search with ChromaDB
poetry run cli search "What are the Server Error Codes?" \
  --chroma-collection 'chroma_test' \
  --top-k 3

# Talk with FileSystem
poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'fs_test' \
  --top-k 3

# Talk with ChromaDB
poetry run cli talk "What are the Server Error Codes?" \
  --chroma-collection 'chroma_test' \
  --top-k 3
```

---

### Performance Testing

Test different configurations to find optimal settings:

```bash
# Test different chunk sizes
for size in 500 1000 1500 2000; do
  poetry run cli save data length_based \
    --local-dir "chunks_${size}" \
    --config "{\"chunk_size\": ${size}, \"chunk_overlap\": 200, \"mode\": \"character\"}" \
    --clean
  
  poetry run cli search "What are the Server Error Codes?" \
    --local-dir "chunks_${size}" \
    --top-k 3
done

# Test different candidate counts
for candidates in 10 20 30 40; do
  echo "Testing with ${candidates} candidates..."
  poetry run cli talk "What are the Server Error Codes?" \
    --chroma-collection 'structure_docs' \
    --candidates ${candidates} \
    --top-k 5
done
```

---

### Testing with FileSystem Storage

#### 1. Save with Length-Based Chunking
```bash
# Character mode
poetry run cli save data length_based \
  --local-dir 'output_chunks/length_based/character' \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Token mode
poetry run cli save data length_based \
  --local-dir 'output_chunks/length_based/token' \
  --config '{"chunk_size": 500, "chunk_overlap": 100, "mode": "token"}'
```

#### 2. Save with Structure-Based Chunking
```bash
poetry run cli save data structure_based \
  --local-dir 'output_chunks/structure_based' \
  --config '{"chunk_size": 1500, "chunk_overlap": 150, "max_header_levels": 4}'
```

#### 3. Save with Semantic Chunking
```bash
# Percentile threshold
poetry run cli save data semantic \
  --local-dir 'output_chunks/semantic_based' \
  --config '{"threshold_mode": "percentile", "threshold_value": 95.0, "min_sentences": 2}'

# Standard deviation threshold
poetry run cli save data semantic \
  --local-dir 'output_chunks/semantic_based' \
  --config '{"threshold_mode": "std", "threshold_value": 1.5, "min_sentences": 1, "max_sentences": 8}'
```

#### 4. Search with FileSystem Storage
```bash
# Full enhancement (ensemble + reranking)
poetry run cli search "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 5 \
  --candidates 20

# Fast search (no reranking)
poetry run cli search "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 3 \
  --no-rerank

# Basic search (no ensemble, no reranking)
poetry run cli search "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 5. Talk with FileSystem Storage
```bash
# Full enhancement
poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 5 \
  --candidates 20

# Balanced mode (no reranking)
poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 3 \
  --candidates 15 \
  --no-rerank

# Fast mode
poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'output_chunks/length_based/character' \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 6. Clean FileSystem Storage
```bash
poetry run cli clean --local-dir 'output_chunks/length_based/character'
```

---

### Full Workflow Example

Complete workflow from scratch:

```bash
# 1. Clean any existing data
poetry run cli clean --local-dir 'demo_chunks'

# 2. Save documents with semantic chunking
poetry run cli save data semantic \
  --local-dir 'demo_chunks' \
  --config '{"threshold_mode": "percentile", "threshold_value": 95.0, "min_sentences": 2}'

# 3. Search for relevant chunks
poetry run cli search "What are the Server Error Codes?" \
  --local-dir 'demo_chunks' \
  --top-k 5 \
  --candidates 20

# 4. Ask questions
poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'demo_chunks' \
  --top-k 5 \
  --candidates 20

poetry run cli talk "What are the Server Error Codes?" \
  --local-dir 'demo_chunks' \
  --top-k 3 \
  --no-rerank

# 5. Clean up
poetry run cli clean --local-dir 'demo_chunks'
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
