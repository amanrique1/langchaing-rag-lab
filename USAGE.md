# Usage Guide - LangChain RAG Lab

This guide provides comprehensive instructions for installing, configuring, and running the LangChain RAG Lab system.

> **📚 Looking for architecture details?** See [README.md](README.md) for conceptual information and system design.

## Table of Contents

- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Command Structure](#command-structure)
  - [Subcommands](#subcommands)
  - [Storage Options](#storage-options-all-commands)
  - [Arguments and Options](#arguments-and-options)
  - [Configuration Options](#configuration-options)
- [Configuration Details](#configuration-details)
- [Comprehensive Testing Examples](#comprehensive-testing-examples)
  - [Testing with LanceDB](#testing-with-lancedb-default-)
  - [Testing with ChromaDB](#testing-with-chromadb-explicit-opt-in)
  - [Testing with FileSystem](#testing-with-filesystem-storage)
  - [Full Workflow Example](#full-workflow-example-lancedb-with-custom-paths)
  - [Backend Comparison Example](#backend-comparison-example)
- [RAG Evaluation](#rag-evaluation)
- [Running Tests](#running-tests)
- [Linting and Formatting](#linting-and-formatting)
- [Troubleshooting](#troubleshooting)

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
| `search` | Searches for document chunks most relevant to a query and displays them. |
| `talk` | Asks a question, retrieves relevant documents, and generates a conversational answer. |
| `chat` | Starts an interactive chat session with memory. |
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
*   **`--expand <strategy>`**: Optional query expansion strategy (`hyde`, `stepback`, `subqueries`, `zero_shot`, or `few_shot`).
*   **`--no-rerank`**: Disable reranking completely.
*   **`--llm-rerank`**: Use LLM-based reranking instead of the default Encoder-based reranking.
*   **`--verbose`**: Enable verbose output for debugging.
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

#### `search` Subcommand
`poetry run cli search <query> [OPTIONS]`
*   **`query`**: (Required) The search term or phrase.
*   **`--top-k <number>`**: Optional number of relevant chunks to retrieve. Default is `5`.
*   **`--candidates <number>`**: Optional number of candidates to retrieve before reranking. Default is `20`.
*   **`--expand <strategy>`**: Optional query expansion strategy (`hyde`, `stepback`, `subqueries`, `zero_shot`, or `few_shot`).
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

#### `chat` Subcommand
`poetry run cli chat [OPTIONS]`

**Description**: Start an interactive chat session with conversation memory using modern LangChain message-based history.

**Options**:
*   **`--user-id <str>`**: User ID for session context (default: `default_user`)
*   **`--session-id <str>`**: Session ID to group conversation turns (auto-generated if not provided)
*   **`--window <number>`**: Number of conversation exchanges (pairs) to keep in memory (default: `5`)
*   **`--top-k <number>`**: Number of final chunks to use for answer generation (default: `5`)
*   **`--candidates <number>`**: Number of candidates to retrieve before reranking (default: `20`)
*   **`--expand <strategy>`**: Query expansion strategy (`hyde`, `stepback`, `subqueries`, `zero_shot`, or `few_shot`)
*   **`--no-rerank`**: Disable reranking completely
*   **`--llm-rerank`**: Use LLM-based reranking instead of default Encoder-based reranking
*   **`--verbose`**: Enable verbose output for debugging
*   **Storage Options**: `--collection`, `--storage-path`, `--lance`, `--chroma`, `--filesystem`, `--single-collection`

**Interactive Commands** (inside Chat Mode):
*   **`/exit` or `/quit`**: End the chat session and save conversation history
*   **`/clear`**: Clear the current conversation history
*   **`/history`**: Display recent conversation in a formatted table
*   **`/stats`**: Show session statistics (messages, exchanges, memory window)
*   **`/sessions`**: Display all active chat sessions in the container
*   **`/help`**: Show available commands and usage tips

**Session Management**:
- Chat sessions are cached by `(user_id, session_id)` pair
- Resuming with the same IDs continues the previous conversation
- Each session maintains independent conversation history
- Sessions persist until explicitly cleared or container is reset

### Configuration Options

The enhanced pipeline is highly configurable:

| Option | Effect | Default |
|--------|--------|--------|
| `--expand hyde` | Use HyDE query expansion | None |
| `--expand stepback` | Use Step-Back query expansion | None |
| `--expand subqueries` | Use Subqueries query expansion | None |
| `--expand zero_shot` | Use Zero-Shot query expansion | None |
| `--expand few_shot` | Use Few-Shot query expansion | None |
| `--single-collection` | Use only content search (Disable Ensemble) | False |
| `--no-rerank` | Skip reranking step | False |
| `--llm-rerank` | Enable Gemini-based reranking | False |
| `--candidates N` | Candidates before reranking | 20 |
| `--top-k K` | Final results for answer | 5 |
| `--lance` | Use LanceDB instead of ChromaDB | True |
| `--chroma` | Use ChromaDB instead of LanceDB | False |
| `--filesystem` | Use FileSystem instead of LanceDB | False |
| `--storage-path <path>` | Custom storage directory | None (uses default) |

**Query Expansion Strategies**:
- **`hyde`**: Generates a hypothetical document that would answer the query, then searches for similar content
- **`stepback`**: Creates broader, conceptual versions of specific queries for better context retrieval
- **`subqueries`**: Decomposes complex queries into multiple focused sub-questions for comprehensive answers
- **`zero_shot`**: Uses prompt engineering to reformulate queries without examples
- **`few_shot`**: Leverages example-based learning to improve query understanding and expansion

**Performance Profiles**:
- **Maximum Accuracy**: `--expand hyde --llm-rerank --candidates 30 --top-k 5` (Best results, highest latency & cost)
- **High Accuracy**: `--expand stepback --llm-rerank --candidates 20 --top-k 5` (Excellent results, moderate latency)
- **Complex Queries**: `--expand subqueries --candidates 25 --top-k 5` (Best for multi-part questions)
- **Balanced (Default)**: `--candidates 20 --top-k 5` (Uses local Encoder reranking + LanceDB hybrid search, fast & free)
- **Fast**: `--single-collection --no-rerank --top-k 5` (Lowest latency, no ensemble, no reranking)

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
poetry run cli save assets/data full_doc --clean

# Length-based with custom collection and location
poetry run cli save assets/data length_based \
  --collection 'length_docs' \
  --storage-path ./my_vectordb \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Structure-based with custom collection only
poetry run cli save assets/data structure_based \
  --collection 'structure_docs' \
  --config '{"chunk_size": 1500, "chunk_overlap": 100}'

# Semantic with default settings
poetry run cli save assets/data semantic \
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

# With Subqueries expansion (for complex questions)
poetry run cli search "What are the authentication methods and error handling strategies?" \
  --expand subqueries \
  --top-k 5 \
  --candidates 25

# With Zero-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --expand zero_shot \
  --top-k 5 \
  --candidates 20

# With Few-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --expand few_shot \
  --top-k 5 \
  --candidates 20

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

# Complex multi-part questions (Subqueries)
poetry run cli talk "Explain authentication methods, rate limiting, and error codes" \
  --expand subqueries \
  --top-k 5 \
  --candidates 25

# Zero-Shot expansion
poetry run cli talk "What are the Server Error Codes?" \
  --expand zero_shot \
  --top-k 5 \
  --candidates 20

# Few-Shot expansion
poetry run cli talk "What are the Server Error Codes?" \
  --expand few_shot \
  --top-k 5 \
  --candidates 20

# Fast mode
poetry run cli talk "What are the Server Error Codes?" \
  --top-k 3 \
  --single-collection \
  --no-rerank
```

#### 4. Chat with LanceDB (Interactive with Persistent Memory)
```bash
# Default LanceDB chat
poetry run cli chat

# Chat with specific collection and user ID
poetry run cli chat --collection 'structure_docs' --user-id 'andres_m' --window 5

# Chat with HyDE expansion and LLM reranking
poetry run cli chat --collection 'structure_docs' --expand hyde --llm-rerank --top-k 5
```

#### 5. Manage LanceDB Collections
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
poetry run cli save assets/data structure_based \
  --chroma \
  --collection 'chroma_docs' \
  --clean

# Save to ChromaDB with custom location
poetry run cli save assets/data structure_based \
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

# ChromaDB with Subqueries expansion
poetry run cli search "What are authentication and authorization mechanisms?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand subqueries \
  --top-k 5

# ChromaDB with Zero-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand zero_shot \
  --top-k 5

# ChromaDB with Few-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand few_shot \
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

# ChromaDB with Subqueries for complex questions
poetry run cli talk "How do I implement authentication, handle errors, and manage rate limits?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand subqueries \
  --top-k 5

# ChromaDB with Zero-Shot
poetry run cli talk "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand zero_shot \
  --top-k 5

# ChromaDB with Few-Shot
poetry run cli talk "What are the Server Error Codes?" \
  --chroma \
  --collection 'chroma_docs' \
  --expand few_shot \
  --top-k 5
```

#### 4. Chat with ChromaDB (Interactive)
```bash
# Chat with ChromaDB backend
poetry run cli chat --chroma --collection 'chroma_docs'
```

#### 5. Manage ChromaDB Collections
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
poetry run cli save assets/data length_based \
  --filesystem \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}'

# Structure-based with custom location and collection
poetry run cli save assets/data structure_based \
  --filesystem \
  --collection 'fs_docs' \
  --storage-path ./my_filesystem_db \
  --config '{"chunk_size": 1500, "chunk_overlap": 150}'

# Semantic with default location
poetry run cli save assets/data semantic \
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

# With Subqueries expansion
poetry run cli search "What are the API design patterns and best practices?" \
  --filesystem \
  --expand subqueries \
  --top-k 5

# With Zero-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --filesystem \
  --expand zero_shot \
  --top-k 5

# With Few-Shot expansion
poetry run cli search "What are the Server Error Codes?" \
  --filesystem \
  --expand few_shot \
  --top-k 5
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

# With Subqueries
poetry run cli talk "Explain versioning, authentication, and pagination strategies" \
  --filesystem \
  --expand subqueries \
  --top-k 5

# With Zero-Shot
poetry run cli talk "What are the Server Error Codes?" \
  --filesystem \
  --expand zero_shot \
  --top-k 5

# With Few-Shot
poetry run cli talk "What are the Server Error Codes?" \
  --filesystem \
  --expand few_shot \
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
poetry run cli save assets/data semantic \
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

# 6. Complex multi-part questions with Subqueries
poetry run cli talk "What are the authentication methods, rate limiting policies, and error handling best practices?" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --expand subqueries \
  --top-k 5 \
  --candidates 25

# 7. Zero-Shot expansion
poetry run cli talk "How should I handle API versioning?" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --expand zero_shot \
  --top-k 5

# 8. Few-Shot expansion
poetry run cli talk "What are common API security concerns?" \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb \
  --expand few_shot \
  --top-k 5

# 9. View storage info
poetry run cli info \
  --collection 'production_docs' \
  --storage-path /data/production/vectordb

# 10. Clean up
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
poetry run cli save assets/data semantic \
  --collection 'lance_test' \
  --storage-path ./test_dbs/lance \
  --clean

poetry run cli save assets/data semantic \
  --chroma \
  --collection 'chroma_test' \
  --storage-path ./test_dbs/chroma \
  --clean

poetry run cli save assets/data semantic \
  --filesystem \
  --collection 'fs_test' \
  --storage-path ./test_dbs/filesystem \
  --clean

# Compare: Same query, different backends, different expansion strategies
echo "=== LanceDB with HyDE ==="
poetry run cli search "API error handling" \
  --collection 'lance_test' \
  --storage-path ./test_dbs/lance \
  --expand hyde \
  --top-k 3

echo "=== ChromaDB with Step-Back ==="
poetry run cli search "API error handling" \
  --chroma \
  --collection 'chroma_test' \
  --storage-path ./test_dbs/chroma \
  --expand stepback \
  --top-k 3

echo "=== FileSystem with Subqueries ==="
poetry run cli search "API error handling" \
  --filesystem \
  --collection 'fs_test' \
  --storage-path ./test_dbs/filesystem \
  --expand subqueries \
  --top-k 3

echo "=== LanceDB with Zero-Shot ==="
poetry run cli search "API error handling" \
  --collection 'lance_test' \
  --storage-path ./test_dbs/lance \
  --expand zero_shot \
  --top-k 3

echo "=== ChromaDB with Few-Shot ==="
poetry run cli search "API error handling" \
  --chroma \
  --collection 'chroma_test' \
  --storage-path ./test_dbs/chroma \
  --expand few_shot \
  --top-k 3

# Cleanup
poetry run cli clean --collection 'lance_test' --storage-path ./test_dbs/lance --force
poetry run cli clean --chroma --collection 'chroma_test' --storage-path ./test_dbs/chroma --force
poetry run cli clean --filesystem --collection 'fs_test' --storage-path ./test_dbs/filesystem --force
```

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

**Strategy-Specific Notes**:
- **`subqueries`**: Generates multiple sub-questions, may take longer but provides comprehensive coverage for complex queries
- **`zero_shot` & `few_shot`**: Generally faster than HyDE/Step-Back as they focus on query reformulation
- **`hyde`**: Slowest but most effective for finding semantically similar content
- **`stepback`**: Good balance between speed and effectiveness for overly specific queries

#### 3. **Query Expansion Not Improving Results**

**Analysis**:
- **HyDE** works best for questions with detailed answers in documents
- **Step-Back** works best for overly specific questions
- **Subqueries** works best for multi-part or complex questions that need decomposition
- **Zero-Shot** works best when you want simple query reformulation without examples
- **Few-Shot** works best when you want guided query understanding with minimal examples
- For simple keyword searches, expansion may not help

**Recommendation**: A/B test with and without expansion for your use case. Try different strategies:
```bash
# Compare expansion strategies for the same query
poetry run cli search "complex question" --expand hyde
poetry run cli search "complex question" --expand stepback
poetry run cli search "complex question" --expand subqueries
poetry run cli search "complex question" --expand zero_shot
poetry run cli search "complex question" --expand few_shot
```

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
poetry run cli save assets/data semantic --collection 'project_a_docs'
poetry run cli save assets/data semantic --collection 'project_b_docs'

# Organize by data type
poetry run cli save assets/data semantic --collection 'technical_docs'
poetry run cli save assets/data semantic --collection 'business_docs'

# Use custom paths for isolation
poetry run cli save assets/data semantic \
  --collection 'prod_docs' \
  --storage-path /data/production/vectordb

poetry run cli save assets/data semantic \
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
poetry run cli save assets/data semantic --collection 'docs'  # → ./lancedb/docs
poetry run cli save assets/data semantic --collection 'docs' --storage-path ./custom  # → ./custom/docs
```

#### 8. **Choosing the Right Expansion Strategy**

**Decision Guide**:

| Query Type | Recommended Strategy | Reason |
|------------|---------------------|---------|
| Simple keyword search | None or `zero_shot` | Minimal overhead, direct matching |
| Specific technical question | `hyde` | Generates hypothetical detailed answer |
| Overly specific query | `stepback` | Broadens to conceptual level |
| Multi-part complex question | `subqueries` | Decomposes into focused sub-questions |
| Need quick reformulation | `zero_shot` | Fast, no example overhead |
| Domain-specific queries | `few_shot` | Leverages examples for better understanding |

**Examples**:
```bash
# Simple: "API authentication"
poetry run cli search "API authentication"  # No expansion needed

# Specific: "What is JWT token expiration handling?"
poetry run cli search "What is JWT token expiration handling?" --expand hyde

# Overly specific: "How to fix 401 error in POST /api/v2/users endpoint?"
poetry run cli search "How to fix 401 error in POST /api/v2/users endpoint?" --expand stepback

# Complex: "Explain authentication, authorization, and rate limiting"
poetry run cli search "Explain authentication, authorization, and rate limiting" --expand subqueries

# Need reformulation: "server problems"
poetry run cli search "server problems" --expand zero_shot

# Domain-specific: "microservices communication patterns"
poetry run cli search "microservices communication patterns" --expand few_shot
```

---

**Need more help?** Check the [README.md](README.md) for architecture details or open an issue on GitHub.