# LangChain RAG Lab

This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.

## Features

*   **Multiple Chunking Strategies**: Supports Length-Based, Structure-Based, and Semantic Chunking.
*   **Hexagonal Architecture**: Clean separation of concerns for robust and testable code.
*   **Pluggable Chunk Stores**: Stores processed chunks in either the local file system or ChromaDB.
*   **Task-Based CLI**: Easy-to-use command-line interface for document processing, searching, and management.
*   **Google Gemini Integration**: Uses Google's embedding models for semantic chunking.

## Technologies Used

*   **Python 3.11+**: The primary programming language.
*   **Poetry**: For dependency management and project packaging.
*   **LangChain**: A framework for developing applications powered by language models.
*   **ChromaDB**: An open-source embedding database for storing and retrieving document chunks.
*   **Google Gemini**: Embedding model (`models/embedding-001`) for semantic chunking.
*   **NLTK**: Natural Language Toolkit for sentence tokenization.
*   **scikit-learn**: For cosine similarity calculations in semantic chunking.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
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

    *   **Key Features**:
        *   Respects document hierarchy (`#`, `##`, `###`, etc.)
        *   Configurable header levels
        *   Optional header stripping
        *   Falls back to length-based chunking for oversized sections

3.  **Semantic Chunking**:
    This advanced technique uses NLP models to split the text based on semantic similarity. It identifies topic shifts and creates chunks that are contextually coherent. It's best for unstructured or semi-structured documents where preserving meaning is crucial.

    *   **Key Features**:
        *   Uses sentence embeddings (Google Gemini)
        *   Multiple threshold algorithms
        *   Configurable min/max chunk sizes
        *   Intelligent breakpoint detection

## Architecture

This project is built using a **Hexagonal Architecture** (also known as Ports and Adapters). This design pattern isolates the core application logic from external concerns such as databases, user interfaces, and third-party APIs.

### Layers

*   **Domain Layer** (`src/domain`): Contains the business logic, data models (`Document`, `Chunk`), and abstract definitions for chunking strategies (`ChunkingStrategy`). This layer is independent of any infrastructure concerns.
    
*   **Application Layer** (`src/application`): Contains use cases (`ChunkingUseCase`) that orchestrate the flow of data and apply domain logic. Defines ports (interfaces) for external services:
    *   `DocumentLoader`: Interface for loading documents
    *   `ChunkStore`: Interface for storing and retrieving chunks

*   **Infrastructure Layer** (`src/infrastructure`): Provides concrete implementations (adapters):
    *   **Document Loaders**: `MarkdownDocumentLoader`
    *   **Chunk Stores**: `FileSystemChunkStore`, `ChromaChunkStore`
    *   **CLI**: Command-line interface (`main.py`)

### Data Flow Overview

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
│   Chunking      │ (Strategy Pattern)
│   Strategy      │ - Length-Based
│                 │ - Structure-Based
│                 │ - Semantic
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

## Use Cases

| Strategy | Best For | When to Use It | Performance |
| :--- | :--- | :--- | :--- |
| **Length-Based** | Quick and simple chunking | Ideal for unstructured text where content hierarchy is not important. Good for initial tests or when you need a baseline chunking method. | ⚡ Fast |
| **Structure-Based** | Well-structured documents (e.g., Markdown, HTML) | Use for technical documentation, legal documents, or any text with clear headings. It preserves the document's logical sections. | ⚡ Fast |
| **Semantic** | Complex, narrative, or semi-structured documents | Best for research papers, articles, or any text where maintaining the flow of ideas is critical. It creates context-aware chunks that are better for RAG systems. | 🐢 Slow (requires embeddings) |

## Technical Deep Dive

### Length-Based Chunking

This method is the most basic approach. It iterates through the text and splits it into chunks of a fixed size.

*   **How it works**: It can split by character count or by token count (using a tokenizer). The `chunk_overlap` parameter allows for a certain number of characters or tokens to be repeated at the beginning of the next chunk to maintain some context.
*   **Pros**: Simple, fast, and predictable. No external dependencies.
*   **Cons**: Ignores sentence boundaries, logical structure, and semantic meaning. This can lead to chunks that are not coherent.
*   **Use Case**: Baseline chunking, quick prototyping, simple documents.

### Structure-Based Chunking

This method is designed for documents with a clear hierarchical structure, such as Markdown files with headers.

*   **How it works**: It uses LangChain's `MarkdownHeaderTextSplitter` to split documents based on headers (`#`, `##`, `###`, etc.). Each section under a header is treated as a potential chunk. If a section exceeds `chunk_size`, it's further subdivided using a length-based approach.
*   **Pros**: Preserves the logical structure of the document, creating chunks that are more coherent and contextually relevant. Respects document hierarchy.
*   **Cons**: Only effective for documents with a well-defined structure. It will perform poorly on unstructured text.
*   **Use Case**: Technical documentation, API docs, structured articles, legal documents.

### Semantic Chunking

This is the most advanced method, using machine learning to split the text based on its meaning.

*   **How it works**: 
    1. Text is split into sentences using NLTK's `sent_tokenize`
    2. Each sentence is converted to an embedding vector using Google Gemini's `models/embedding-001`
    3. Cosine similarity is calculated between consecutive sentence embeddings
    4. A threshold is applied to identify "breakpoints" (significant drops in similarity)
    5. Text is chunked at these breakpoints, indicating topic shifts

*   **Threshold Types**:
    *   **Percentile**: Uses the Nth percentile of all similarities as the threshold
    *   **Standard Deviation**: Uses mean - (N * std) as the threshold
    *   **Interquartile**: Uses Q1 - (N * IQR) as the threshold
    *   **Absolute**: Uses a fixed similarity value as the threshold

*   **Pros**: Creates the most coherent and contextually relevant chunks, as it is based on the meaning of the text. Best for RAG systems.
*   **Cons**: Computationally expensive and slower than other methods. Requires API calls to Google's embedding service. The quality depends on the embedding model.
*   **Use Case**: Research papers, narrative content, complex technical articles, conversational documents.

---

## Installation

### Prerequisites

*   **Python 3.11+**: Ensure you have Python installed
*   **Poetry**: Package manager for Python

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
# Google API Key (required for semantic chunking)
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Override default embedding model
EMBEDDING_MODEL=models/embedding-001

# Optional: ChromaDB settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Getting a Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and add it to your `.env` file

**Note**: The semantic chunking strategy requires a valid Google API key. Other strategies work without it.

---

## Usage

The project uses a **task-based CLI** for different operations.

### Command Structure

```bash
poetry run cli [TASK] [SOURCE] [STRATEGY] [OPTIONS]
```

### Available Tasks

| Task | Description | Required Arguments |
| :--- | :--- | :--- |
| `save` | Process and save document chunks | `SOURCE`, `STRATEGY` |
| `search` | Search for similar chunks | `--query` |
| `talk` | Ask a question and get an answer from the documents | `--query` |
| `delete` | Not yet implemented. | - |
| `clean` | Clear the chunk store | - |

### Common Arguments

*   `SOURCE`: Path to the folder with markdown files to be processed
*   `STRATEGY`: Chunking strategy to use (`length_based`, `structure_based`, `semantic`)

### Options

| Option | Description | Default |
| :--- | :--- | :--- |
| `--config` | JSON string with strategy configuration | `{}` |
| `--query` | The question to ask for `talk` and `search` tasks | `None` |
| `--top-k` | Number of top relevant chunks to retrieve for `search` and `talk` tasks | `5` |
| `--local` | Store chunks locally in the file system | `false` (uses ChromaDB) |
| `--output-dir` | Directory to save chunks (with `--local`) | `output_chunks` |
| `--collection-name` | ChromaDB collection name | `default_collection` |

---

## Configuration Details

The `--config` option accepts a JSON string to customize the behavior of each strategy.

### `length_based`

| Parameter | Type | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_size` | int | Max size of each chunk | `1000` | `1000` |
| `chunk_overlap` | int | Overlap between chunks | `200` | `200` |
| `mode` | string | Splitting mode: `character` or `token` | `character` | `"character"` |

**Example Config**:
```json
{
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "mode": "character"
}
```

### `structure_based`

| Parameter | Type | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `chunk_size` | int | Max size after splitting by headers | `1000` | `1000` |
| `chunk_overlap` | int | Overlap between sub-chunks | `200` | `200` |
| `strip_headers` | bool | Remove header text from chunks | `false` | `false` |
| `max_header_levels` | int | Max header level to consider (1-6) | `3` | `4` |

**Example Config**:
```json
{
  "chunk_size": 1500,
  "chunk_overlap": 100,
  "strip_headers": false,
  "max_header_levels": 4
}
```

### `semantic`

| Parameter | Type | Description | Default | Example |
| :--- | :--- | :--- | :--- | :--- |
| `breakpoint_threshold_type` | string | Threshold algorithm (see below) | `percentile` | `"percentile"` |
| `breakpoint_threshold_amount` | float | Value for the threshold type | `95.0` | `95.0` |
| `min_chunk_size` | int | Min sentences per chunk | `1` | `1` |
| `max_chunk_size` | int | Max sentences per chunk | `null` | `10` |

**Threshold Types**:

| Type | Description | Typical Values | Effect |
| :--- | :--- | :--- | :--- |
| `percentile` | Breaks at Nth percentile of similarity | `85.0` - `95.0` | Higher = fewer, larger chunks |
| `standard_deviation` | Breaks at mean - (N × std) | `1.0` - `2.0` | Higher = fewer chunks |
| `interquartile` | Breaks at Q1 - (N × IQR) | `1.0` - `2.0` | Higher = fewer chunks |
| `absolute` | Breaks at fixed similarity value | `0.7` - `0.9` | Higher = fewer chunks |

**Example Config**:
```json
{
  "breakpoint_threshold_type": "percentile",
  "breakpoint_threshold_amount": 90.0,
  "min_chunk_size": 2,
  "max_chunk_size": 15
}
```

**Threshold Tuning Guide**:

| Goal | Recommended Settings |
| :--- | :--- |
| **More, smaller chunks** (fine-grained retrieval) | `percentile: 85`, `absolute: 0.75` |
| **Balanced chunks** (general purpose) | `percentile: 90`, `absolute: 0.80` |
| **Fewer, larger chunks** (context preservation) | `percentile: 95`, `absolute: 0.85` |

---

## Examples

### Example 1: Basic Length-Based Chunking (Local Storage)

**Scenario**: Quick chunking for testing, stored locally.

```bash
poetry run cli save data length_based \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}' \
  --local \
  --output-dir 'output_chunks/length_based'
```

**Output**: Creates JSON files in `output_chunks/length_based/`

---

### Example 2: Structure-Based Chunking (ChromaDB)

**Scenario**: Index technical documentation with preserved structure.

```bash
poetry run cli save data structure_based \
  --config '{"chunk_size": 1500, "chunk_overlap": 100, "max_header_levels": 4}' \
  --collection-name 'technical_docs'
```

**Output**: Stores chunks in ChromaDB collection `technical_docs`

---

### Example 3: Semantic Chunking with Custom Threshold

**Scenario**: Research papers with context-aware chunking.

```bash
poetry run cli save data semantic \
  --config '{"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 90.0, "min_chunk_size": 2, "max_chunk_size": 10}' \
  --collection-name 'research_papers'
```

**Output**: Creates semantically coherent chunks in ChromaDB

---

### Example 4: Token-Based Chunking

**Scenario**: Chunking optimized for LLM token limits.

```bash
poetry run cli save data length_based \
  --config '{"chunk_size": 512, "chunk_overlap": 50, "mode": "token"}' \
  --collection-name 'token_chunks'
```

---

### Example 5: Cleaning Stores

**Clean ChromaDB Collection**:
```bash
poetry run cli clean --collection-name 'technical_docs'
```

**Clean Local Directory**:
```bash
poetry run cli clean --local --output-dir 'output_chunks'
```

---

### Example 6: Talk to Your Documents

**Scenario**: Ask a question about the content of your documents.

First, save your documents to a collection:
```bash
poetry run cli save data semantic --collection-name 'my_docs'
```

Then, ask a question using the `talk` task:
```bash
poetry run cli talk --query "What are the main software architecture principles?" --collection-name 'my_docs'
```

**Output**:
```
Question: What are the main software architecture principles?

Answer: The main software architecture principles are... (The model will generate an answer based on the retrieved content).
```

---

### Example 7: Search for Relevant Chunks

**Scenario**: Find document chunks most relevant to a specific query.

First, ensure your documents are saved to a collection (e.g., `my_docs` as in the previous example):
```bash
poetry run cli save data semantic --collection-name 'my_docs'
```

Then, search for relevant chunks using the `search` task:
```bash
poetry run cli search --query "What is hexagonal architecture?" --collection-name 'my_docs' --top-k 3
```

**Output**:
```
Found 3 relevant chunks:

--- Chunk 1 ---
Content: This project is built using a **Hexagonal Architecture** (also known as Ports and Adapters). This design pattern isolates the core application logic from external concerns such as databases, user interfaces, and third-party APIs.
Metadata: {'source': 'data/software_architecture_guide.md', 'header': 'Architecture'}

--- Chunk 2 ---
Content: The main software architecture principles are... (The model will generate an answer based on the retrieved content).
Metadata: {'source': 'data/software_architecture_guide.md', 'header': 'Core Concepts'}

--- Chunk 3 ---
Content: This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.
Metadata: {'source': 'data/README.md', 'header': 'LangChain RAG Lab'}
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

### Run Specific Test File

```bash
poetry run pytest tests/domain/strategies/test_semantic_chunking.py
```

### Run Tests in Verbose Mode

```bash
poetry run pytest -v
```

**Test Coverage**: The project maintains >90% code coverage across all modules.

---

## Troubleshooting

### Common Issues

#### 1. **"GOOGLE_API_KEY not set" Error**

**Problem**: Semantic chunking requires a Google API key.

**Solution**:
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_key_here" > .env
```

#### 2. **"NLTK punkt tokenizer not found"**

**Problem**: NLTK sentence tokenizer is not downloaded.

**Solution**:
```bash
poetry run python -c "import nltk; nltk.download('punkt')"
```

#### 3. **ChromaDB Connection Issues**

**Problem**: ChromaDB fails to persist data.

**Solution**:
```bash
# Ensure directory exists and has write permissions
mkdir -p ./chroma_db
chmod 755 ./chroma_db
```

#### 4. **Import Errors**

**Problem**: Modules not found when running CLI.

**Solution**:
```bash
# Reinstall dependencies
poetry install --no-cache
```

#### 5. **Slow Semantic Chunking**

**Problem**: Semantic chunking takes too long.

**Solution**:
- Use `length_based` or `structure_based` for faster processing
- Reduce document size
- Use batch mode sparingly
- Consider caching embeddings (future feature)

---

## Project Structure

```
langchain-rag-lab/
├── data/                          # Sample markdown documents
├── src/
│   ├── domain/                    # Core business logic
│   │   ├── models/                # Domain entities (Document, Chunk)
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   └── enums.py
│   │   └── strategies/            # Chunking strategies
│   │       ├── chunking_strategy.py
│   │       ├── length_based_chunking.py
│   │       ├── structure_based_chunking.py
│   │       └── semantic_chunking.py
│   ├── application/               # Use cases and ports
│   │   ├── ports/                 # Interfaces
│   │   │   ├── document_loader.py
│   │   │   └── chunk_store.py
│   │   ├── services/
│   │   │   └── chunking_service.py
│   │   └── use_cases/
│   │       └── chunking_use_case.py
│   └── infrastructure/            # Adapters and external integrations
│       ├── adapters/
│       │   ├── document_loaders/
│       │   │   └── markdown_loader.py
│       │   └── chunk_stores/
│       │       ├── file_system_chunk_store.py
│       │       └── chroma_chunk_store.py
│       └── cli/
│           └── main.py            # CLI entry point
├── tests/                         # Unit and integration tests
├── output_chunks/                 # Default local output directory
├── chroma_db/                     # ChromaDB persistence
├── .env                           # Environment variables
├── pyproject.toml                 # Poetry configuration
└── README.md
```

---

## Roadmap

- [ ] **Delete Operations**: Add chunk deletion by ID or filter
- [ ] **Additional Loaders**: Support for PDF, DOCX, HTML
- [ ] **Embedding Cache**: Cache embeddings to speed up semantic chunking
- [ ] **Async Processing**: Support for asynchronous document processing
- [ ] **Custom Embedding Models**: Support for other embedding providers (Ollama, OpenAI, HuggingFace)
- [ ] **Chunk Evaluation**: Metrics for chunk quality assessment

---

## Development Setup

```bash
# Install dev dependencies
poetry install --with dev

# Run linting
poetry run black src/
poetry run pylint src/

# Run tests before committing
poetry run pytest --cov=src
```