# LangChain RAG Lab

This project serves as a **conversational AI lab**, providing a flexible framework for **Retrieval Augmented Generation (RAG) pipelines**. It focuses on intelligently chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability, scalability, and modularity.

## Features

*   **Multiple Chunking Strategies**: Supports Length-Based, Structure-Based, and Semantic Chunking.
*   **Hexagonal Architecture**: Clean separation of concerns for robust and testable code.
*   **Pluggable Chunk Stores**: Stores processed chunks in either the local file system or ChromaDB.
*   **Modern CLI**: Easy-to-use subcommand-based interface (`save`, `talk`, `search`, `clean`).
*   **Google Gemini Integration**: Uses Google's embedding and language models.
*   **RAG Evaluation**: Built-in evaluation suite using the Ragas library to measure performance.

## Technologies Used

*   **Python 3.11+**: The primary programming language.
*   **Poetry**: For dependency management and project packaging.
*   **Ruff**: For linting and formatting.
*   **LangChain**: A framework for developing applications powered by language models.
*   **ChromaDB**: An open-source embedding database for storing and retrieving document chunks.
*   **Google Gemini**: Embedding and language models.
*   **Ragas**: A framework for evaluating RAG pipelines.
*   **NLTK**: Natural Language Toolkit for sentence tokenization.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Architecture](#architecture)
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

## Architecture

This project is built using a **Hexagonal Architecture** (also known as Ports and Adapters). This design pattern isolates the core application logic from external concerns such as databases, user interfaces, and third-party APIs.

### Layers

*   **Domain Layer** (`src/domain`): Contains the business logic, data models (`Document`, `Chunk`), and abstract definitions for chunking strategies (`ChunkingStrategy`). This layer is independent of any infrastructure concerns.
    
*   **Application Layer** (`src/application`): Contains use cases (`ChunkingUseCase`, `StorageUseCase`, `TalkUseCase`) that orchestrate the flow of data and apply domain logic. Defines ports (interfaces) for external services:
    *   `DocumentLoader`: Interface for loading documents.
    *   `ChunkStore`: Interface for storing and retrieving chunks.
    *   `LanguageModel`: Interface for interacting with a language model.

*   **Infrastructure Layer** (`src/infrastructure`): Provides concrete implementations (adapters):
    *   **Document Loaders**: `MarkdownDocumentLoader`.
    *   **Chunk Stores**: `FileSystemChunkStore`, `ChromaChunkStore`.
    *   **Language Models**: `GoogleGenAILanguageModel`.
    *   **CLI**: Command-line interface (`main.py`).

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

#### Talk Command Data Flow
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
┌─────────────────┐
│ Language Model  │ (GoogleGenAILanguageModel)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Generated      │
│    Answer       │
└─────────────────┘
```

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
*   **`--top-k <number>`**: Optional number of relevant chunks to retrieve. Default is `5`.

#### `search` Subcommand
`poetry run cli search <query> [OPTIONS]`
*   **`query`**: (Required) The search term or phrase.
*   **`--top-k <number>`**: Optional number of relevant chunks to retrieve. Default is `5`.

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
| `breakpoint_threshold_type` | string | Threshold algorithm | `percentile` |
| `breakpoint_threshold_amount` | float | Value for the threshold type | `95.0` |
| `min_chunk_size` | int | Min sentences per chunk | `1` |
| `max_chunk_size` | int | Max sentences per chunk | `null` |

---

## Examples

### Example 1: Basic Length-Based Chunking (Local Storage)

```bash
poetry run cli save data length_based \
  --config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}' \
  --local-dir 'output_chunks/length_based'
```

### Example 2: Structure-Based Chunking (ChromaDB)

```bash
poetry run cli save data structure_based \
  --config '{"chunk_size": 1500, "chunk_overlap": 100, "max_header_levels": 4}' \
  --chroma-collection 'technical_docs'
```

### Example 3: Semantic Chunking with Custom Threshold

```bash
poetry run cli save data semantic \
  --config '{"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 90.0, "min_chunk_size": 2, "max_chunk_size": 10}' \
  --chroma-collection 'research_papers'
```

### Example 4: Talk to Your Documents

First, save your documents to a collection:
```bash
poetry run cli save data semantic --chroma-collection 'my_docs'
```

Then, use the `talk` subcommand to ask a question:
```bash
poetry run cli talk "What are the main software architecture principles?" --chroma-collection 'my_docs'
```

### Example 5: Search for Relevant Chunks

First, ensure your documents are saved to a collection (e.g., `my_docs`).

Then, use the `search` subcommand:
```bash
poetry run cli search "What is hexagonal architecture?" --chroma-collection 'my_docs' --top-k 3
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

#### 2. **"NLTK punkt tokenizer not found"**

**Solution**:
```bash
poetry run python -c "import nltk; nltk.download('punkt')"
```
