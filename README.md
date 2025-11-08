# LangChain RAG Lab

This project provides a flexible framework for chunking text documents using various strategies, built with a hexagonal architecture to ensure maintainability and scalability.

## Core Concepts

This project is designed around three core chunking strategies, each suited for different types of documents and use cases. Understanding these strategies will help you choose the best approach for your needs.

### Chunking Strategies

1.  **Length-Based Chunking**:
    This is the most straightforward method. It splits the text into chunks of a specified size, with an optional overlap between them. It's fast and simple but doesn't consider the content's structure or meaning.

2.  **Structure-Based Chunking**:
    This method leverages the document's structure, such as Markdown headers, to create more meaningful chunks. It's ideal for well-structured documents like technical manuals or articles, as it keeps related content together.

3.  **Semantic Chunking**:
    This advanced technique uses NLP models to split the text based on semantic similarity. It identifies topic shifts and creates chunks that are contextually coherent. It's best for unstructured or semi-structured documents where preserving meaning is crucial.

## Architecture

This project is built using a **Hexagonal Architecture** (also known as Ports and Adapters). This design pattern isolates the core application logic from external concerns such as databases, user interfaces, and third-party APIs.

*   **Core Application (Domain)**: Contains the business logic of the application, including the chunking strategies and data models.
*   **Ports**: Defines the interfaces for interacting with the core application.
*   **Adapters**: Implements the ports and provides the concrete implementations for external services, such as the CLI, document loaders, and chunk stores.

This architecture makes the project more modular, testable, and easier to maintain.


## Use Cases

| Strategy | Best For | When to Use It |
| :--- | :--- | :--- |
| **Length-Based** | Quick and simple chunking | Ideal for unstructured text where content hierarchy is not important. Good for initial tests or when you need a baseline chunking method. |
| **Structure-Based** | Well-structured documents (e.g., Markdown, HTML) | Use for technical documentation, legal documents, or any text with clear headings. It preserves the document's logical sections. |
| **Semantic** | Complex, narrative, or semi-structured documents | Best for research papers, articles, or any text where maintaining the flow of ideas is critical. It creates context-aware chunks that are better for RAG systems. |

## Technical Deep Dive

### Length-Based Chunking

This method is the most basic approach. It iterates through the text and splits it into chunks of a fixed size.

*   **How it works**: It can split by character count or by token count (using a tokenizer). The `chunk_overlap` parameter allows for a certain number of characters or tokens to be repeated at the beginning of the next chunk to maintain some context.
*   **Pros**: Simple, fast, and predictable.
*   **Cons**: Ignores sentence boundaries, logical structure, and semantic meaning. This can lead to chunks that are not coherent.

### Structure-Based Chunking

This method is designed for documents with a clear hierarchical structure, such as Markdown files with headers.

*   **How it works**: It splits the document based on its headers (`#`, `##`, `###`, etc.). Each section under a header is treated as a potential chunk. If a section is too large, it can be further subdivided using a length-based approach.
*   **Pros**: Preserves the logical structure of the document, creating chunks that are more coherent and contextually relevant.
*   **Cons**: Only effective for documents with a well-defined structure. It will perform poorly on unstructured text.

### Semantic Chunking

This is the most advanced method, using machine learning to split the text based on its meaning.

*   **How it works**: It uses a sentence embedding model to represent each sentence as a vector in a high-dimensional space. It then calculates the cosine similarity between adjacent sentences. A significant drop in similarity is considered a "breakpoint" or a shift in topic, and the text is split at that point.
*   **Pros**: Creates the most coherent and contextually relevant chunks, as it is based on the meaning of the text.
*   **Cons**: Computationally expensive and slower than the other methods. The quality of the chunks depends on the quality of the embedding model.


## Installation

1.  **Install Poetry:** If you don't have Poetry, follow the instructions on the [official website](https://python-poetry.org/docs/#installation).

2.  **Install Dependencies:** Navigate to the project root and run:
    ```bash
    poetry install
    ```

## Usage

The project is run via a command-line interface (CLI).

### Command

```bash
poetry run cli [SOURCE] [STRATEGY] [OPTIONS]
```

### Options

*   `source`: Path to the folder with markdown files.
*   `strategy`: Chunking strategy to use (`length_based`, `structure_based`, `semantic`).
*   `--config`: JSON string with the strategy configuration.
*   `--local`: Store chunks locally in the file system. If not specified, chunks are stored in ChromaDB.
*   `--clean`: Clean the chunk store before processing.
*   `--output-dir`: Directory to save the chunks (used with `--local`).
*   `--collection-name`: Name of the ChromaDB collection.

---

## Examples

Below are examples of how to run the chunking process with different strategies and configurations.

### 1. Chunking to the File System

**Use Case**: Generating chunk files for local use or inspection.

**Command**:
Splits documents into 1000-character chunks and saves them to `output_chunks/length_based_char`.

```bash
poetry run cli data length_based \
--config '{"chunk_size": 1000, "chunk_overlap": 200, "mode": "character"}' \
--local --output-dir 'output_chunks/length_based_char'
```

### 2. Storing Chunks in ChromaDB

**Use Case**: Indexing document chunks for retrieval in a RAG system.

**Command**:
Chunks documents using a semantic strategy and stores them in the `my_collection` ChromaDB collection.

```bash
poetry run cli data semantic \
--config '{"breakpoint_threshold_type": "percentile", "breakpoint_threshold_amount": 90.0}' \
--collection-name 'my_collection'
```

### 3. Cleaning the Chunk Store

**Use Case**: Resetting the chunk store (either local or ChromaDB).

**Example 1: Cleaning a ChromaDB Collection**
Deletes the `my_collection` collection from ChromaDB.

```bash
poetry run cli --clean --collection-name 'my_collection'
```

**Example 2: Cleaning a Local Directory**
Deletes the contents of the `output_chunks` directory.

```bash
poetry run cli --clean --local --output-dir 'output_chunks'
```


---

## Configuration Details

The `--config` option accepts a JSON string to customize the behavior of each strategy.

### `length_based`

| Name (Type) | Example Value | Description |
| :--- | :--- | :--- |
| `chunk_size` (int) | `1000` | Max size of each chunk (characters or tokens). |
| `chunk_overlap` (int) | `200` | Overlap between chunks. |
| `mode` (string) | `"character"` | Splitting mode: `"character"` or `"token"`. |

### `structure_based`

| Name (Type) | Example Value | Description |
| :--- | :--- | :--- |
| `chunk_size` (int) | `1000` | Max size of each chunk after splitting by headers. |
| `chunk_overlap` (int) | `200` | Overlap between sub-chunks. |
| `strip_headers` (bool) | `false` | Whether to remove header text from the chunk. |
| `max_header_levels` (int) | `4` | Max header level to consider (e.g., `4` = `####`). |

### `semantic`

| Name (Type) | Example Value | Description |
| :--- | :--- | :--- |
| `breakpoint_threshold_type` (string) | `"percentile"` | Method for determining breakpoints: `"percentile"`, `"standard_deviation"`, `"interquartile"`, `"absolute"`. |
| `breakpoint_threshold_amount` (float) | `95.0` | Value for the chosen threshold type. |
| `min_chunk_size` (int) | `1` | Minimum number of sentences per chunk. |
| `max_chunk_size` (int) | `null` | Maximum number of sentences per chunk (`null` for no limit). |

**Threshold Guide:**

| Threshold Type | Best For | Typical Value |
| :--- | :--- | :--- |
| `percentile` | General-purpose, adaptive | **95.0 (Conservative):** Fewer, larger chunks.<br>**90.0 (Balanced):** Good starting point.<br>**85.0 (Aggressive):** More, smaller chunks. |
| `standard_deviation` | Deviations from average similarity | **1.5:** Common value. Higher means fewer chunks. |
| `interquartile` | Robustness to outliers | **1.5:** Common value. Higher means fewer chunks. |
| `absolute` | Fixed similarity threshold | **0.8:** Requires experimentation. |

## Running Tests

To run the suite of unit tests, execute the following command from the project root:

```bash
poetry run pytest
```
