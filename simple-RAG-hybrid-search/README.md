
# Simple RAG with Hybrid Search

> **Note:** The project currently uses Python 3.12 due to compatibility constraints in the development environment, including PyTorch support on Intel-based Macs.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration and Data Formats](#3-configuration-and-data-formats)
4. [Running the Project](#4-running-the-project)
5. [RAG Pipeline](#5-rag-pipeline)
6. [Known Limitations](#6-known-limitations)

## 1. Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Weaviate for hybrid search and Gemma 3 4B IT for answer generation. Documents are parsed, divided into chunks, converted into embeddings, and stored in a Weaviate collection. When a user submits a question, the system retrieves relevant chunks and provides them as context to the LLM.

| Component | Technology | Purpose |
|---|---|---|
| Vector database | Weaviate | Stores document chunks, metadata, and vectors; supports hybrid search. |
| LLM | Gemma 3 4B IT | Generates answers based on the retrieved context. |
| Embedding model | `all-MiniLM-L6-v2` | Converts text into dense vector embeddings; primarily suited to English text. |
| Document processing | Docling | Extracts document content and structure. |
| Sentence segmentation | PySBD | Detects sentence boundaries for English text before further chunking. |
| API framework | FastAPI | Exposes document ingestion, collection management, and retrieval endpoints. |
| API documentation | Swagger UI | Provides an interactive interface for exploring and testing API endpoints. |
| Development tools | `uv`, `just` | Manages the Python environment and provides project commands. |

## 2. Project Structure

| Folder | Purpose |
|---|---|
| `config/` | YAML configuration files for application settings, LLM parameters, prompts, and logging. |
| `db/` | Local storage for extracted text, processed chunks, logs, and RAG records used for inspection and debugging. |
| `routers/` | FastAPI route definitions for exposing functionality through the API. |
| `src/` | Core application logic for document processing, embedding generation, retrieval, and database integration. |
| `src/core/` | Application settings, Weaviate client, and logging configuration. |
| `src/llm/` | LLM configuration and generation-related functionality. |
| `src/processing/` | Document parsing, text processing, and chunking. |
| `src/rag/` | Embeddings, context construction, and answer generation. |

## 3. Configuration and Data Formats

| Category | Formats | Purpose |
|---|---|---|
| Configuration | YAML | Defines application settings, LLM parameters, prompts, and logging configuration. |
| Structured data | JSON, JSONL, CSV | Stores processed data and records used for inspection and evaluation, where applicable. |
| Logs | Log files | Records application events and errors. |
| Documents | PDF, Markdown | Provides input documents and locally exported text for inspection. |

## 4. Running the Project

### Prerequisites

Clone the repository, install the project dependencies, and set up the environment according to the project's configuration.

### Start the application

From the project root, run:

```bash
just dev
```

This starts the FastAPI development server.

Open the following URL in your browser to access Swagger UI and test the available endpoints:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 5. RAG Pipeline

The following sections describe how documents move through the system, from ingestion to answer generation.

### 5.1. Document Parsing

**Module:** `src/processing/preprocessor.py`

Docling processes the uploaded file and converts it into a structured `DoclingDocument` containing extracted text, tables, and layout information.

- **Document conversion:** The uploaded file bytes are wrapped in a `DocumentStream` and converted into a structured document.
- **Markdown export:** The extracted content is exported to Markdown and saved locally in `db/text/` for inspection and debugging.
- **Passing the document forward:** The structured `DoclingDocument` is returned to the indexing pipeline for chunking. The exported Markdown file is not used as the chunker's input.

### 5.2. Document Chunking

**Module:** `src/processing/chunking.py`

The structured document is divided into smaller text chunks suitable for embedding and retrieval.

1. **Document segmentation:** Docling's `HybridChunker` extracts structured sections from the document.
2. **Token counting:** Sections are measured using the configured embedding model's tokenizer to help respect its input token limit.
3. **Sentence splitting:** Sections exceeding the configured token limit are split into sentences using PySBD.
4. **Chunk creation:** Sentences are grouped into chunks within the configured token limit. Where possible, sentences overlap between consecutive chunks to preserve context.
5. **Fallback splitting:** Sentences that exceed the token limit are split into token-based windows with overlap.
6. **Metadata assignment:** Each chunk receives an ID, a sequential index, and source metadata, including the filename and parent section index.
7. **Local storage:** Processed chunks, metadata, and token counts are saved as JSON for inspection and debugging.

### 5.3. Generating Embeddings

**Module:** `src/rag/embedder.py`

The chunk text is converted into dense numerical vectors using the configured SentenceTransformer embedding model.

1. **Text extraction:** The text is extracted from each chunk and collected into a list.
2. **Model loading:** The embedding model is loaded through `get_embedder()`. The `lru_cache` decorator allows the same embedder instance to be reused within the process.
3. **Vector generation:** The model's `encode()` method generates one embedding for each input chunk.
4. **Output conversion:** The resulting NumPy array is converted into a list of Python lists, with one vector corresponding to each chunk.

### 5.4. Storing Chunks in Weaviate

**Module:** `src/core/weaviate_client.py`

The chunk text, metadata, and generated vectors are stored in the target Weaviate collection.

1. **Preparing properties:** Each chunk is represented by a dictionary containing its text under the `content` property and its associated metadata.
2. **Input validation:** The number of property dictionaries is checked against the number of vectors to ensure that each chunk has a corresponding embedding.
3. **Object insertion:** Each chunk's properties and vector are inserted into Weaviate as an individual object.
4. **Indexing summary:** The number of stored chunks is returned along with the filename and collection name.

### 5.5. Retrieving Relevant Chunks

**Module:** `src/core/weaviate_client.py`

When a user submits a question, the application searches the indexed chunks to find relevant information.

1. **Query embedding:** The user's question is converted into a vector using the same embedding model used during document indexing.
2. **Hybrid search:** Weaviate combines vector similarity search with BM25 keyword search. The `alpha` parameter controls the balance between the two methods.
3. **Result retrieval:** The search returns matching Weaviate objects containing the stored chunk text and metadata.

### 5.6. Building the LLM Context

**Module:** `src/rag/context_formatting.py`

The retrieved chunks are selected and formatted to fit within the configured context budget for Gemma.

1. **Chunk formatting:** Each retrieved object is converted into a `RetrievedChunk` containing its Weaviate UUID, source filename, and text.
2. **Token budget calculation:** Gemma's tokenizer is used to estimate the available context space after accounting for the prompt, question, and reserved answer tokens.
3. **Context selection:** Retrieved chunks are processed in search-result order and included until the next chunk would exceed the available budget.
4. **Context formatting:** The selected chunks are combined into a single context string, with filename and chunk ID headers.
5. **Evaluation records:** The selected chunk texts are retained separately for logging and potential RAG evaluation.

### 5.7. Generating the Answer

**Module:** `src/rag/llm.py`

Gemma 3 generates an answer using the user's question and the selected context.

1. **Model loading:** The Gemma model and processor are loaded when needed.
2. **Prompt construction:** The system prompt, user question, and retrieved context are combined using the configured prompt templates.
3. **Chat formatting and tokenization:** The prompt is formatted using Gemma's chat template and converted into model inputs.
4. **Response generation:** The model generates a response according to the configured generation parameters, including the maximum number of new tokens and sampling settings.
5. **Response decoding:** The generated tokens are decoded into text, excluding special tokens.
6. **RAG record storage:** The question, selected context chunks, generated answer, and timestamp are saved as a JSONL record for inspection and evaluation.

## 6. Known Limitations

- **LLM instruction following:** Gemma 3 4B IT is a relatively small model and may not consistently follow the instructions provided in the prompt.
- **Language support:** The current setup is primarily configured for English. Supporting additional languages would require reviewing the preprocessing and retrieval components.
  - **Sentence segmentation:** The sentence-splitting step currently uses PySBD for English text. It should be adapted or replaced with a segmenter suitable for the target language.
  - **Embedding model:** `all-MiniLM-L6-v2` is primarily intended for English. A multilingual embedding model would be more suitable for retrieving documents in other languages.
  - **Retrieval configuration:** The performance of both vector search and BM25 keyword search should be evaluated for the target language. Weaviate already supports hybrid search, so changing the database is not necessarily required.
- **Hardware constraints:** Running the local LLM and embedding model is limited by the available hardware and memory. Performance may vary depending on the model and workload.
