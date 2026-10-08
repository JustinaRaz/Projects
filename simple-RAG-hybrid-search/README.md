# Simple RAG with hybrid search

Note: The project is constrained to Python 3.12 version due incompatability with the hardware (Mac Intel issues with PyTorch, mainly). 

## Project overview

The project relies on:
- Weaviate database
- Gemma3 4B IT LLM model
- all-MiniLM-L6-v2 text embedding model for computing dense vectors for English language
- Swagger UI

## Known drawbacks of this project

- Gemma3 4B model is small and does not strictly follow the instructions, which makes this model harder to prompt.
- The current setup works well enough (considering the hardware) only for English language. To include other languages, the follwing should be revised: 
    1) Sentence-based splitting before chunking takes place (maybe with SpaCy). 
    2) Switch to another embedding model - this one works for English language. 
    3) Use another vector storage database - to my knowledge, it stores only dense vectors, and provides the hybrid search functionality itself by computing BM25 scores as an additional layer.

## Run the project

To explore the project, clone the repository, get the virtual environment and from the project root run:

```
just dev
```

This will run the Swagger UI. Then, in your browser, go to:
```
http://127.0.0.1:8000 /docs
```

## Structure of the project

TBA: refactoring is required.