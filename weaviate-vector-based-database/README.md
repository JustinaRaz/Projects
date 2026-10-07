# Standard RAG (simple project) using Weaviate

The project is constrained to Python 3.12 version due to hardware compatability.

Notes so myself:
- The sentence boundary detection is currently working for EN language. Another idea could be to split text into sentences using SpaCy (for Lithuanian language, for example), or at least check if that splitting is better.