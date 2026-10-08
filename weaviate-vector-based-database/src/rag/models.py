from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    filename: str
    text: str
