import logging

from src.rag.models import RetrievedChunk

logger = logging.getLogger("src.llm")

class ContextBuilder:
    def __init__(self, tokenizer, max_context_tokens: int):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def _format_retrieved_chunks(self, response) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=str(obj.uuid),
                filename=obj.properties["source_filename"],
                text=obj.properties["content"],
            )
            for obj in response.objects
        ]

    def build_context(self,
        response: dict
    ) -> str:
        context_parts = []
        current_tokens = 0

        chunks = self._format_retrieved_chunks(response)

        for chunk in chunks:
            formatted_chunk = (
                f"[File name: {chunk.filename}, chunk ID: {chunk.chunk_id}]\n"
                f"{chunk.text}"
            )

            chunk_tokens = len(self.tokenizer.encode(formatted_chunk))

            if current_tokens + chunk_tokens > self.max_context_tokens:
                break

            context_parts.append(formatted_chunk)
            current_tokens += chunk_tokens

        context = "\n\n".join(context_parts)
        logger.debug(f"Context built: {context}")

        return context



