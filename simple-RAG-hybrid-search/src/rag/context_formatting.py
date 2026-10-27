import logging

from src.rag.models import RetrievedChunk

llm_logger = logging.getLogger("src.llm")

class ContextBuilder:
    """Selects retrieved chunks so that the whole prompt plus the answer
    fits into max_context_tokens.
    """
    def __init__(
        self,
        tokenizer,
        render_prompt,
        max_context_tokens: int,
        max_new_tokens: int,
    ):
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens

        self.fixed_tokens = self._count_tokens(render_prompt("", ""))
        self.separator_tokens = self._count_tokens("\n\n")

    def _count_tokens(self, text: str) -> int:

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _format_chunk(self, chunk: RetrievedChunk) -> str:
        return (
            f"[File name: {chunk.filename}, chunk ID: {chunk.chunk_id}]\n"
            f"{chunk.text}"
        )

    def _format_retrieved_chunks(
        self,
        retrieved_chunks,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id=str(obj.uuid),
                filename=obj.properties["source_filename"],
                text=obj.properties["content"],
            )
            for obj in retrieved_chunks.objects
        ]

    def build_context(self, question: str, retrieved_chunks) -> str:

        question_tokens = self._count_tokens(question)
        budget = (self.max_context_tokens
                  - self.fixed_tokens - question_tokens - self.max_new_tokens
        )

        if budget <= 0:
            llm_logger.warning(
                "No budget for context (fixed=%d, question=%d, answer=%d, limit=%d)",
                self.fixed_tokens, question_tokens,
                self.max_new_tokens, self.max_context_tokens,
            )
            return "", []

        chunks = self._format_retrieved_chunks(retrieved_chunks)
        parts: list[str] = []
        texts: list[str] = [] # Stores text for Ragas
        tokens_used = 0

        for chunk in chunks:
            text = self._format_chunk(chunk)
            token_num = self._count_tokens(text)
            if parts:
                token_num += self.separator_tokens

            if tokens_used + token_num > budget:
                llm_logger.warning("Some text was discarded due to context limits.")
                break

            parts.append(text)
            texts.append(chunk.text)
            tokens_used += token_num

        return "\n\n".join(parts), texts



