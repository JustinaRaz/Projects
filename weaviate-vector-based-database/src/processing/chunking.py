import json
import logging
import uuid
from functools import lru_cache

import pysbd
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument

from routers.schemas.document import Chunk
from src.core.settings import get_settings
from src.processing.tokenizer import get_tokenizer

logger = logging.getLogger("src")

settings = get_settings()


class Chunker:
    def __init__(self):
        self.tokenizer = get_tokenizer()

        self.max_tokens = settings.MAX_TOKENS_PER_CHUNK
        self.overlap_tokens = settings.OVERLAP_TOKENS

        self.segmenter = pysbd.Segmenter(
            language=settings.sentence_language,
            clean=False,
        )

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in text."""

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def _sentence_split(self, text: str) -> list[str]:
        """Split text into sentences using PySBD."""

        sentences = self.segmenter.segment(text)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def _hard_split_by_tokens(self, text: str) -> list[str]:
        """Split an oversized sentence directly by tokens."""

        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        step = self.max_tokens - self.overlap_tokens

        windows = []

        start = 0

        while start < len(token_ids):
            end = min(
                start + self.max_tokens,
                len(token_ids),
            )

            chunk_token_ids = token_ids[start:end]

            chunk_text = self.tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True,
            )

            windows.append(chunk_text)

            if end >= len(token_ids):
                break

            start += step

        return windows

    def _split_with_overlap(self, text: str) -> list[str]:
        """Split oversized text into overlapping, sentence-aligned chunks."""

        sentences = self._sentence_split(text)

        if not sentences:
            return []

        sentence_tokens = [
            self._count_tokens(sentence)
            for sentence in sentences
        ]

        windows: list[str] = []

        current_sentences: list[str] = []
        current_tokens = 0

        sentence_index = 0

        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            sentence_token_count = sentence_tokens[sentence_index]

            # A single sentence is larger than the entire chunk limit.
            if sentence_token_count > self.max_tokens:

                # Save the current window first.
                if current_sentences:
                    windows.append(
                        " ".join(current_sentences)
                    )

                    current_sentences = []
                    current_tokens = 0

                # Nothing can preserve the sentence boundary,
                # so fall back to token-level splitting.
                windows.extend(
                    self._hard_split_by_tokens(sentence)
                )

                sentence_index += 1
                continue

            # Sentence fits into the current window.
            if (
                current_tokens + sentence_token_count
                <= self.max_tokens
            ):
                current_sentences.append(sentence)
                current_tokens += sentence_token_count
                sentence_index += 1
                continue

            # Current window is full.
            windows.append(
                " ".join(current_sentences)
            )

            # Build overlap from the end of the previous window.
            overlap_sentences: list[str] = []
            overlap_token_count = 0

            for sentence_text, token_count in zip(
                reversed(current_sentences),
                reversed(
                    [
                        self._count_tokens(sentence)
                        for sentence in current_sentences
                    ]
                ), strict = True
            ):
                if (
                    overlap_token_count + token_count
                    > self.overlap_tokens
                ):
                    break

                overlap_sentences.insert(
                    0,
                    sentence_text,
                )

                overlap_token_count += token_count

            current_sentences = overlap_sentences
            current_tokens = overlap_token_count

        if current_sentences:
            windows.append(
                " ".join(current_sentences)
            )

        return windows

    def _save_chunks_to_json(
        self,
        chunks: list[Chunk],
        filename: str,
    ) -> None:
        """Save chunks and metadata to a JSON file for inspection."""

        data = [
            {
                "chunk_index": chunk.chunk_index,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "token_count": self._count_tokens(chunk.text),
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]

        output_path = (
            settings.doc_upload_dir
            / f"{filename}_chunks.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def chunk_document(
        self,
        document: DoclingDocument,
        *,
        source_filename: str,
    ) -> list[Chunk]:
        """Split a DoclingDocument into embedding-ready chunks."""

        chunker = HybridChunker(
            tokenizer=self.tokenizer,
        )

        raw_sections = list(
            chunker.chunk(document)
        )

        chunks: list[Chunk] = []

        for section_index, section in enumerate(raw_sections):

            section_text = chunker.contextualize(
                section
            )

            section_tokens = self._count_tokens(
                section_text
            )

            # Normal case: the Docling chunk already fits.
            if section_tokens <= self.max_tokens:

                chunks.append(
                    Chunk(
                        text=section_text,
                        chunk_index=len(chunks),
                        chunk_id = str(uuid.uuid4()),
                        metadata={
                            "source_filename": source_filename,
                            "parent_section_index": section_index,
                        },
                    )
                )

                continue

            # Oversized Docling chunk:
            # split it using sentence boundaries + overlap.
            sub_texts = self._split_with_overlap(
                section_text
            )

            for sub_index, sub_text in enumerate(
                sub_texts
            ):
                chunks.append(
                    Chunk(
                        text=sub_text,
                        chunk_index=len(chunks),
                        chunk_id = str(uuid.uuid4()),
                        metadata={
                            "source_filename": source_filename,
                            "parent_section_index": section_index,
                            "sub_chunk_index": sub_index,
                        },
                    )
                )

        logger.debug(
            "Chunked '%s' into %d chunks",
            source_filename,
            len(chunks),
        )

        self._save_chunks_to_json(
            chunks,
            source_filename,
        )

        return chunks

@lru_cache
def get_chunker() -> Chunker:
    return Chunker()
