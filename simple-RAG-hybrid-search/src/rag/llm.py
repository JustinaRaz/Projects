import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import torch
import yaml
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from src.core.settings import get_settings
from src.llm.llm_config import load_llm_config
from src.rag.context_formatting import ContextBuilder

llm_logger = logging.getLogger("src.llm")

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_write_lock = Lock()

def save_rag_record(path: Path,
                    question: str,
                    contexts: list[str],
                    answer: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
    }
    with _write_lock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class Gemma:
    """Gemma: stores relevant llm functions.
    """
    def __init__(self):

        self.settings = get_settings()
        prompts = load_config(self.settings.system_prompt_path)
        self.system_prompt = prompts["system"]
        self.user_template = prompts["user"]

        self.config = load_llm_config()

        self.model = None
        self.processor = None
        self.context_builder = None

    def _load(self):

        if self.model is not None and self.processor is not None and self.context_builder is not None:
            return

        llm_logger.info("Loading the model's processor.")

        model_settings = self.config.model
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(
                model_settings.model_id,
                cache_dir = model_settings.cache_dir,
                use_fast=False)

        llm_logger.info(f"Loading {model_settings.model_id}.")

        if self.model is None:
            dtype = model_settings.torch_dtype
            if dtype != "auto":
                dtype = getattr(torch, dtype)

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                model_settings.model_id,
                dtype=dtype,
                device_map=model_settings.device_map,
                cache_dir=model_settings.cache_dir
            ).eval()

        llm_logger.info("Loading the LLM's context builder.")

        if self.context_builder is None:
            llm_logger.info("Creating the context builder.")
            self.context_builder = ContextBuilder(
                tokenizer=self.processor.tokenizer,
                render_prompt=self._render_prompt,
                max_context_tokens=self.config.context.max_context_tokens,
                max_new_tokens=self.config.generation.max_new_tokens,
            )

    def _build_user_prompt(self, question: str, context: str) -> str:
        """Fills in the placeholders of the user prompt.

        Args:
            question (str): User's question.
            context (str): Retrieved relevant text chunks. Formatted.

        Returns:
            str: User prompt
        """

        llm_logger.info("Building user's prompt.")

        return (
            self.user_template
            .replace("{question}", question)
            .replace("{context}", context)
        )

    def _format_chat(self, user_prompt: str) -> list[dict]:
        """Creates chat messages in the format valid for Gemma.

        Args:
            user_prompt (str): User prompt/text

        Returns:
            list[dict]: Formatted chat entry
        """

        llm_logger.info("Constructing a chat structure for Gemma model.")

        chat = [
            {"role": "system", "content": [{"type": "text",
                                            "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text",
                                          "text": user_prompt}]},
        ]

        return chat

    def _render_prompt(self, question: str, context: str) -> str:
        """Prompt text exactly as the model sees it (chat template applied)."""
        chat = self._format_chat(self._build_user_prompt(question, context))

        return self.processor.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

    def generate(self, question: str, chunks: dict) -> str:

        # Loads the LLM
        self._load()

        context, context_chunks = self.context_builder.build_context(question, chunks)

        chat = self._format_chat(self._build_user_prompt(question, context))

        model_inputs = self.processor.apply_chat_template(
            chat,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        model_inputs.pop("token_type_ids", None)

        llm_logger.debug(f"Model inputs: {model_inputs}")

        input_len = model_inputs["input_ids"].shape[-1]

        # Loads parameters for LLM generation
        llm_logger.info("Loading parameters for setting LLM generation.")

        gen_settings = self.config.generation
        params = {
            "max_new_tokens": gen_settings.max_new_tokens,
            "do_sample": gen_settings.do_sample,
        }

        if gen_settings.do_sample:
            params["temperature"] = gen_settings.temperature
            params["top_p"] = gen_settings.top_p

        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                **params,
            )

        raw = self.processor.decode(output[0][input_len:], skip_special_tokens=True)

        save_rag_record(path = self.settings.rag_record_upload_dir,
                        question = question,
                        contexts = context_chunks,
                        answer = raw)
        llm_logger.info(f"RAG record saved at {self.settings.rag_record_upload_dir}")

        return raw
