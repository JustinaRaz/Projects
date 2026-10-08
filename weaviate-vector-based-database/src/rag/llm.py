import json
import logging
from pathlib import Path

import torch
import yaml
from transformers import AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration

from src.core.settings import Settings, get_settings
from src.rag.context_formatting import ContextBuilder

logger = logging.getLogger("src")
llm_logger = logging.getLogger("src.llm")

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

class Gemma:
    """Gemma: stores relevant llm functions.
    """
    def __init__(self, settings: Settings | None = None):

        settings = settings or get_settings()
        prompts = load_config(settings.system_prompt_path)
        self.system_prompt = prompts["system"]
        self.user_template = prompts["user"]
        self.llm_config = load_config(settings.llm_config_path)
        self.model = None
        self.processor = None
        self.model_id = self.llm_config["model_name"]
        self.tokenizer = None
        self.context_builder = None

    def _load(self):

        if self.processor is not None and self.model is not None:
            return

        logger.info("Loading the model and its processor.")

        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.model is None:
            dtype = self.llm_config["torch_dtype"]
            if dtype != "auto":
                dtype = getattr(torch, dtype)

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=self.llm_config["device"],
            ).eval()

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def _format_params(self) -> dict:
        """Sampling kwargs for generate(); empty dict -> greedy decoding."""
        kwargs = {}
        if "temperature" in self.llm_config:
            kwargs["temperature"] = self.llm_config["temperature"]
        if "top_p" in self.llm_config:
            kwargs["top_p"] = self.llm_config["top_p"]
        return kwargs


    def _build_user_prompt(self, question: str, context: str) -> str:
        """Fills in the placeholders of the user prompt.

        Args:
            question (str): User's question.
            context (str): Retrieved relevant text chunks. Formatted.

        Returns:
            str: User prompt
        """

        logger.info("Building user's prompt.")

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

        logger.info("Constructing a chat for Gemma model.")

        chat = [
            {"role": "system", "content": [{"type": "text",
                                            "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "text",
                                          "text": user_prompt}]},
        ]

        llm_logger.debug(f"Constructed chat: {chat}")

        return chat

    def generate(self, question: str, context: dict) -> str:
        self._load()

        context_builder = ContextBuilder(
                        tokenizer=self.tokenizer,
                        max_context_tokens=self.llm_config["max_context_tokens"]
                    )
        context = context_builder.build_context(context)

        kwargs = self._format_params()
        do_sample = len(kwargs) > 0
        if not do_sample:
            logger.info("No sampling params given, setting do_sample=False")

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

        with torch.inference_mode():
            output = self.model.generate(
                **model_inputs,
                max_new_tokens=self.llm_config["max_new_tokens"],
                do_sample=do_sample,
                **kwargs,
            )

        raw = self.processor.decode(output[0][input_len:], skip_special_tokens=True)
        llm_logger.debug("LLM RAW OUTPUT:\n%s", raw)
        try:
            response = json.loads(raw)
            answer = response["answer"]
        except (json.JSONDecodeError, KeyError):
            logger.warning("Model returned invalid JSON, using raw text")
            answer = raw

        llm_logger.debug("LLM OUTPUT:\n%s", answer)

        return answer
