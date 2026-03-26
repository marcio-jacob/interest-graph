"""
llm/hf_client.py
================
HuggingFace Transformers client for local text generation.

Loads a causal LM with 4-bit BitsAndBytes quantization to fit in 6 GB VRAM.
Exposes the same is_available() / generate() interface as OllamaClient so the
rest of the pipeline needs no changes.

Usage:
    client = HuggingFaceClient(cfg)
    if client.is_available():
        text = client.generate("Write a short TikTok description about cooking.")
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class HuggingFaceClient:
    """
    Thin wrapper around a HuggingFace causal LM pipeline.

    Parameters
    ----------
    cfg : dict
        Merged config dict.  Reads cfg["llm"]["model"] and cfg["llm"] settings.
    model_name : str | None
        Override model name (falls back to cfg["llm"]["model"]).
    """

    def __init__(self, cfg: dict, model_name: str | None = None) -> None:
        llm_cfg = cfg.get("llm", {})
        self.model_name = (
            model_name
            or os.environ.get("HF_MODEL")
            or llm_cfg.get("model", "Qwen/Qwen2.5-7B")
        )
        self.max_new_tokens = int(llm_cfg.get("max_new_tokens", 180))
        self.temperature = float(llm_cfg.get("temperature", 0.9))
        self._pipe = None
        self._available: bool | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )

            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_cfg,
                device_map={"": 0},  # force all layers to GPU 0
                dtype=torch.float16,
            )

            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
            )
            logger.info("HuggingFaceClient loaded: %s", self.model_name)
            return True
        except Exception as exc:
            logger.warning("HuggingFaceClient failed to load %s: %s", self.model_name, exc)
            return False

    # ------------------------------------------------------------------
    # Public API (mirrors OllamaClient)
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._load()
        return self._available

    def _post_process(self, text: str, stop: list[str] | str | None) -> str:
        text = text.strip()
        if stop:
            stops = [stop] if isinstance(stop, str) else stop
            for s in stops:
                if s in text:
                    text = text[: text.index(s)]
        return text

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | str | None = None,
        **_kwargs,
    ) -> str:
        """
        Generate text continuation for a single prompt.

        Returns an empty string on failure so the caller can fall back gracefully.
        """
        results = self.generate_batch([prompt], max_tokens=max_tokens,
                                      temperature=temperature, stop=stop)
        return results[0] if results else ""

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stop: list[str] | str | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """
        Generate text for a list of prompts in batches.

        Returns a list of strings (one per prompt); empty string on error.
        """
        if not self.is_available() or not prompts:
            return [""] * len(prompts)

        max_new = max_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        try:
            self._pipe.tokenizer.padding_side = "left"
            if self._pipe.tokenizer.pad_token_id is None:
                self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id

            outputs = self._pipe(
                prompts,
                max_new_tokens=max_new,
                temperature=temp,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.15,
                return_full_text=False,
                batch_size=batch_size,
            )
            return [
                self._post_process(out[0]["generated_text"], stop)
                for out in outputs
            ]
        except Exception as exc:
            logger.warning("HuggingFaceClient.generate_batch error: %s", exc)
            return [""] * len(prompts)
