"""Lifecycle-managed local Qwen backend with deterministic batched generation."""

from __future__ import annotations

import gc
import re
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import LocalModelConfig

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_block(text: str) -> str:
    """Qwen3 đôi khi vẫn chèn <think>...</think> dù enable_thinking=False
    (tuỳ phiên bản template) — strip phòng hờ trước khi đưa cho json_guard,
    không tin tuyệt đối cờ config."""
    return _THINK_BLOCK_RE.sub("", text).strip()


class LocalLLM:
    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_stats: list[dict[str, Any]] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = {
            "auto": "auto",
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(self.config.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported LLM dtype: {self.config.dtype}")
        model_kwargs = {
            "revision": self.config.revision,
            "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": dtype,
        }
        if self.config.attention_implementation:
            model_kwargs["attn_implementation"] = self.config.attention_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, **model_kwargs,
            )
        except (ImportError, ValueError) as exc:
            if model_kwargs.pop("attn_implementation", None) is None:
                raise
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, **model_kwargs,
            )
        self.model.eval()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _build_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        template_kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        # CHỈ truyền enable_thinking nếu model hỗ trợ — Qwen2.5-Instruct
        # không có tham số này trong chat template, truyền vào sẽ lỗi.
        if self.config.supports_thinking:
            template_kwargs["enable_thinking"] = self.config.enable_thinking
        return self.tokenizer.apply_chat_template(messages, **template_kwargs)

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Trả raw text đã decode (chưa parse JSON — dùng llm.json_guard
        ở tầng gọi để parse, không parse ở đây vì 2 task có schema khác nhau)."""
        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model chưa load được")

        prompt_text = self._build_prompt_text(system_prompt, user_prompt)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.config.max_context_length,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            top_p=None,
            top_k=None,
        )
        generated = outputs[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        if self.config.supports_thinking:
            text = _strip_thinking_block(text)

        return text.strip()

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
        *,
        batch_size: int = 4,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Generate multiple short JSON responses with left-padded batches."""
        if batch_size <= 0:
            raise ValueError("batch_size phải dương")
        if not prompts:
            return []
        self.load()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model chưa load được")

        responses = []
        for start in range(0, len(prompts), batch_size):
            group = prompts[start:start + batch_size]
            prompt_texts = [
                self._build_prompt_text(system_prompt, user_prompt)
                for system_prompt, user_prompt in group
            ]
            inputs = self.tokenizer(
                prompt_texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.config.max_context_length,
            ).to(self.model.device)
            started = time.perf_counter()
            token_budget = int(max_new_tokens or self.config.max_new_tokens)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                top_p=None,
                top_k=None,
            )
            prompt_width = inputs["input_ids"].shape[1]
            for group_index, row in enumerate(outputs):
                continuation = row[prompt_width:]
                text = self.tokenizer.decode(continuation, skip_special_tokens=True)
                if self.config.supports_thinking:
                    text = _strip_thinking_block(text)
                responses.append(text.strip())
                output_tokens = int(continuation.shape[0])
                self.generation_stats.append({
                    "input_tokens": int(inputs["attention_mask"][group_index].sum().item()),
                    "output_tokens": output_tokens,
                    "token_budget": token_budget,
                    "finish_reason": "length" if output_tokens >= token_budget else "eos",
                    "latency_seconds": time.perf_counter() - started,
                })
        return responses
