"""Lifecycle-managed Qwen backend with token-aware incremental microbatches."""

from __future__ import annotations

import gc
import math
import re
import time
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .config import LocalModelConfig

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_block(text: str) -> str:
    return _THINK_BLOCK_RE.sub("", text).strip()


class LocalLLM:
    def __init__(self, config: LocalModelConfig) -> None:
        self.config = config
        self.model = None
        self.tokenizer = None
        self.generation_stats: list[dict[str, Any]] = []
        self.batch_generation_stats: list[dict[str, Any]] = []
        self.load_stats: dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        started = time.perf_counter()
        quantization_config = None
        if self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, revision=self.config.revision,
            cache_dir=self.config.cache_dir, local_files_only=self.config.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = {"auto": "auto", "bfloat16": torch.bfloat16, "float16": torch.float16}.get(self.config.dtype)
        if dtype is None:
            raise ValueError(f"Unsupported LLM dtype: {self.config.dtype}")
        if self.config.device_map_mode not in {"single_gpu", "auto"}:
            raise ValueError("device_map_mode must be single_gpu or auto")
        device_map: Any = "auto"
        if self.config.device_map_mode == "single_gpu" and torch.cuda.is_available():
            device_map = {"": 0}
        model_kwargs = {
            "revision": self.config.revision, "cache_dir": self.config.cache_dir,
            "local_files_only": self.config.local_files_only,
            "quantization_config": quantization_config, "device_map": device_map,
            "dtype": dtype, "use_safetensors": True,
        }
        if self.config.attention_implementation:
            model_kwargs["attn_implementation"] = self.config.attention_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
        except TypeError:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
        except (ImportError, ValueError):
            if model_kwargs.pop("attn_implementation", None) is None:
                raise
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **model_kwargs)
        self.model.eval()
        device_map_value = getattr(self.model, "hf_device_map", None)
        devices = {str(value) for value in (device_map_value or {}).values()}
        offloaded = any(value == "cpu" or value == "disk" for value in devices)
        primary_device = str(next(self.model.parameters()).device)
        self.load_stats = {
            "load_seconds": time.perf_counter() - started,
            "hf_device_map": device_map_value, "primary_device": primary_device,
            "cpu_or_disk_offload": offloaded,
        }
        print(
            f"[Qwen] loaded in {self.load_stats['load_seconds']:.2f}s "
            f"device={primary_device} device_map={device_map_value}", flush=True,
        )
        not_full_gpu = offloaded or not primary_device.startswith("cuda")
        if not_full_gpu:
            message = f"Qwen is not fully on CUDA (primary={primary_device}, map={device_map_value})"
            if self.config.require_full_gpu:
                raise RuntimeError(message)
            print(f"[Qwen] WARNING: {message}", flush=True)

    def unload(self) -> None:
        self.model = None; self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_generation_stats(self) -> None:
        self.generation_stats.clear(); self.batch_generation_stats.clear()

    def _build_prompt_text(self, system_prompt: str, user_prompt: str) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.config.supports_thinking:
            kwargs["enable_thinking"] = self.config.enable_thinking
        return self.tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], **kwargs)

    def count_prompt_tokens(
        self, prompts: list[tuple[str, str]], *, enforce_limit: bool = True,
    ) -> list[int]:
        self.load()
        texts = [self._build_prompt_text(*prompt) for prompt in prompts]
        encoded = self.tokenizer(
            texts, add_special_tokens=False, padding=False, truncation=False,
        )
        masks = encoded.get("attention_mask")
        if masks is not None:
            lengths = [int(sum(row)) for row in masks]
        else:
            lengths = [len(row) for row in encoded["input_ids"]]
        oversized = [value for value in lengths if value > self.config.max_context_length]
        if oversized and enforce_limit:
            raise ValueError(
                f"prompt exceeds max_context_length={self.config.max_context_length}: max={max(oversized)}"
            )
        return lengths

    @staticmethod
    def _bucket_limit(length: int) -> int:
        for limit in (512, 768, 1024, 1536):
            if length <= limit:
                return limit
        return 10**9

    def _pack_batches(
        self, lengths: list[int], *, batch_size: int, output_budget: int,
        max_batch_tokens: int | None, dynamic_batching: bool,
    ) -> list[list[int]]:
        order = list(range(len(lengths)))
        if dynamic_batching:
            order.sort(key=lambda index: (self._bucket_limit(lengths[index]), lengths[index], index))
        groups: list[list[int]] = []
        current: list[int] = []
        for index in order:
            proposed = [*current, index]
            width = max(lengths[value] for value in proposed)
            estimated = (width + output_budget) * len(proposed)
            if current and (len(proposed) > batch_size or (max_batch_tokens and estimated > max_batch_tokens)):
                groups.append(current); current = [index]
            else:
                current = proposed
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _trim_continuation(row, *, eos_token_id: int | None, pad_token_id: int | None):
        values = row.tolist()
        end = len(values)
        for index, value in enumerate(values):
            if eos_token_id is not None and value == eos_token_id:
                end = index + 1; break
            if pad_token_id is not None and value == pad_token_id:
                end = index; break
        return row[:end]

    @torch.inference_mode()
    def _run_microbatch(self, prompt_texts: list[str], *, max_new_tokens: int) -> tuple[list[str], dict]:
        inputs = self.tokenizer(
            prompt_texts, return_tensors="pt", add_special_tokens=False,
            padding=True, truncation=True, max_length=self.config.max_context_length,
        ).to(self.model.device)
        input_lengths = [int(row.sum().item()) for row in inputs["attention_mask"]]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        outputs = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )
        latency = time.perf_counter() - started
        prompt_width = int(inputs["input_ids"].shape[1])
        responses, output_lengths, finish_reasons = [], [], []
        for row in outputs:
            continuation = self._trim_continuation(
                row[prompt_width:], eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            )
            text = self.tokenizer.decode(continuation, skip_special_tokens=True)
            if self.config.supports_thinking:
                text = _strip_thinking_block(text)
            responses.append(text.strip())
            output_lengths.append(int(continuation.shape[0]))
            finish_reasons.append("length" if len(continuation) >= max_new_tokens else "eos")
        real_output = sum(output_lengths)
        stats = {
            "request_count": len(prompt_texts), "padded_input_width": prompt_width,
            "real_input_tokens": sum(input_lengths), "min_input_tokens": min(input_lengths),
            "max_input_tokens": max(input_lengths), "real_output_tokens": real_output,
            "output_tokens_by_row": output_lengths, "finish_reasons": finish_reasons,
            "latency_seconds": latency,
            "decode_tokens_per_second": real_output / latency if latency else 0.0,
            "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        self.batch_generation_stats.append(stats)
        batch_id = len(self.batch_generation_stats)
        for input_count, output_count, finish in zip(input_lengths, output_lengths, finish_reasons):
            self.generation_stats.append({
                "batch_id": batch_id, "input_tokens": input_count,
                "output_tokens": output_count, "token_budget": max_new_tokens,
                "finish_reason": finish, "batch_latency_seconds": latency,
            })
        return responses, stats

    def generate_batches(
        self, prompts: list[tuple[str, str]], *, batch_size: int = 4,
        max_new_tokens: int | None = None, max_batch_tokens: int | None = None,
        min_batch_size: int = 1, dynamic_batching: bool = True,
        prompt_lengths: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        if batch_size <= 0 or min_batch_size <= 0 or min_batch_size > batch_size:
            raise ValueError("invalid batch size bounds")
        if not prompts:
            return
        self.load()
        prompt_texts = [self._build_prompt_text(*prompt) for prompt in prompts]
        lengths = list(prompt_lengths) if prompt_lengths is not None else self.count_prompt_tokens(prompts)
        if len(lengths) != len(prompts):
            raise ValueError("prompt_lengths must align with prompts")
        budget = int(max_new_tokens or self.config.max_new_tokens)
        groups = self._pack_batches(
            lengths, batch_size=batch_size, output_budget=budget,
            max_batch_tokens=max_batch_tokens or self.config.max_batch_tokens,
            dynamic_batching=dynamic_batching,
        )
        queue = list(groups); completed = 0
        while queue:
            indexes = queue.pop(0)
            try:
                responses, stats = self._run_microbatch(
                    [prompt_texts[index] for index in indexes], max_new_tokens=budget,
                )
            except torch.cuda.OutOfMemoryError:
                if len(indexes) <= min_batch_size:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                split = max(min_batch_size, len(indexes) // 2)
                queue = [indexes[:split], indexes[split:], *queue]
                continue
            completed += 1
            yield {
                "indexes": indexes, "responses": responses, "stats": stats,
                "batch_total": completed + len(queue),
            }

    @torch.inference_mode()
    def generate_batch(
        self, prompts: list[tuple[str, str]], *, batch_size: int = 4,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        results: list[str | None] = [None] * len(prompts)
        for batch in self.generate_batches(
            prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
            max_batch_tokens=self.config.max_batch_tokens,
            min_batch_size=self.config.min_batch_size,
            dynamic_batching=self.config.dynamic_batching,
        ):
            for index, response in zip(batch["indexes"], batch["responses"]):
                results[index] = response
        return [item or "" for item in results]

    @torch.inference_mode()
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.generate_batch([(system_prompt, user_prompt)], batch_size=1)[0]
