"""Incremental JSONL cache and microbatch orchestration shared by LLM stages."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable


class VersionedJsonlCache:
    """Append-only, crash-resilient prompt cache.

    The last row for a key wins. ``replace=True`` is used when an older cached
    response fails the current response validator; this keeps stale malformed
    rows from permanently poisoning resume runs.
    """

    def __init__(self, path: str | Path, *, fsync: bool = False):
        self.path = Path(path)
        self.fsync = fsync
        self.values: dict[str, str] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    key, response = row["key"], row["response"]
                    if isinstance(key, str) and isinstance(response, str):
                        self.values[key] = response
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue

    @staticmethod
    def make_key(
        model_id: str,
        task: str,
        prompt: tuple[str, str],
        *,
        prompt_version: str,
        generation_config: dict | None = None,
    ) -> str:
        payload = json.dumps({
            "model_id": model_id,
            "task": task,
            "prompt_version": prompt_version,
            "system": prompt[0],
            "user": prompt[1],
            "generation_config": generation_config or {},
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def put(self, key: str, response: str, *, replace: bool = False) -> bool:
        if key in self.values and not replace:
            return False
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({
                "key": key,
                "response": response,
            }, ensure_ascii=False) + "\n")
            handle.flush()
            if self.fsync:
                os.fsync(handle.fileno())
        self.values[key] = response
        return True


def generate_with_cache(
    llm,
    prompts: list[tuple[str, str]],
    *,
    batch_size: int,
    model_id: str,
    task: str,
    prompt_version: str,
    cache: VersionedJsonlCache | None = None,
    max_new_tokens: int | None = None,
    max_batch_tokens: int | None = None,
    min_batch_size: int = 1,
    dynamic_batching: bool = True,
    progress_every: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    prompt_token_counts: list[int] | None = None,
    response_validator: Callable[[int, str], bool] | None = None,
) -> list[str]:
    """Generate pending prompts and persist only validator-approved responses.

    ``response_validator`` receives the index in the original ``prompts`` list.
    Invalid cached rows are ignored and replaced only after a valid regeneration.
    Invalid newly generated rows are returned to the caller for audit/retry but
    are never stored as successful cache entries.
    """
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "max_batch_tokens": max_batch_tokens,
        "batch_size": batch_size,
        "min_batch_size": min_batch_size,
        "dynamic_batching": dynamic_batching,
    }
    results: list[str | None] = [None] * len(prompts)
    pending_indexes: list[int] = []
    pending_prompts: list[tuple[str, str]] = []
    pending_keys: list[str] = []
    stale_cached_keys: set[str] = set()
    cache_hits = 0

    for index, prompt in enumerate(prompts):
        key = VersionedJsonlCache.make_key(
            model_id,
            task,
            prompt,
            prompt_version=prompt_version,
            generation_config=generation_config,
        )
        cached = cache.get(key) if cache else None
        cached_valid = (
            cached is not None
            and (
                response_validator is None
                or response_validator(index, cached)
            )
        )
        if cached_valid:
            results[index] = cached
            cache_hits += 1
        else:
            if cached is not None:
                stale_cached_keys.add(key)
            pending_indexes.append(index)
            pending_prompts.append(prompt)
            pending_keys.append(key)

    pending_token_counts = (
        [prompt_token_counts[index] for index in pending_indexes]
        if prompt_token_counts is not None
        else None
    )
    prefix = f"[Qwen:{task}]"
    print(
        f"{prefix} prompts={len(prompts)} pending={len(pending_prompts)} "
        f"cache_hits={cache_hits} stale_cache={len(stale_cached_keys)}",
        file=sys.stderr,
        flush=True,
    )
    if not pending_prompts:
        return [item or "" for item in results]

    def completed(
        local_indexes: list[int],
        responses: list[str],
        stats: dict,
        batch_no: int,
        batch_total: int,
    ) -> None:
        if len(local_indexes) != len(responses):
            raise ValueError("microbatch returned the wrong number of responses")
        valid_count = 0
        invalid_count = 0
        for local_index, response in zip(local_indexes, responses):
            original_index = pending_indexes[local_index]
            key = pending_keys[local_index]
            results[original_index] = response
            is_valid = (
                response_validator is None
                or response_validator(original_index, response)
            )
            if is_valid:
                valid_count += 1
                if cache:
                    cache.put(
                        key,
                        response,
                        replace=key in stale_cached_keys,
                    )
            else:
                invalid_count += 1
        event = {
            "task": task,
            "batch_index": batch_no,
            "batch_total": batch_total,
            "indexes": [pending_indexes[index] for index in local_indexes],
            "responses": len(responses),
            "valid_responses": valid_count,
            "invalid_responses": invalid_count,
            "cache_hits": cache_hits,
            "stale_cache_rows": len(stale_cached_keys),
            "total_cached": len(cache.values) if cache else 0,
            **(stats or {}),
        }
        if progress_callback:
            progress_callback(event)
        if progress_every > 0 and (
            batch_no == 1
            or batch_no == batch_total
            or batch_no % progress_every == 0
        ):
            print(
                f"{prefix} batch {batch_no}/{batch_total} done "
                f"latency={float(stats.get('latency_seconds', 0.0)):.2f}s "
                f"input_tokens={stats.get('real_input_tokens', '?')} "
                f"output_tokens={stats.get('real_output_tokens', '?')} "
                f"valid={valid_count} invalid={invalid_count} "
                f"cached={len(cache.values) if cache else cache_hits}",
                file=sys.stderr,
                flush=True,
            )

    if hasattr(llm, "generate_batches"):
        iterator = llm.generate_batches(
            pending_prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            max_batch_tokens=max_batch_tokens,
            min_batch_size=min_batch_size,
            dynamic_batching=dynamic_batching,
            prompt_lengths=pending_token_counts,
        )
        batches = list(iterator) if isinstance(iterator, list) else iterator
        batch_no = 0
        for batch in batches:
            batch_no += 1
            completed(
                list(batch["indexes"]),
                list(batch["responses"]),
                dict(batch.get("stats", {})),
                batch_no,
                int(batch.get("batch_total", batch_no)),
            )
    else:
        total = math.ceil(len(pending_prompts) / batch_size)
        for batch_no, start in enumerate(
            range(0, len(pending_prompts), batch_size),
            1,
        ):
            group = pending_prompts[start:start + batch_size]
            kwargs = {"batch_size": batch_size}
            if max_new_tokens is not None:
                kwargs["max_new_tokens"] = max_new_tokens
            try:
                generated = llm.generate_batch(group, **kwargs)
            except TypeError:
                if "max_new_tokens" not in kwargs:
                    raise
                generated = llm.generate_batch(group, batch_size=batch_size)
            completed(
                list(range(start, start + len(group))),
                generated,
                {},
                batch_no,
                total,
            )
    if any(item is None for item in results):
        raise RuntimeError("generation ended before all pending prompts completed")
    return [item or "" for item in results]