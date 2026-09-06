#!/usr/bin/env python3
"""GPU benchmark for SGLang teacher prefix reuse and hidden-state semantics."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from kdflow.backend.sglang.sglang_engine import EngineConfig, SGLangEngineService
from kdflow.datasets.utils import convert_to_openai_messages


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-key", default="messages")
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.8)
    parser.add_argument("--answer", default="This is a cache benchmark answer.")
    parser.add_argument("--disable-radix-cache", action="store_true")
    return parser.parse_args()


def longest_common_prefix(sequences):
    if not sequences:
        return 0
    for index, values in enumerate(zip(*sequences)):
        if len(set(values)) != 1:
            return index
    return min(map(len, sequences))


def load_prompt_ids(path, input_key, tokenizer, count):
    prompts = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            value = record[input_key]
            if isinstance(value, str):
                ids = tokenizer.encode(value, add_special_tokens=False)
            else:
                chat = convert_to_openai_messages(value)
                while chat and chat[-1].get("role", "user") == "assistant":
                    chat.pop()
                template_kwargs = {}
                if "enable_thinking" in str(tokenizer.chat_template):
                    template_kwargs["enable_thinking"] = False
                prompt = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompts.append(list(ids))
            if len(prompts) == count:
                break
    if len(prompts) < count:
        raise ValueError(f"Dataset contains only {len(prompts)} usable records")
    return prompts


def gpu_memory_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return [line.strip() for line in output.splitlines()]
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = load_prompt_ids(
        args.dataset, args.input_key, tokenizer, args.num_requests
    )
    answer_ids = tokenizer.encode(args.answer, add_special_tokens=False)
    inputs = [prompt + answer_ids for prompt in prompt_ids]
    masks = [
        np.asarray(
            [False] * (len(prompt) - 1)
            + [True] * len(answer_ids)
            + [False],
            dtype=bool,
        )
        for prompt in prompt_ids
    ]

    print(json.dumps({
        "event": "input_summary",
        "prompt_lengths": list(map(len, prompt_ids)),
        "answer_tokens": len(answer_ids),
        "exact_prompt_lcp_tokens": longest_common_prefix(prompt_ids),
        "radix_cache_enabled": not args.disable_radix_cache,
    }))

    service = SGLangEngineService(EngineConfig(
        model_path=args.model,
        tp_size=args.tp_size,
        disable_radix_cache=args.disable_radix_cache,
        enable_return_hidden_states=True,
        enable_memory_saver=False,
        enable_weights_cpu_backup=False,
        mem_fraction_static=args.mem_fraction_static,
    ))
    service.start()
    try:
        for index, (input_ids, loss_mask) in enumerate(zip(inputs, masks)):
            started = time.perf_counter()
            [(hidden, metadata)] = service.generate(
                input_ids=[input_ids],
                loss_masks=[loss_mask],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_hidden_states=True,
                return_metadata=True,
            )
            elapsed = time.perf_counter() - started
            print(json.dumps({
                "event": "request",
                "index": index,
                "temperature": "cold" if index == 0 else "warm",
                "latency_seconds": elapsed,
                "input_tokens": len(input_ids),
                "selected_hidden_shape": list(hidden.shape),
                "gpu_memory_mib": gpu_memory_mib(),
                "metadata": metadata,
            }, default=str))
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
