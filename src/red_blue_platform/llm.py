from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .config import ApiConfig


class OpenAICompatibleClient:
    def __init__(self, config: ApiConfig):
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.8) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        last_error: httpx.HTTPStatusError | None = None
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                resp = client.post(url, headers=self.headers, json=payload)
                try:
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    retryable = exc.response.status_code in {429, 500, 502, 503, 504}
                    if not retryable or attempt >= self.config.max_retries:
                        raise
                    retry_after = exc.response.headers.get("retry-after")
                    if retry_after:
                        delay = float(retry_after)
                    else:
                        delay = self.config.retry_backoff_seconds * (2 ** attempt)
                    time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Chat completion failed without a response.")


class LocalTransformersClient:
    def __init__(self, model_path: str, device_map: str = "auto", max_new_tokens: int = 256):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local model generation requires torch and transformers in the active Python environment."
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.8) -> str:
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        generation_args: dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_args["temperature"] = temperature

        with self.torch.inference_mode():
            outputs = self.model.generate(**generation_args)

        prompt_tokens = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True).strip()
