"""
Multi-Model LLM Provider Interface - ECCV 2026

Supports:
- OpenAI (GPT-4o, GPT-5.2)
- Google Gemini (Gemini 3 Pro)
- Local Models via OpenAI-compatible API (MLX, vLLM, LM Studio)
- Ollama
"""

import os
import time
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


# Cost tracking (per 1k tokens)
MODEL_COSTS = {
    # OpenAI
    "gpt-5.2": {"prompt": 0.003, "completion": 0.012},
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4.1-mini": {"prompt": 0.001, "completion": 0.002},
    # Gemini
    "gemini-3-pro-preview": {"prompt": 0.00125, "completion": 0.005},
    "gemini-2.0-flash": {"prompt": 0.0001, "completion": 0.0004},
    # Local models - free
    "default_local": {"prompt": 0, "completion": 0},
}


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        """
        Get completion from LLM

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (response_text, cost)
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get model identifier string"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.costs = MODEL_COSTS.get(model, {"prompt": 0.003, "completion": 0.012})
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        max_retries = 20
        for i in range(max_retries):
            try:
                # GPT-5.x and newer models use max_completion_tokens
                if self.model.startswith("gpt-5") or self.model.startswith("o1") or self.model.startswith("o3"):
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_completion_tokens=max_tokens
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                cost = (prompt_tokens / 1000 * self.costs["prompt"] +
                       completion_tokens / 1000 * self.costs["completion"])
                return response.choices[0].message.content, cost
            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"OpenAI error: {type(e).__name__}: {e}")
                    return "Error", 0

    def get_model_name(self) -> str:
        return f"openai/{self.model}"


class GeminiProvider(LLMProvider):
    """Google Gemini API provider with rate limiting"""

    def __init__(self, api_key: str, model: str = "gemini-3-pro-preview"):
        self.api_key = api_key
        self.model = model
        self.costs = MODEL_COSTS.get(model, {"prompt": 0.002, "completion": 0.008})

        # Rate limiting: 25 RPM = 2.4s per request, use 3s for safety margin
        self.min_request_interval = 3.0
        self.last_request_time = 0

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model)

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        import google.generativeai as genai

        # Rate limiting to avoid hitting RPM limits
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

        max_retries = 20
        for i in range(max_retries):
            try:
                gemini_messages = []
                system_prompt = ""

                for msg in messages:
                    role = msg["role"]
                    content = msg["content"]

                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        gemini_messages.append({"role": "user", "parts": [content]})
                    elif role == "assistant":
                        gemini_messages.append({"role": "model", "parts": [content]})

                chat = self.client.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])

                last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
                if system_prompt:
                    last_message = f"{system_prompt}\n\n{last_message}"

                response = chat.send_message(
                    last_message,
                    generation_config=genai.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                )

                # Estimate token counts
                prompt_tokens = sum(len(m["parts"][0]) // 4 for m in gemini_messages)
                completion_tokens = len(response.text) // 4
                cost = (prompt_tokens / 1000 * self.costs["prompt"] +
                       completion_tokens / 1000 * self.costs["completion"])

                return response.text, cost

            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Gemini error: {type(e).__name__}: {e}")
                    return "Error", 0

    def get_model_name(self) -> str:
        return f"gemini/{self.model}"


class LocalAPIProvider(LLMProvider):
    """Local model provider via OpenAI-compatible API (MLX, vLLM, LM Studio)"""

    def __init__(
        self,
        model: str = "mlx-community/Llama-3.3-70B-Instruct-4bit",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.costs = {"prompt": 0, "completion": 0}

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        import requests

        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"], 0

            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Local API error: {type(e).__name__}: {e}")
                    return "Error", 0

    def get_model_name(self) -> str:
        return f"local/{self.model}"


class OllamaProvider(LLMProvider):
    """Ollama local model provider"""

    def __init__(self, model: str = "llama3:8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
        self.costs = {"prompt": 0, "completion": 0}

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        import requests

        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                            "num_predict": max_tokens,
                        }
                    },
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"], 0

            except Exception as e:
                if i < max_retries - 1:
                    time.sleep(2)
                else:
                    print(f"Ollama error: {type(e).__name__}: {e}")
                    return "Error", 0

    def get_model_name(self) -> str:
        return f"ollama/{self.model}"


class MultiModelLLM:
    """Multi-model LLM manager with parallel execution"""

    def __init__(self, provider: LLMProvider, num_workers: int = 10):
        self.provider = provider
        self.num_workers = num_workers

    def get_completion(
        self,
        messages: List[Dict],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[str, float]:
        return self.provider.get_completion(messages, temperature, max_tokens)

    def get_multiple_completions(
        self,
        dialogs: List[List[Dict]],
        temperature: float = 0,
        max_tokens: int = 800
    ) -> Tuple[List[str], float]:
        """
        Get completions for multiple dialogs in parallel

        Args:
            dialogs: List of dialog message lists
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (list of responses, total cost)
        """
        get_completion_partial = partial(
            self.provider.get_completion,
            temperature=temperature,
            max_tokens=max_tokens
        )

        results = [None] * len(dialogs)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(get_completion_partial, d): i
                for i, d in enumerate(dialogs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        total_cost = sum(cost for _, cost in results)
        responses = [response for response, _ in results]

        return responses, total_cost

    def get_model_name(self) -> str:
        return self.provider.get_model_name()


def create_llm_provider(
    provider_type: str,
    model: str = None,
    api_key: str = None,
    base_url: str = None,
) -> LLMProvider:
    """
    Factory function to create LLM provider instance

    Args:
        provider_type: "openai", "gemini", "ollama", or "local"
        model: Model name/identifier
        api_key: API key (required for openai and gemini)
        base_url: Base URL for local API server

    Returns:
        LLMProvider instance
    """
    if provider_type == "openai":
        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        model = model or "gpt-4o"
        return OpenAIProvider(api_key=api_key, model=model)

    elif provider_type == "gemini":
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key required")
        model = model or "gemini-3-pro-preview"
        return GeminiProvider(api_key=api_key, model=model)

    elif provider_type == "ollama":
        model = model or "llama3:8b"
        return OllamaProvider(model=model)

    elif provider_type == "local":
        model = model or "mlx-community/Llama-3.3-70B-Instruct-4bit"
        base_url = base_url or "http://localhost:8000/v1"
        return LocalAPIProvider(model=model, base_url=base_url, api_key=api_key or "not-needed")

    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
