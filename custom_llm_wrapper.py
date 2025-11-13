"""Custom LLM Wrapper for GoToCompany's LiteLLM Endpoint"""
import os
import json
import requests
from typing import Any, Dict, List, Optional


class CustomLLMWrapper:
    """
    Wrapper for GoToCompany's LiteLLM endpoint that bypasses LiteLLM's provider detection.
    Makes direct HTTP calls to the endpoint using the OpenAI-compatible API format.
    """

    def __init__(
        self,
        model: str = "GoToCompany/Llama-Sahabat-AI-v2-70B-R",
        base_url: str = "https://litellm-staging.gopay.sh",
        api_key: Optional[str] = None,
        temperature: float = 0.6
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.getenv("CUSTOM_LLM_API_KEY")
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(
                "API key is required. Set CUSTOM_LLM_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def call(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Make a synchronous call to the LLM endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            The generated text content
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": False
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        # Add any additional kwargs
        payload.update(kwargs)

        headers = {
            "Content-Type": "application/json"
        }

        # Only add Authorization header if API key is provided and not a placeholder
        if self.api_key and self.api_key not in ["dummy", "not-needed", ""]:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300  # 5 minute timeout
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling custom LLM endpoint: {str(e)}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format: {str(e)}") from e

    async def acall(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Async version of call (for compatibility).
        Currently just calls the sync version.
        """
        return self.call(messages, temperature, max_tokens, stop, **kwargs)

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Allow the wrapper to be called directly."""
        return self.call(messages, **kwargs)
