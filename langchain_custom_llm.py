"""LangChain-compatible wrapper for GoToCompany's LiteLLM endpoint"""
import os
from typing import Any, List, Optional
import requests
from langchain_core.language_models.llms import LLM as BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun


class CustomGoToLLM(BaseLLM):
    """
    LangChain-compatible LLM wrapper for GoToCompany's endpoint.
    Makes direct HTTP calls without authentication headers.
    """

    model: str = "GoToCompany/Llama-Sahabat-AI-v2-70B-R"
    base_url: str = "https://litellm-staging.gopay.sh"
    temperature: float = 0.6
    max_tokens: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "custom_goto_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: The prompt to send
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional parameters

        Returns:
            The generated text
        """
        messages = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": False
        }

        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling custom LLM endpoint: {str(e)}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected response format: {str(e)}") from e
