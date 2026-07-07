"""LLM provider abstraction.

Swappable backend so agents never depend on a specific vendor SDK. "mock"
produces a deterministic, purely extractive answer from the supplied context
— it lets the whole pipeline (ingestion -> retrieval -> reasoning -> citations)
run and be demoed with zero API keys, which matters when working offline
during a hackathon. Switch LLM_PROVIDER=anthropic once a key is available.
"""

from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class MockLLMProvider:
    """Deterministic, extractive fallback — no network or API key required."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "[mock-llm response — set LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY "
            "for a real generated answer]\n\n"
            f"{user_prompt[:1200]}"
        )


class AnthropicLLMProvider:
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def build_llm_provider(provider: str, api_key: str, model: str) -> LLMProvider:
    if provider == "anthropic" and api_key:
        return AnthropicLLMProvider(api_key=api_key, model=model)
    return MockLLMProvider()
