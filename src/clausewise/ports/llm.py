"""LLM provider port."""

from typing import Protocol, runtime_checkable

from clausewise.domain import LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Text-completion interface for grounded answer generation.

    Implementations: Gemini (default, free tier), Anthropic, OpenAI.
    Raise ``ProviderError`` on failure after internal retries.
    """

    @property
    def model_name(self) -> str: ...

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate a completion. Deterministic settings (t=0) are the default
        because extractive legal Q&A rewards consistency over creativity."""
        ...
