"""Generation-side domain values: answers, citations, refusals, token usage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """LLM token consumption for one call. Basis for cost accounting."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw provider response, normalized. Providers never leak their own types."""

    text: str
    model: str
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class Citation:
    """A verified reference from an answer back to a source chunk.

    ``quote`` must be an exact (or near-exact, post-normalization) substring of
    the cited chunk's text — enforced by the citation verifier (Checkpoint 13),
    never taken on the LLM's word.
    """

    chunk_id: str
    quote: str
    relevance: float  # score of the cited chunk at generation time


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer. Invariant: every claim is backed by ``citations``."""

    text: str
    citations: tuple[Citation, ...]
    confidence: float  # derived from retrieval/rerank scores, in [0, 1]
    model: str
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class Refusal:
    """Explicit 'not found in the provided contracts' outcome.

    A first-class result — not an error. The refusal path is a feature
    (PROJECT_PLAN Ckpt 13) and is measured in evaluation (Ckpt 15).
    """

    reason: str
    best_score: float | None = None  # top rerank score that failed the threshold


# What the generation service returns: an answer or an honest refusal.
GenerationOutcome = Answer | Refusal
