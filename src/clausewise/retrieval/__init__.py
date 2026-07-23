"""Retrieval: fusion and the mode-switched retrieval service."""

from clausewise.retrieval.fusion import reciprocal_rank_fusion
from clausewise.retrieval.service import RetrievalMode, RetrievalService

__all__ = ["RetrievalMode", "RetrievalService", "reciprocal_rank_fusion"]
