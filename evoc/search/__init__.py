"""Nearest-neighbour search over a fitted EVoC model's training data."""
from .index import SearchIndex, prepare_search

__all__ = ["SearchIndex", "prepare_search"]
