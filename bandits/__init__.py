"""
bandits/
========
Contextual bandits for adaptive engine mixing.

Stage 1 of the RL progression: instead of using fixed engine_weights in the
reranker, a bandit learns which engines produce clicked/completed content for
each user context.

Exports
-------
BanditContext        — feature vector describing the current request context
EpsilonGreedyBandit  — simple exploration with decaying epsilon
LinUCBBandit         — upper confidence bound with linear reward model
"""

from .contextual import BanditContext, EpsilonGreedyBandit, LinUCBBandit, ENGINE_NAMES

__all__ = ["BanditContext", "EpsilonGreedyBandit", "LinUCBBandit", "ENGINE_NAMES"]
