"""
bandits/contextual.py
=====================
Contextual bandits for adaptive recommendation engine mixing.

At each request the bandit decides which engines to activate. After the user
interacts with served content, the observed completion_rate is used as the
reward signal to update the bandit's model.

Two implementations are provided:

1. EpsilonGreedyBandit
   Simple epsilon-greedy: explore with probability ε (decaying), otherwise
   exploit the engines with the highest historical mean reward.

2. LinUCBBandit
   Disjoint linear UCB: each engine arm maintains a linear reward model over
   context features. Exploration is guided by an upper confidence bound.
   Reference: Li et al., "A Contextual-Bandit Approach to Personalized
   News Article Recommendation", WWW 2010.

Context features (5-dim, see BanditContext.to_array):
  0 — dominant_topic_score   : max(INTERESTED_IN_TOPIC scores) ∈ [0, 1]
  1 — topic_diversity        : normalised Shannon entropy of topic distribution
  2 — avg_session_completion : recent mean completion rate ∈ [0, 1]
  3 — has_peers              : 1.0 if user has SIMILAR_TO peers with sim ≥ 0.3
  4 — has_embeddings         : 1.0 if node2vec embedding exists for this user

Reward signal: completion_rate ∈ [0, 1] of the served item.
  ≥ 0.8  → strong positive (replay or watch-through)
  < 0.15 → negative (skip)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


ENGINE_NAMES: list[str] = [
    "content_based",
    "collab_filter",
    "embedding",
    "trending",
    "creator",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class BanditContext:
    """Feature vector describing the current user × request context."""
    dominant_topic_score: float = 0.0     # max normalised topic score [0, 1]
    topic_diversity: float = 0.0          # normalised Shannon entropy
    avg_session_completion: float = 0.0   # recent mean completion rate
    has_peers: bool = True                # user has similar peers
    has_embeddings: bool = True           # node2vec embedding exists

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.dominant_topic_score,
                self.topic_diversity,
                self.avg_session_completion,
                float(self.has_peers),
                float(self.has_embeddings),
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_session_features(cls, session) -> "BanditContext":
        """Build a BanditContext from a SessionFeatures object."""
        lt = session.long_term_topics
        dominant = max(lt.values(), default=0.0)
        diversity = session.topic_diversity_entropy()
        return cls(
            dominant_topic_score=dominant,
            topic_diversity=diversity,
            avg_session_completion=session.avg_completion,
            has_peers=True,         # set by caller after checking SIMILAR_TO
            has_embeddings=True,    # set by caller after checking embedding
        )


# ---------------------------------------------------------------------------
# Arm statistics helper
# ---------------------------------------------------------------------------

@dataclass
class ArmStats:
    """Running statistics for one bandit arm (engine)."""
    pulls: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.pulls if self.pulls > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. Epsilon-Greedy Bandit
# ---------------------------------------------------------------------------

class EpsilonGreedyBandit:
    """
    Epsilon-greedy bandit over recommendation engines.

    At each request:
      - With probability epsilon → random subset (exploration)
      - Otherwise → top-n engines by historical mean reward (exploitation)

    Epsilon decays multiplicatively after each update call.

    Parameters
    ----------
    epsilon       : initial exploration rate
    epsilon_decay : multiplicative decay per update step
    epsilon_min   : floor for epsilon
    seed          : optional RNG seed for reproducibility
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.02,
        seed: int | None = None,
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self._rng = np.random.default_rng(seed)
        self._arms: Dict[str, ArmStats] = {e: ArmStats() for e in ENGINE_NAMES}

    def select_engines(self, n_engines: int = 3) -> List[str]:
        """
        Select `n_engines` engines to activate for this request.

        Returns a list of engine names (ordered by expected reward when
        exploiting; random order when exploring).
        """
        if self._rng.random() < self.epsilon:
            chosen = [
                str(e) for e in
                self._rng.choice(ENGINE_NAMES, size=n_engines, replace=False)
            ]
            return chosen

        sorted_arms = sorted(
            ENGINE_NAMES,
            key=lambda e: self._arms[e].mean_reward,
            reverse=True,
        )
        return sorted_arms[:n_engines]

    def update(self, engine: str, reward: float) -> None:
        """
        Record an observed reward for an engine and decay epsilon.

        Parameters
        ----------
        engine : engine name that produced the served content
        reward : observed completion_rate ∈ [0, 1]
        """
        arm = self._arms[engine]
        arm.pulls += 1
        arm.total_reward += reward
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def stats(self) -> Dict[str, dict]:
        """Return current arm statistics."""
        return {
            e: {
                "pulls":       a.pulls,
                "mean_reward": round(a.mean_reward, 4),
                "epsilon":     round(self.epsilon, 4),
            }
            for e, a in self._arms.items()
        }


# ---------------------------------------------------------------------------
# 2. LinUCB Bandit (disjoint model)
# ---------------------------------------------------------------------------

class LinUCBBandit:
    """
    LinUCB bandit (disjoint model) for context-aware engine selection.

    Each engine arm maintains its own linear reward model:
        expected_reward(a) = θ_a · context
    with an upper confidence bound for exploration:
        UCB(a) = θ_a · x + α √(x⊤ A_a⁻¹ x)

    Parameters
    ----------
    alpha       : exploration coefficient (higher → more exploration)
    context_dim : dimensionality of the context feature vector (default 5)
    seed        : optional RNG seed
    """

    def __init__(
        self,
        alpha: float = 0.5,
        context_dim: int = 5,
        seed: int | None = None,
    ) -> None:
        self.alpha = alpha
        self._d = context_dim
        self._rng = np.random.default_rng(seed)
        # Per-arm ridge regression matrices
        self._A: Dict[str, np.ndarray] = {
            e: np.eye(context_dim, dtype=np.float64) for e in ENGINE_NAMES
        }
        self._b: Dict[str, np.ndarray] = {
            e: np.zeros(context_dim, dtype=np.float64) for e in ENGINE_NAMES
        }

    def select_engines(
        self,
        context: BanditContext,
        n_engines: int = 3,
    ) -> List[str]:
        """
        Select `n_engines` with highest UCB scores given the context.

        Returns a list of engine names sorted by UCB DESC.
        """
        x = context.to_array().astype(np.float64)
        ucb_scores: Dict[str, float] = {}
        for e in ENGINE_NAMES:
            A_inv = np.linalg.inv(self._A[e])
            theta = A_inv @ self._b[e]
            ucb = float(theta @ x + self.alpha * math.sqrt(float(x @ A_inv @ x)))
            ucb_scores[e] = ucb
        return sorted(ENGINE_NAMES, key=ucb_scores.__getitem__, reverse=True)[:n_engines]

    def update(
        self,
        engine: str,
        context: BanditContext,
        reward: float,
    ) -> None:
        """
        Update the linear model for the given arm.

        Parameters
        ----------
        engine  : arm that was pulled
        context : context at the time of the pull
        reward  : observed completion_rate ∈ [0, 1]
        """
        x = context.to_array().astype(np.float64)
        self._A[engine] += np.outer(x, x)
        self._b[engine] += reward * x

    def theta(self, engine: str) -> np.ndarray:
        """Return the current reward model weights for an engine."""
        return np.linalg.inv(self._A[engine]) @ self._b[engine]

    def stats(self) -> Dict[str, dict]:
        """Return current model weights for all engines."""
        return {
            e: {"theta": self.theta(e).round(4).tolist()}
            for e in ENGINE_NAMES
        }
