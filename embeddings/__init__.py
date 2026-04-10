"""
embeddings/
===========
Embedding fetch and cosine similarity utilities.

The EmbeddingStore provides:
  - In-memory caching of user and video embeddings within a pipeline run
  - Batched fetch to stay within the Aura free-tier 278 MB transaction limit
  - Client-side cosine similarity (NumPy) instead of in-DB gds.similarity.cosine()

Usage
-----
    from embeddings.store import EmbeddingStore

    store = EmbeddingStore(uri, user, password)
    top_k = store.cosine_top_k_unseen(user_id, k=20)
    # → [("video_id_1", 0.813), ("video_id_2", 0.804), ...]
"""

from .store import EmbeddingStore

__all__ = ["EmbeddingStore"]
