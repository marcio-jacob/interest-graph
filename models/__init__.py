"""
models/
=======
Graph neural network models for recommendation scoring.

Current modules
---------------
graph_sage : Two-hop GraphSAGE mean aggregator using Node2Vec embeddings
             as initial node features. Provides a link-prediction score
             for user–video pairs without requiring PyTorch.

Upgrade path
------------
Replace the NumPy-based GraphSAGEScorer with torch_geometric.nn.SAGEConv
once the feature pipeline is validated and training data is assembled.
A training script would:
  1. Load all User and Video node2vec embeddings as initial features
  2. Sample positive (LIKED/VIEWED high-cr) and negative (SKIPPED) edges
  3. Train the bilinear W matrix via cross-entropy on link prediction
  4. Save W to models/weights/graph_sage_W.npy for inference
"""

from .graph_sage import GraphSAGEScorer

__all__ = ["GraphSAGEScorer"]
