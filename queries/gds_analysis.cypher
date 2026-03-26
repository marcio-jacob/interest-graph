// =============================================================================
// gds_analysis.cypher
// =============================================================================
// Reference Cypher for GDS algorithm calls.
// In production these are executed via analysis/gds_runner.py using the
// GDS Python client against an Aura GDS Session.
// This file documents the graph projections and algorithm parameters.
// =============================================================================


// ---------------------------------------------------------------------------
// GRAPH PROJECTIONS
// ---------------------------------------------------------------------------

// 1. User-Topic interest graph (for Louvain, NodeSimilarity, Node2Vec)
CALL gds.graph.project(
  'user_topic_interest',
  ['User', 'Topic'],
  {
    INTERESTED_IN_TOPIC: {
      type: 'INTERESTED_IN_TOPIC',
      orientation: 'NATURAL',
      properties: 'topic_score'
    }
  }
);

// 2. Follow graph among all users (for PageRank, Betweenness, WCC)
CALL gds.graph.project(
  'follow_graph',
  'User',
  {
    FOLLOWS: {
      type: 'FOLLOWS',
      orientation: 'NATURAL',
      properties: 'engagement_score'
    }
  }
);

// 3. Weighted view graph: User → Video (for view-weighted PageRank on videos)
CALL gds.graph.project(
  'view_graph',
  ['User', 'Video'],
  {
    VIEWED_WEIGHTED: {
      type: 'VIEWED',
      orientation: 'NATURAL',
      properties: 'completion_rate'
    }
  }
);

// 4. Full interaction graph for Node2Vec embeddings
// Includes users, videos, topics, entities, hashtags
CALL gds.graph.project(
  'interaction_graph',
  ['User', 'Video', 'Topic', 'Entity', 'Hashtag'],
  {
    INTERESTED_IN_TOPIC:    {orientation: 'UNDIRECTED', properties: 'topic_score'},
    INTERESTED_IN_ENTITY:   {orientation: 'UNDIRECTED', properties: 'entity_score'},
    INTERESTED_IN_HASHTAG:  {orientation: 'UNDIRECTED', properties: 'hashtag_score'},
    IS_ABOUT:               {orientation: 'UNDIRECTED'},
    HAS_HASHTAG:            {orientation: 'UNDIRECTED'},
    MENTIONS:               {orientation: 'UNDIRECTED'},
    VIEWED:                 {orientation: 'UNDIRECTED', properties: 'completion_rate'}
  }
);


// ---------------------------------------------------------------------------
// 1. LOUVAIN COMMUNITY DETECTION
// ---------------------------------------------------------------------------
// Finds interest clusters among users based on shared topic affinities.
// Results written back to User.louvain_community and User.community_id.

CALL gds.louvain.write(
  'user_topic_interest',
  {
    writeProperty: 'louvain_community',
    relationshipWeightProperty: 'topic_score',
    maxIterations: 10,
    maxLevels: 10,
    includeIntermediateCommunities: false
  }
)
YIELD communityCount, modularity, modularities
RETURN communityCount, round(modularity, 4) AS modularity;

// Copy to community_id (canonical field used by recommendation queries)
MATCH (u:User) WHERE u.louvain_community IS NOT NULL
SET u.community_id = u.louvain_community;


// ---------------------------------------------------------------------------
// 2. PAGERANK ON CREATORS (follow graph)
// ---------------------------------------------------------------------------
// Measures creator influence within the follow network.
// Written back to User.pagerank_score (only meaningful for Creator nodes).

CALL gds.pageRank.write(
  'follow_graph',
  {
    writeProperty: 'pagerank_score',
    relationshipWeightProperty: 'engagement_score',
    maxIterations: 20,
    dampingFactor: 0.85
  }
)
YIELD nodePropertiesWritten, ranIterations
RETURN nodePropertiesWritten, ranIterations;


// ---------------------------------------------------------------------------
// 3. VIEW-WEIGHTED PAGERANK ON VIDEOS
// ---------------------------------------------------------------------------
// Ranks videos by how much completion_rate flows into them from users.
// High score = users tend to finish watching this video (quality signal).
// Written back to Video.view_pagerank.

CALL gds.pageRank.write(
  'view_graph',
  {
    writeProperty: 'view_pagerank',
    relationshipWeightProperty: 'completion_rate',
    maxIterations: 20,
    dampingFactor: 0.85,
    scaler: 'L1Norm'
  }
)
YIELD nodePropertiesWritten, ranIterations
RETURN nodePropertiesWritten, ranIterations;


// ---------------------------------------------------------------------------
// 4. NODE SIMILARITY (Jaccard) — USER-USER
// ---------------------------------------------------------------------------
// Computes per-user Jaccard similarity based on shared INTERESTED_IN_TOPIC
// edges.  Writes SIMILAR_TO relationships between users, used by the
// collaborative filtering recommendation query.

CALL gds.nodeSimilarity.write(
  'user_topic_interest',
  {
    writeRelationshipType: 'SIMILAR_TO',
    writeProperty: 'similarity',
    similarityCutoff: 0.1,
    topK: 10,
    relationshipWeightProperty: 'topic_score'
  }
)
YIELD nodesCompared, relationshipsWritten
RETURN nodesCompared, relationshipsWritten;


// ---------------------------------------------------------------------------
// 5. BETWEENNESS CENTRALITY — FOLLOW GRAPH
// ---------------------------------------------------------------------------
// Identifies "bridge" users who connect otherwise-separate interest communities.
// High betweenness = user is a conduit between topic clusters.
// Written back to User.betweenness_centrality.

CALL gds.betweenness.write(
  'follow_graph',
  {
    writeProperty: 'betweenness_centrality',
    samplingSize: 500      -- approximate for large graphs
  }
)
YIELD minimumScore, maximumScore, nodePropertiesWritten
RETURN
  round(minimumScore, 4) AS min_betweenness,
  round(maximumScore, 4) AS max_betweenness,
  nodePropertiesWritten;


// ---------------------------------------------------------------------------
// 6. WEAKLY CONNECTED COMPONENTS (sanity check)
// ---------------------------------------------------------------------------
// Verifies the graph is well-connected: ideally 1 giant component.

CALL gds.wcc.write(
  'follow_graph',
  { writeProperty: 'wcc_component' }
)
YIELD componentCount, nodePropertiesWritten
RETURN componentCount, nodePropertiesWritten;

// Inspect component sizes
MATCH (u:User)
WITH u.wcc_component AS comp, count(*) AS size
ORDER BY size DESC LIMIT 5
RETURN comp, size;


// ---------------------------------------------------------------------------
// 7. NODE2VEC EMBEDDINGS
// ---------------------------------------------------------------------------
// Learns low-dimensional vector representations of users and videos by
// simulating random walks on the interaction graph.
// Embeddings are stored as float[] on each node and used as input features
// for a downstream GNN (e.g. GraphSAGE).

CALL gds.node2vec.write(
  'interaction_graph',
  {
    writeProperty: 'node2vec_embedding',
    embeddingDimension: 64,
    walkLength: 80,
    walksPerNode: 10,
    inOutFactor: 1.0,       -- p: return parameter (1.0 = neutral)
    returnFactor: 1.0,      -- q: in-out parameter (1.0 = neutral)
    iterations: 5,
    windowSize: 5,
    negativeSamplingRate: 5,
    relationshipWeightProperty: null   -- unweighted for topology capture
  }
)
YIELD nodePropertiesWritten, lossPerIteration
RETURN nodePropertiesWritten, lossPerIteration;


// ---------------------------------------------------------------------------
// 8. CLEANUP — DROP GRAPH PROJECTIONS
// ---------------------------------------------------------------------------

CALL gds.graph.drop('user_topic_interest', false) YIELD graphName;
CALL gds.graph.drop('follow_graph', false)         YIELD graphName;
CALL gds.graph.drop('view_graph', false)           YIELD graphName;
CALL gds.graph.drop('interaction_graph', false)    YIELD graphName;
