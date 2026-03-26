"""
analysis/gds_runner.py
======================
Runs GDS algorithms against the interest graph using a Neo4j Aura GDS Session
(serverless graph analytics).  After each algorithm, results are written back
to the main Aura database so recommendation queries can use them.

Algorithms run:
  1.  Louvain community detection   → User.louvain_community / User.community_id
  2.  PageRank (follow graph)        → User.pagerank_score  (creator centrality)
  3.  View-weighted PageRank         → Video.view_pagerank  (quality signal)
  4.  NodeSimilarity (Jaccard)       → :SIMILAR_TO{similarity} relationships
  5.  Betweenness Centrality         → User.betweenness_centrality
  6.  WCC (sanity check)             → User.wcc_component
  7.  Node2Vec embeddings            → User/Video.node2vec_embedding (float[])

Usage
-----
    python analysis/gds_runner.py [--skip-embeddings]

Requires:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD  (in .env)
    AURA_CLIENT_ID, AURA_CLIENT_SECRET     (in .env — from console.neo4j.io)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from dotenv import load_dotenv
from graphdatascience.session import (
    AuraAPICredentials,
    DbmsConnectionInfo,
    GdsSessions,
    SessionMemory,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USER     = os.environ["NEO4J_USER"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
CLIENT_ID      = os.environ["AURA_CLIENT_ID"]
CLIENT_SECRET  = os.environ["AURA_CLIENT_SECRET"]

SESSION_NAME   = "interest-graph-analytics"
SESSION_MEMORY = SessionMemory.m_8GB   # adjust to your Aura tier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step: int, title: str) -> None:
    print(f"\n── Step {step}: {title} ──")


def _fmt(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:.6f}"
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Graph projections
# ---------------------------------------------------------------------------

def _project_graphs(gds) -> dict:
    """Create all in-memory graph projections needed by the algorithms."""
    projs = {}

    print("  Projecting user_topic_interest...")
    G, res = gds.graph.project(
        "user_topic_interest",
        ["User", "Topic"],
        {"INTERESTED_IN_TOPIC": {"orientation": "NATURAL", "properties": ["topic_score"]}},
    )
    projs["user_topic"] = G
    print(f"    nodes={res['nodeCount']:,}  rels={res['relationshipCount']:,}")

    print("  Projecting follow_graph...")
    G, res = gds.graph.project(
        "follow_graph",
        "User",
        {"FOLLOWS": {"orientation": "NATURAL", "properties": ["engagement_score"]}},
    )
    projs["follow"] = G
    print(f"    nodes={res['nodeCount']:,}  rels={res['relationshipCount']:,}")

    print("  Projecting view_graph...")
    G, res = gds.graph.project(
        "view_graph",
        ["User", "Video"],
        {"VIEWED": {"orientation": "NATURAL", "properties": ["completion_rate"]}},
    )
    projs["view"] = G
    print(f"    nodes={res['nodeCount']:,}  rels={res['relationshipCount']:,}")

    print("  Projecting interaction_graph (for Node2Vec)...")
    G, res = gds.graph.project(
        "interaction_graph",
        ["User", "Video", "Topic", "Entity", "Hashtag"],
        {
            "INTERESTED_IN_TOPIC":   {"orientation": "UNDIRECTED", "properties": ["topic_score"]},
            "INTERESTED_IN_ENTITY":  {"orientation": "UNDIRECTED", "properties": ["entity_score"]},
            "INTERESTED_IN_HASHTAG": {"orientation": "UNDIRECTED", "properties": ["hashtag_score"]},
            "IS_ABOUT":              {"orientation": "UNDIRECTED"},
            "HAS_HASHTAG":           {"orientation": "UNDIRECTED"},
            "MENTIONS":              {"orientation": "UNDIRECTED"},
            "VIEWED":                {"orientation": "UNDIRECTED", "properties": ["completion_rate"]},
        },
    )
    projs["interaction"] = G
    print(f"    nodes={res['nodeCount']:,}  rels={res['relationshipCount']:,}")

    return projs


def _drop_graphs(projs: dict) -> None:
    for name, G in projs.items():
        try:
            G.drop()
            print(f"  Dropped projection: {name}")
        except Exception as e:
            print(f"  Warning: could not drop {name}: {e}")


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def run_louvain(gds, G) -> None:
    """Community detection on user-topic interest graph."""
    res = gds.louvain.write(
        G,
        writeProperty="louvain_community",
        relationshipWeightProperty="topic_score",
        maxIterations=10,
        maxLevels=10,
    )
    print(f"    communities={_fmt(res['communityCount'])}  "
          f"modularity={res['modularity']:.4f}  "
          f"written={_fmt(res['nodePropertiesWritten'])}")

    # Copy louvain_community → community_id for use in recommendation queries
    with gds._query_runner._driver.session() as session:
        session.run("""
            MATCH (u:User) WHERE u.louvain_community IS NOT NULL
            SET u.community_id = u.louvain_community
        """)
    print("    community_id synced.")


def run_pagerank_creators(gds, G) -> None:
    """PageRank on follow graph — creator influence score."""
    res = gds.pageRank.write(
        G,
        writeProperty="pagerank_score",
        relationshipWeightProperty="engagement_score",
        maxIterations=20,
        dampingFactor=0.85,
    )
    print(f"    written={_fmt(res['nodePropertiesWritten'])}  "
          f"iterations={res['ranIterations']}")

    # Print top 5 creators by pagerank
    with gds._query_runner._driver.session() as session:
        top = session.run("""
            MATCH (c:Creator)
            WHERE c.pagerank_score IS NOT NULL
            RETURN c.username AS username, round(c.pagerank_score, 6) AS score
            ORDER BY score DESC LIMIT 5
        """).data()
    print("    Top creators by PageRank:")
    for r in top:
        print(f"      {r['username']:<20} {r['score']}")


def run_view_pagerank(gds, G) -> None:
    """View-weighted PageRank on videos — completion quality signal."""
    res = gds.pageRank.write(
        G,
        writeProperty="view_pagerank",
        relationshipWeightProperty="completion_rate",
        maxIterations=20,
        dampingFactor=0.85,
        scaler="L1Norm",
    )
    print(f"    written={_fmt(res['nodePropertiesWritten'])}  "
          f"iterations={res['ranIterations']}")

    with gds._query_runner._driver.session() as session:
        top = session.run("""
            MATCH (v:Video)
            WHERE v.view_pagerank IS NOT NULL
            RETURN left(v.description, 50) AS desc,
                   round(v.view_pagerank, 8) AS score
            ORDER BY score DESC LIMIT 5
        """).data()
    print("    Top videos by view-PageRank:")
    for r in top:
        print(f"      {r['desc']:<52} {r['score']}")


def run_node_similarity(gds, G) -> None:
    """Jaccard node similarity → SIMILAR_TO relationships for collaborative filtering."""
    res = gds.nodeSimilarity.write(
        G,
        writeRelationshipType="SIMILAR_TO",
        writeProperty="similarity",
        similarityCutoff=0.1,
        topK=10,
        relationshipWeightProperty="topic_score",
    )
    print(f"    compared={_fmt(res['nodesCompared'])}  "
          f"SIMILAR_TO written={_fmt(res['relationshipsWritten'])}")


def run_betweenness(gds, G) -> None:
    """Betweenness centrality on follow graph — bridge user detection."""
    res = gds.betweenness.write(
        G,
        writeProperty="betweenness_centrality",
        samplingSize=500,
    )
    print(f"    written={_fmt(res['nodePropertiesWritten'])}  "
          f"min={res['minimumScore']:.4f}  max={res['maximumScore']:.4f}")


def run_wcc(gds, G) -> None:
    """Weakly Connected Components — connectivity sanity check."""
    res = gds.wcc.write(G, writeProperty="wcc_component")
    print(f"    components={_fmt(res['componentCount'])}  "
          f"written={_fmt(res['nodePropertiesWritten'])}")
    if res["componentCount"] <= 3:
        print("    ✓ Graph is well connected")
    else:
        print(f"    ⚠ {res['componentCount']} components — may have isolated nodes")


def run_node2vec(gds, G) -> None:
    """Node2Vec embeddings for User and Video nodes (64-dim)."""
    res = gds.node2vec.write(
        G,
        writeProperty="node2vec_embedding",
        embeddingDimension=64,
        walkLength=80,
        walksPerNode=10,
        inOutFactor=1.0,
        returnFactor=1.0,
        iterations=5,
        windowSize=5,
        negativeSamplingRate=5,
    )
    print(f"    written={_fmt(res['nodePropertiesWritten'])}")
    if "lossPerIteration" in res:
        losses = [f"{l:.4f}" for l in res["lossPerIteration"]]
        print(f"    loss per iter: {' → '.join(losses)}")


# ---------------------------------------------------------------------------
# Creator analytics writebacks
# ---------------------------------------------------------------------------

def run_creator_topic_centrality(driver) -> None:
    """
    For each (Creator, Topic) pair, compute a weighted centrality score:
        creator_topic_score = pagerank_score × sum(topic_interests from followers)

    Written back as Creator.top_topic (dominant topic slug) and
    Creator.top_topic_score (composite score).
    """
    with driver.session() as session:
        session.run("""
            MATCH (c:Creator)<-[:FOLLOWS]-(follower:User)-[ti:INTERESTED_IN_TOPIC]->(t:Topic)
            WITH c, t,
                 coalesce(c.pagerank_score, 0.001) AS pr,
                 sum(ti.topic_score) AS follower_affinity
            WITH c, t, pr * follower_affinity AS composite
            ORDER BY c.user_id, composite DESC
            WITH c, collect({topic: t.slug, score: composite})[0] AS best
            SET c.top_topic       = best.topic,
                c.top_topic_score = round(best.score, 6)
        """)
        count = session.run(
            "MATCH (c:Creator) WHERE c.top_topic IS NOT NULL RETURN count(c) AS n"
        ).single()["n"]
    print(f"    top_topic written to {count:,} creators")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(skip_embeddings: bool = False) -> None:
    t_start = time.perf_counter()

    print("Connecting to Aura GDS Sessions API...")
    sessions = GdsSessions(
        api_key=AuraAPICredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )

    db_info = DbmsConnectionInfo(
        uri=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )

    print(f"Creating / reusing GDS session '{SESSION_NAME}' ({SESSION_MEMORY})...")
    gds = sessions.get_or_create(
        session_name=SESSION_NAME,
        memory=SESSION_MEMORY,
        db_connection=db_info,
    )
    print("  Session ready.")

    projs = {}
    try:
        _banner(1, "Project graphs into GDS")
        projs = _project_graphs(gds)

        _banner(2, "Louvain community detection")
        run_louvain(gds, projs["user_topic"])

        _banner(3, "PageRank — creator follow graph")
        run_pagerank_creators(gds, projs["follow"])

        _banner(4, "View-weighted PageRank on videos")
        run_view_pagerank(gds, projs["view"])

        _banner(5, "NodeSimilarity (Jaccard) → SIMILAR_TO")
        run_node_similarity(gds, projs["user_topic"])

        _banner(6, "Betweenness centrality")
        run_betweenness(gds, projs["follow"])

        _banner(7, "Weakly Connected Components")
        run_wcc(gds, projs["follow"])

        if not skip_embeddings:
            _banner(8, "Node2Vec embeddings (64-dim)")
            run_node2vec(gds, projs["interaction"])
        else:
            print("\n── Step 8: Node2Vec — skipped (--skip-embeddings) ──")

        _banner(9, "Creator × Topic centrality writeback")
        # Use the underlying Neo4j driver for direct Cypher writebacks
        import neo4j as _neo4j
        driver = _neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        run_creator_topic_centrality(driver)
        driver.close()

    finally:
        _banner("X", "Dropping graph projections")
        _drop_graphs(projs)

        print("\n── Closing GDS session ──")
        sessions.delete(SESSION_NAME)
        print("  Session deleted.")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"GDS analysis complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print("Properties written to graph:")
    print("  User  : louvain_community, community_id, pagerank_score,")
    print("          betweenness_centrality, wcc_component, node2vec_embedding")
    print("  Video : view_pagerank, node2vec_embedding")
    print("  Creator: top_topic, top_topic_score")
    print("  Rels  : SIMILAR_TO{similarity}  (user-user collaborative filtering)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GDS algorithms and write results back to Neo4j Aura."
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip Node2Vec (slow; can be run separately).",
    )
    args = parser.parse_args()
    main(skip_embeddings=args.skip_embeddings)
