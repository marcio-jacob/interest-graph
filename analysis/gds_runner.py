"""
analysis/gds_runner.py
======================
Runs GDS algorithms against the interest graph using a Neo4j Aura GDS Session
(serverless graph analytics) via the v2 Arrow-based API.

After each algorithm, results are written back to the main Aura database so
recommendation queries can use them.

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
PROJECT_ID     = os.environ.get("PROJECT_ID")   # optional but recommended

SESSION_NAME   = "interest-graph-analytics"
SESSION_MEMORY = SessionMemory.m_8GB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(step, title: str) -> None:
    print(f"\n── Step {step}: {title} ──")


def _fmt(n) -> str:
    if isinstance(n, float):
        return f"{n:.6f}"
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Graph projections (v2 Arrow API — Cypher-based remote projection)
# ---------------------------------------------------------------------------

def _project_graphs(gds) -> dict:
    """Create all in-memory graph projections needed by the algorithms."""
    projs = {}

    print("  Projecting user_topic_interest...")
    r = gds.v2.graph.project(
        "user_topic_interest",
        """
        MATCH (u:User)-[rel:INTERESTED_IN_TOPIC]->(t:Topic)
        RETURN gds.graph.project.remote(u, t, {
            sourceNodeLabels: ['User'],
            targetNodeLabels: ['Topic'],
            relationshipType: 'INTERESTED_IN_TOPIC',
            relationshipProperties: {topic_score: rel.topic_score}
        })
        """,
    )
    projs["user_topic"] = r.graph
    print(f"    nodes={r.result.node_count:,}  rels={r.result.relationship_count:,}")

    print("  Projecting follow_graph...")
    r = gds.v2.graph.project(
        "follow_graph",
        """
        MATCH (u:User)-[rel:FOLLOWS]->(v:User)
        RETURN gds.graph.project.remote(u, v, {
            sourceNodeLabels: ['User'],
            targetNodeLabels: ['User'],
            relationshipType: 'FOLLOWS',
            relationshipProperties: {engagement_score: rel.engagement_score}
        })
        """,
    )
    projs["follow"] = r.graph
    print(f"    nodes={r.result.node_count:,}  rels={r.result.relationship_count:,}")

    print("  Projecting view_graph...")
    r = gds.v2.graph.project(
        "view_graph",
        """
        MATCH (s:UserSession)-[rel:VIEWED]->(v:Video)
        RETURN gds.graph.project.remote(s, v, {
            sourceNodeLabels: ['UserSession'],
            targetNodeLabels: ['Video'],
            relationshipType: 'VIEWED',
            relationshipProperties: {completion_rate: rel.completion_rate}
        })
        """,
    )
    projs["view"] = r.graph
    print(f"    nodes={r.result.node_count:,}  rels={r.result.relationship_count:,}")

    print("  Projecting interaction_graph (for Node2Vec)...")
    r = gds.v2.graph.project(
        "interaction_graph",
        """
        MATCH (n)-[rel]->(m)
        WHERE type(rel) IN [
            'INTERESTED_IN_TOPIC', 'INTERESTED_IN_ENTITY', 'INTERESTED_IN_HASHTAG',
            'IS_ABOUT', 'HAS_HASHTAG', 'MENTIONS', 'VIEWED'
        ]
        AND (n:User OR n:Video OR n:Topic OR n:Entity OR n:Hashtag OR n:UserSession)
        AND (m:User OR m:Video OR m:Topic OR m:Entity OR m:Hashtag OR m:UserSession)
        RETURN gds.graph.project.remote(n, m, {
            sourceNodeLabels: labels(n),
            targetNodeLabels: labels(m),
            relationshipType: type(rel)
        })
        """,
        undirected_relationship_types=[
            "INTERESTED_IN_TOPIC", "INTERESTED_IN_ENTITY", "INTERESTED_IN_HASHTAG",
            "IS_ABOUT", "HAS_HASHTAG", "MENTIONS", "VIEWED",
        ],
    )
    projs["interaction"] = r.graph
    print(f"    nodes={r.result.node_count:,}  rels={r.result.relationship_count:,}")

    return projs


def _drop_graphs(gds, projs: dict) -> None:
    for name, G in projs.items():
        try:
            gds.v2.graph.drop(G)
            print(f"  Dropped projection: {name}")
        except Exception as e:
            print(f"  Warning: could not drop {name}: {e}")


# ---------------------------------------------------------------------------
# Algorithm runners (v2 API)
# ---------------------------------------------------------------------------

def run_louvain(gds, G) -> None:
    """Community detection on user-topic interest graph."""
    res = gds.v2.louvain.write(
        G,
        write_property="louvain_community",
        relationship_weight_property="topic_score",
        max_iterations=10,
        max_levels=10,
    )
    print(f"    communities={_fmt(res.community_count)}  "
          f"modularity={res.modularity:.4f}  "
          f"written={_fmt(res.node_properties_written)}")

    # Copy louvain_community → community_id for use in recommendation queries
    gds.run_cypher("""
        MATCH (u:User) WHERE u.louvain_community IS NOT NULL
        SET u.community_id = u.louvain_community
    """)
    print("    community_id synced.")


def run_pagerank_creators(gds, G) -> None:
    """PageRank on follow graph — creator influence score."""
    res = gds.v2.page_rank.write(
        G,
        write_property="pagerank_score",
        relationship_weight_property="engagement_score",
        max_iterations=20,
        damping_factor=0.85,
    )
    print(f"    written={_fmt(res.node_properties_written)}  "
          f"iterations={res.ran_iterations}")

    # Print top 5 creators by pagerank
    top = gds.run_cypher("""
        MATCH (c:Creator)
        WHERE c.pagerank_score IS NOT NULL
        RETURN c.username AS username, round(c.pagerank_score, 6) AS score
        ORDER BY score DESC LIMIT 5
    """)
    print("    Top creators by PageRank:")
    for _, r in top.iterrows():
        print(f"      {r['username']:<20} {r['score']}")


def run_view_pagerank(gds, G) -> None:
    """View-weighted PageRank on videos — completion quality signal."""
    res = gds.v2.page_rank.write(
        G,
        write_property="view_pagerank",
        relationship_weight_property="completion_rate",
        max_iterations=20,
        damping_factor=0.85,
        scaler="L1Norm",
    )
    print(f"    written={_fmt(res.node_properties_written)}  "
          f"iterations={res.ran_iterations}")

    top = gds.run_cypher("""
        MATCH (v:Video)
        WHERE v.view_pagerank IS NOT NULL
        RETURN left(v.description, 50) AS desc,
               round(v.view_pagerank, 8) AS score
        ORDER BY score DESC LIMIT 5
    """)
    print("    Top videos by view-PageRank:")
    for _, r in top.iterrows():
        print(f"      {r['desc']:<52} {r['score']}")


def run_node_similarity(gds, G) -> None:
    """Jaccard node similarity -> SIMILAR_TO relationships for collaborative filtering."""
    res = gds.v2.node_similarity.write(
        G,
        write_relationship_type="SIMILAR_TO",
        write_property="similarity",
        similarity_cutoff=0.1,
        top_k=10,
        relationship_weight_property="topic_score",
    )
    print(f"    compared={_fmt(res.nodes_compared)}  "
          f"SIMILAR_TO written={_fmt(res.relationships_written)}")


def run_betweenness(gds, G) -> None:
    """Betweenness centrality on follow graph — bridge user detection."""
    res = gds.v2.betweenness_centrality.write(
        G,
        write_property="betweenness_centrality",
        sampling_size=500,
    )
    dist = res.centrality_distribution
    print(f"    written={_fmt(res.node_properties_written)}  "
          f"min={dist.get('min', 0):.4f}  max={dist.get('max', 0):.4f}")


def run_wcc(gds, G) -> None:
    """Weakly Connected Components — connectivity sanity check."""
    res = gds.v2.wcc.write(G, write_property="wcc_component")
    print(f"    components={_fmt(res.component_count)}  "
          f"written={_fmt(res.node_properties_written)}")
    if res.component_count <= 3:
        print("    Graph is well connected")
    else:
        print(f"    Warning: {res.component_count} components — may have isolated nodes")


def run_node2vec(gds, G) -> None:
    """Node2Vec embeddings for User and Video nodes (64-dim)."""
    res = gds.v2.node2vec.write(
        G,
        write_property="node2vec_embedding",
        embedding_dimension=64,
        walk_length=80,
        walks_per_node=10,
        in_out_factor=1.0,
        return_factor=1.0,
        iterations=5,
        window_size=5,
        negative_sampling_rate=5,
    )
    print(f"    written={_fmt(res.node_properties_written)}")
    if res.loss_per_iteration:
        losses = [f"{l:.4f}" for l in res.loss_per_iteration]
        print(f"    loss per iter: {' -> '.join(losses)}")


# ---------------------------------------------------------------------------
# Creator analytics writebacks
# ---------------------------------------------------------------------------

def run_creator_topic_centrality(gds) -> None:
    """
    For each (Creator, Topic) pair, compute a weighted centrality score:
        creator_topic_score = pagerank_score x sum(topic_interests from followers)

    Written back as Creator.top_topic (dominant topic slug) and
    Creator.top_topic_score (composite score).
    """
    gds.run_cypher("""
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
    count = gds.run_cypher(
        "MATCH (c:Creator) WHERE c.top_topic IS NOT NULL RETURN count(c) AS n"
    ).iloc[0]["n"]
    print(f"    top_topic written to {count:,} creators")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(skip_embeddings: bool = False) -> None:
    t_start = time.perf_counter()

    print("Connecting to Aura GDS Sessions API...")
    sessions = GdsSessions(
        api_credentials=AuraAPICredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            project_id=PROJECT_ID,
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

        _banner(3, "PageRank -- creator follow graph")
        run_pagerank_creators(gds, projs["follow"])

        _banner(4, "View-weighted PageRank on videos")
        run_view_pagerank(gds, projs["view"])

        _banner(5, "NodeSimilarity (Jaccard) -> SIMILAR_TO")
        run_node_similarity(gds, projs["user_topic"])

        _banner(6, "Betweenness centrality")
        run_betweenness(gds, projs["follow"])

        _banner(7, "Weakly Connected Components")
        run_wcc(gds, projs["follow"])

        if not skip_embeddings:
            _banner(8, "Node2Vec embeddings (64-dim)")
            run_node2vec(gds, projs["interaction"])
        else:
            print("\n-- Step 8: Node2Vec -- skipped (--skip-embeddings) --")

        _banner(9, "Creator x Topic centrality writeback")
        run_creator_topic_centrality(gds)

    finally:
        _banner("X", "Dropping graph projections")
        _drop_graphs(gds, projs)

        print("\n-- Closing GDS session --")
        try:
            sessions.delete(session_name=SESSION_NAME)
            print("  Session deleted.")
        except Exception as e:
            print(f"  Warning: could not delete session: {e}")

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
