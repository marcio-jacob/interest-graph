// =============================================================================
// exploration.cypher
// =============================================================================
// Sanity checks, distribution queries, and sample path traversals.
// Run these in Neo4j Browser or cypher-shell to validate the graph.
// =============================================================================


// ---------------------------------------------------------------------------
// 1. NODE & RELATIONSHIP COUNTS
// ---------------------------------------------------------------------------

MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count ORDER BY count DESC;

MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count ORDER BY count DESC;


// ---------------------------------------------------------------------------
// 2. ORPHAN / INTEGRITY CHECKS
// ---------------------------------------------------------------------------

// Videos with no creator
MATCH (v:Video) WHERE NOT (v)-[:CREATED_BY]->(:User)
RETURN count(v) AS videos_without_creator;

// Sessions with no viewed or skipped video
MATCH (s:UserSession)
WHERE NOT (s)-[:VIEWED]->(:Video) AND NOT (s)-[:SKIPPED]->(:Video)
RETURN count(s) AS empty_sessions;

// Comments not linked to a user or video
MATCH (c:Comment)
WHERE NOT (c)-[:WRITTEN_BY]->(:User) OR NOT (c)-[:ON_VIDEO]->(:Video)
RETURN count(c) AS orphaned_comments;

// Users following themselves
MATCH (u:User)-[:FOLLOWS]->(u)
RETURN count(u) AS self_follows;


// ---------------------------------------------------------------------------
// 3. DEGREE DISTRIBUTIONS
// ---------------------------------------------------------------------------

// Average in-degree per rel type
MATCH (u:User)
OPTIONAL MATCH ()-[:FOLLOWS]->(u)
WITH u, count(*) AS in_follows
RETURN
  min(in_follows)        AS min_followers,
  max(in_follows)        AS max_followers,
  round(avg(in_follows), 1) AS avg_followers,
  percentileCont(in_follows, 0.5)  AS median_followers,
  percentileCont(in_follows, 0.95) AS p95_followers;

// Videos per session (histogram buckets)
MATCH (s:UserSession)-[:VIEWED]->(v:Video)
WITH s, count(v) AS n_viewed
RETURN
  CASE
    WHEN n_viewed <= 5  THEN '1-5'
    WHEN n_viewed <= 15 THEN '6-15'
    WHEN n_viewed <= 30 THEN '16-30'
    WHEN n_viewed <= 60 THEN '31-60'
    ELSE '60+'
  END AS bucket,
  count(*) AS sessions
ORDER BY bucket;


// ---------------------------------------------------------------------------
// 4. COMPLETION RATE DISTRIBUTION
// ---------------------------------------------------------------------------

MATCH ()-[r:VIEWED]->()
RETURN
  CASE
    WHEN r.completion_rate < 0.15 THEN 'skip-level (<15%)'
    WHEN r.completion_rate < 0.40 THEN 'partial (15-40%)'
    WHEN r.completion_rate < 0.80 THEN 'engaged (40-80%)'
    WHEN r.completion_rate < 1.00 THEN 'completed (80-100%)'
    ELSE 'replay (>100%)'
  END AS bucket,
  count(*) AS views,
  round(count(*) * 100.0 / 306206, 1) AS pct
ORDER BY bucket;


// ---------------------------------------------------------------------------
// 5. TOPIC INTEREST SCORE DISTRIBUTION
// ---------------------------------------------------------------------------

MATCH ()-[r:INTERESTED_IN_TOPIC]->()
RETURN
  CASE
    WHEN r.topic_score <= 0.1 THEN '0.0-0.1'
    WHEN r.topic_score <= 0.3 THEN '0.1-0.3'
    WHEN r.topic_score <= 0.6 THEN '0.3-0.6'
    WHEN r.topic_score <= 0.8 THEN '0.6-0.8'
    ELSE '0.8-1.0'
  END AS bucket,
  count(*) AS edges
ORDER BY bucket;


// ---------------------------------------------------------------------------
// 6. TOP CREATORS BY FOLLOWERS
// ---------------------------------------------------------------------------

MATCH (c:Creator)<-[:FOLLOWS]-(f:User)
WITH c, count(f) AS follower_count
ORDER BY follower_count DESC LIMIT 10
RETURN c.username, c.user_id, follower_count,
       coalesce(c.pagerank_score, 'not computed') AS pagerank_score;


// ---------------------------------------------------------------------------
// 7. TOP VIDEOS BY ENGAGEMENT
// ---------------------------------------------------------------------------

MATCH (v:Video)
OPTIONAL MATCH ()-[:VIEWED]->(v)
OPTIONAL MATCH ()-[:LIKED]->(v)
WITH v, count { ()-[:VIEWED]->(v) } AS views, count { ()-[:LIKED]->(v) } AS likes
WITH v, views, likes,
     CASE WHEN views > 0 THEN round(likes * 1.0 / views, 3) ELSE 0 END AS like_rate
ORDER BY views DESC LIMIT 10
RETURN v.video_id, left(v.description, 60) AS description, views, likes, like_rate;


// ---------------------------------------------------------------------------
// 8. SAMPLE PATH — SHARED INTEREST BRIDGE BETWEEN TWO USERS
// ---------------------------------------------------------------------------
// Finds the shortest path between two random users through shared topics.
// Replace $user_a and $user_b with actual user IDs.

MATCH path = shortestPath(
  (a:User {user_id: $user_a})-[*..6]-(b:User {user_id: $user_b})
)
RETURN path, length(path) AS hops;


// ---------------------------------------------------------------------------
// 9. COMMUNITY SUMMARY (after GDS Louvain writeback)
// ---------------------------------------------------------------------------

MATCH (u:User)
WHERE u.community_id IS NOT NULL
WITH u.community_id AS community, count(*) AS members
ORDER BY members DESC LIMIT 10
RETURN community, members;

// Top topic per community
MATCH (u:User)-[ti:INTERESTED_IN_TOPIC]->(t:Topic)
WHERE u.community_id IS NOT NULL
WITH u.community_id AS community, t.name AS topic, sum(ti.topic_score) AS total_affinity
ORDER BY community, total_affinity DESC
WITH community, collect({topic: topic, score: total_affinity})[0] AS top_topic
RETURN community, top_topic.topic AS dominant_topic, round(top_topic.score, 2) AS affinity
ORDER BY affinity DESC LIMIT 10;


// ---------------------------------------------------------------------------
// 10. GDS ANALYTICS RESULTS SUMMARY (after gds_runner.py)
// ---------------------------------------------------------------------------

MATCH (u:User)
RETURN
  count(u)                                          AS total_users,
  count(u.community_id)                             AS users_with_community,
  count(u.louvain_community)                        AS users_with_louvain,
  round(avg(coalesce(u.betweenness_centrality, 0)), 4) AS avg_betweenness,
  count(u.node2vec_embedding)                       AS users_with_embedding;

MATCH (v:Video)
RETURN
  count(v)                                 AS total_videos,
  count(v.view_pagerank)                   AS videos_with_pagerank,
  count(v.node2vec_embedding)              AS videos_with_embedding,
  round(avg(coalesce(v.view_pagerank, 0)), 6) AS avg_view_pagerank;
