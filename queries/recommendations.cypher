// =============================================================================
// recommendations.cypher
// =============================================================================
// Recommendation queries for the TikTok interest graph.
// Each query is parameterised; pass $user_id (and other params) at call time.
//
// Run from Neo4j Browser, cypher-shell, or the Python driver:
//   session.run(query, user_id="U001", ...)
// =============================================================================


// ---------------------------------------------------------------------------
// 1. CONTENT-BASED RECOMMENDATION
// ---------------------------------------------------------------------------
// Given a user, find videos they have NOT yet watched, ranked by how well the
// video's topic, entity, and hashtag taxonomy overlaps with the user's
// accumulated interest scores.  Higher relevance_score = better match.
//
// Signals combined (weighted sum):
//   • INTERESTED_IN_TOPIC  × 1.0
//   • INTERESTED_IN_ENTITY × 0.6   (entity is more niche than topic)
//   • INTERESTED_IN_HASHTAG × 0.4
//
// :param user_id  — the target user
// :param limit    — number of results (default 20)
// ---------------------------------------------------------------------------

MATCH (u:User {user_id: $user_id})

// Already-seen video ids for exclusion
OPTIONAL MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(seen:Video)
WITH u, collect(seen.video_id) AS seen_ids

// Topic signal
MATCH (u)-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)
WHERE NOT v.video_id IN seen_ids

WITH u, seen_ids, v, sum(ti.topic_score * 1.0) AS topic_signal

// Entity signal
OPTIONAL MATCH (u)-[ei:INTERESTED_IN_ENTITY]->(e:Entity)<-[:MENTIONS]-(v)
WITH u, seen_ids, v, topic_signal, sum(coalesce(ei.entity_score, 0) * 0.6) AS entity_signal

// Hashtag signal
OPTIONAL MATCH (u)-[hi:INTERESTED_IN_HASHTAG]->(h:Hashtag)<-[:HAS_HASHTAG]-(v)
WITH v,
     topic_signal + entity_signal + sum(coalesce(hi.hashtag_score, 0) * 0.4) AS relevance_score

ORDER BY relevance_score DESC
LIMIT $limit

RETURN
  v.video_id        AS video_id,
  v.description     AS description,
  v.play_count      AS play_count,
  round(relevance_score, 4) AS relevance_score;


// ---------------------------------------------------------------------------
// 2. COLLABORATIVE FILTERING — SIMILAR USERS
// ---------------------------------------------------------------------------
// Find users with overlapping INTERESTED_IN_TOPIC vectors (cosine similarity
// proxy: dot product of normalised scores), then surface videos those similar
// users LIKED that our target user has not yet seen.
//
// Pre-requisite: run the GDS NodeSimilarity algorithm first and persist
// SIMILAR_TO relationships (see analysis/gds_runner.py).  This query reads
// from those pre-computed edges.
//
// :param user_id   — target user
// :param min_sim   — minimum similarity score to consider (default 0.3)
// :param limit     — results to return (default 20)
// ---------------------------------------------------------------------------

MATCH (u:User {user_id: $user_id})-[sim:SIMILAR_TO]->(peer:User)
WHERE sim.similarity >= $min_sim

OPTIONAL MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(seen:Video)
WITH u, peer, sim.similarity AS sim_score, collect(DISTINCT seen.video_id) AS seen_ids

MATCH (peer)-[:HAS_SESSION]->(:UserSession)-[:LIKED]->(v:Video)
WHERE NOT v.video_id IN seen_ids

WITH v,
     sum(sim_score) AS collab_score,
     count(DISTINCT peer) AS liked_by_peers

ORDER BY collab_score DESC
LIMIT $limit

RETURN
  v.video_id        AS video_id,
  v.description     AS description,
  liked_by_peers,
  round(collab_score, 4) AS collab_score;


// ---------------------------------------------------------------------------
// 3. TRENDING VIDEOS — ROLLING 7-DAY WINDOW
// ---------------------------------------------------------------------------
// Videos with the highest engagement signal in the most recent 7 days of
// sessions in the dataset.  Engagement = view_count × avg(completion_rate)
// with a like multiplier.
//
// No parameters required.
// ---------------------------------------------------------------------------

MATCH (s:UserSession)-[v:VIEWED]->(vid:Video)
WHERE s.start_date >= (
  datetime({epochMillis: apoc.date.currentTimestamp()}) - duration({days: 7})
)
WITH vid,
     count(v)              AS view_count,
     avg(v.completion_rate) AS avg_cr

OPTIONAL MATCH (:UserSession)-[:LIKED]->(vid)
WITH vid, view_count, avg_cr, count(*) AS like_count

WITH vid,
     view_count * avg_cr * (1.0 + 0.3 * like_count / (view_count + 1)) AS trending_score

ORDER BY trending_score DESC
LIMIT 20

RETURN
  vid.video_id    AS video_id,
  vid.description AS description,
  view_count,
  round(avg_cr, 3)         AS avg_completion,
  like_count,
  round(trending_score, 4) AS trending_score;


// ---------------------------------------------------------------------------
// 4. CREATOR RECOMMENDATION
// ---------------------------------------------------------------------------
// Suggest creators the user should follow, based on:
//   • Topic overlap between the creator's videos and the user's interests
//   • How engaged the user's existing follows are with that creator
//     (engagement_score on FOLLOWS edges the user's follows have)
//   • The creator's overall PageRank centrality (pre-computed by GDS)
//
// :param user_id — target user
// :param limit   — results to return (default 10)
// ---------------------------------------------------------------------------

MATCH (u:User {user_id: $user_id})

// Exclude already-followed creators
OPTIONAL MATCH (u)-[:FOLLOWS]->(already:Creator)
WITH u, collect(already.user_id) AS following_ids

// Topic-matching creators
MATCH (u)-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)<-[:CREATED_BY]-(c:Creator)
WHERE NOT c.user_id IN following_ids
  AND c.user_id <> u.user_id

WITH u, c, sum(ti.topic_score) AS topic_match, following_ids

// Boost by followers of people you follow (social proof)
OPTIONAL MATCH (u)-[:FOLLOWS]->(friend:User)-[fol:FOLLOWS]->(c)
WITH c,
     topic_match,
     coalesce(sum(fol.engagement_score), 0) AS social_boost,
     coalesce(c.pagerank_score, 0) AS centrality

ORDER BY (topic_match + social_boost * 0.5 + centrality * 0.3) DESC
LIMIT $limit

RETURN
  c.user_id        AS creator_id,
  c.username       AS username,
  round(topic_match, 3)    AS topic_match,
  round(social_boost, 3)   AS social_boost,
  round(centrality, 4)     AS centrality;


// ---------------------------------------------------------------------------
// 5. REAL-TIME INTEREST SCORE UPDATE (FEEDBACK LOOP)
// ---------------------------------------------------------------------------
// Called after every new VIEWED interaction is recorded.
// Updates INTERESTED_IN_TOPIC, INTERESTED_IN_ENTITY, and INTERESTED_IN_HASHTAG
// based on the watch completion_rate of the just-watched video.
//
// Delta rules (matching generators/interactions.py):
//   completion >= 0.8  →  +0.5 × completion_rate   (strong positive)
//   completion < 0.15  →  -0.30                      (skip signal)
//   otherwise          →  +0.10                      (partial watch)
//   is_liked = true    →  +0.70 bonus
//   is_commented = true → +0.60 bonus
//   is_reposted = true  → +0.80 bonus
//
// :param user_id         — user who watched
// :param video_id        — video that was watched
// :param completion_rate — fraction watched [0, 1]
// :param is_liked        — boolean
// :param is_commented    — boolean
// :param is_reposted     — boolean
// ---------------------------------------------------------------------------

WITH
  CASE
    WHEN $completion_rate >= 0.8 THEN 0.5 * $completion_rate
    WHEN $completion_rate < 0.15 THEN -0.30
    ELSE 0.10
  END
  + CASE WHEN $is_liked     THEN 0.70 ELSE 0.0 END
  + CASE WHEN $is_commented THEN 0.60 ELSE 0.0 END
  + CASE WHEN $is_reposted  THEN 0.80 ELSE 0.0 END
  AS delta

MATCH (u:User {user_id: $user_id})
MATCH (v:Video {video_id: $video_id})

// Topic update
MATCH (v)-[:IS_ABOUT]->(t:Topic)
MERGE (u)-[ti:INTERESTED_IN_TOPIC]->(t)
  ON CREATE SET ti.topic_score = 0.0
SET ti.topic_score = ti.topic_score + delta

// Entity update (half-weight)
WITH u, v, delta
OPTIONAL MATCH (v)-[:MENTIONS]->(e:Entity)
FOREACH (e IN CASE WHEN e IS NOT NULL THEN [e] ELSE [] END |
  MERGE (u)-[ei:INTERESTED_IN_ENTITY]->(e)
    ON CREATE SET ei.entity_score = 0.0
  SET ei.entity_score = ei.entity_score + delta * 0.5
)

// Hashtag update (quarter-weight)
WITH u, v, delta
OPTIONAL MATCH (v)-[:HAS_HASHTAG]->(h:Hashtag)
FOREACH (h IN CASE WHEN h IS NOT NULL THEN [h] ELSE [] END |
  MERGE (u)-[hi:INTERESTED_IN_HASHTAG]->(h)
    ON CREATE SET hi.hashtag_score = 0.0
  SET hi.hashtag_score = hi.hashtag_score + delta * 0.25
)

RETURN u.user_id AS user_id, $video_id AS video_id, delta AS applied_delta;
