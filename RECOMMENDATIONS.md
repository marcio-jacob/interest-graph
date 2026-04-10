# Recommendation Simulation — `itskevinofficial`

> **Goal**: Given a real user with 40 sessions and 134 likes, generate and compare the top-5 next-video recommendations from every engine we built, and explain what each strategy surfaces and why.

---

## User Profile

| Field | Value |
|---|---|
| User ID | `6c0f0b07-ce08-4f2b-b9f6-e323acdefc7e` |
| Username | **itskevinofficial** |
| Country | 🇺🇸 United States |
| Louvain Community | 1007 |
| Total Sessions | 40 (max in dataset) |
| Total Likes | 134 |
| PageRank (follow graph) | 0.6005 |
| Betweenness Centrality | 16.72 |

### Interest Profile

Accumulated from all 40 sessions of watch/like/comment/repost behaviour, normalised to `[0, 1]`:

| Topic | Interest Score |
|---|---|
| Technology & Science | **1.000** ← dominant |
| Lifestyle & Vlog | 0.171 |
| Comedy & Entertainment | 0.133 |
| Music & Dance | 0.107 |
| Cooking & Food | 0.099 |
| Fashion & Beauty | 0.089 |
| *(6 more topics)* | *< 0.08* |

Technology & Science is saturated at 1.0 — this user watches tech content almost exclusively to completion and actively likes/comments. All other topics are secondary exploration signals.

### Last Session Context

Most-replayed videos in the final session (completion_rate > 1.0 = replayed):

| cr | Topic | Description (truncated) |
|---|---|---|
| 2.98 | Gaming & Esports | *Decision step collection grow pick third Republican…* |
| 2.97 | Lifestyle & Vlog | *Use energy many can discover various nice difficult plant…* |
| 2.94 | Tech & Science | *Structure baby south sport machine understand some of…* |
| 2.87 | Tech & Science | *Simply where never ago among admit not industry finally…* |
| 2.87 | Fashion & Beauty | *Common center consider pay certain it event about fashion…* |

Videos liked in this session: two Tech & Science, one Fitness & Wellness, one Cooking & Food.

---

## GDS Analytics Summary (Used by Engines)

Before running recommendations, `analysis/gds_runner.py` enriched the graph:

| Property | Value |
|---|---|
| `User.community_id` | 12 Louvain communities; Kevin in community 1007 |
| `User.pagerank_score` | Computed on FOLLOWS graph |
| `SIMILAR_TO` edges | 30,000 Jaccard pairs (top-10 per user); Kevin's top peer: `fresh_howto` (similarity 0.908) |
| `Video.view_pagerank` | L1-normalised, based on UserSession→Video VIEWED completions |
| `User.node2vec_embedding` | 64-dim; loss converged 24.8M → 23.5M over 5 epochs |
| `Video.node2vec_embedding` | Same 64-dim space; enables cross-entity similarity |

---

## Engine 1 — Content-Based (Graph Path Traversal)

**Strategy**: Walk `User → INTERESTED_IN_TOPIC → Topic ← IS_ABOUT ← Video` and score each unseen video as:

```
relevance = topic_score × 1.0
           + entity_score × 0.6   (for each matching entity)
           + hashtag_score × 0.4  (for each matching hashtag)
```

**Cypher** (simplified):
```cypher
MATCH (u:User {user_id:$uid})-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)
WHERE NOT EXISTS { MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v) }
WITH u, v, sum(ti.topic_score) AS ts
OPTIONAL MATCH (u)-[ei:INTERESTED_IN_ENTITY]->(e:Entity)<-[:MENTIONS]-(v)
WITH u, v, ts, sum(coalesce(ei.entity_score,0)) AS er
OPTIONAL MATCH (u)-[hi:INTERESTED_IN_HASHTAG]->(hh:Hashtag)<-[:HAS_HASHTAG]-(v)
WITH v, ts + er*0.6 + sum(coalesce(hi.hashtag_score,0))*0.4 AS relevance_score
ORDER BY relevance_score DESC LIMIT 5
```

### Results

| Rank | Score | Topics | Creator | Description |
|---|---|---|---|---|
| 1 | 4.4167 | Technology & Science, Art & Creativity | michael.satisfying | *Action teacher point nearly try type picture walk about art creat…* |
| 2 | 4.3940 | Technology & Science | xoxo.education | *Seek third prove event source this future eat reveal under growth…* |
| 3 | 4.2426 | Technology & Science | itsklausdofficial | *Key might rest society situation middle quickly station assume la…* |
| 4 | 4.2420 | Technology & Science | styleinspowithjoel | *Garden already skill good amount beyond game keep guess about tec…* |
| 5 | 4.1619 | Technology & Science | epic_productivity | *Character amount quite assume become result Republican couple par…* |

### Analysis

The scores are pure multiples of Kevin's dominant topic score (1.0), so all top picks are Technology & Science. The #1 result is boosted slightly by an Art & Creativity cross-topic signal. Entity and hashtag overlap adds small fractional bonuses.

**Strength**: Immediately interpretable; any new video matching Tech & Science rises to the top. No cold-start problem for new videos (only needs a topic tag).

**Weakness**: Locked to Kevin's existing interest graph — it cannot discover genuinely novel content outside his topic profile. All top-5 are the same dominant topic.

---

## Engine 2 — Collaborative Filtering (GDS SIMILAR_TO + Peer Likes)

**Strategy**: Find users with the highest Jaccard similarity to Kevin (pre-computed by GDS NodeSimilarity on `User-INTERESTED_IN_TOPIC` vectors), then surface videos those peers liked that Kevin hasn't seen yet. Score = `Σ(similarity_score)` across all peers who liked each video.

**Cypher** (simplified):
```cypher
MATCH (u:User {user_id:$uid})-[sim:SIMILAR_TO]->(peer:User)
WHERE sim.similarity >= 0.3
WITH u, peer, sim.similarity AS sim_score
MATCH (peer)-[:HAS_SESSION]->(:UserSession)-[:LIKED]->(v:Video)
WHERE NOT EXISTS { MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v) }
WITH v, sum(sim_score) AS collab_score, count(DISTINCT peer) AS peers
ORDER BY collab_score DESC LIMIT 5
```

Kevin's top similar users (Jaccard on topic-interest vectors):

| Peer | Similarity |
|---|---|
| fresh_howto | 0.908 |
| julia.aitools | 0.900 |
| *(others)* | *< 0.9* |

### Results

| Rank | Score | Peers | Topics | Creator | Description |
|---|---|---|---|---|---|
| 1 | 8.109 | 2 | Technology & Science | melissabishop72 | *Ok win gun use customer thousand hospital more about technology s…* |
| 2 | 8.094 | 2 | Technology & Science | styleinspowithjoel | *Garden already skill good amount beyond game keep guess about tec…* |
| 3 | 8.090 | **3** | Gaming & Esports, Tech | itskarlhansofficial | *Be shoulder election face travel every exactly consider use about…* |
| 4 | 5.449 | 1 | Lifestyle & Vlog | mystic_athletics | *Forget song tree trial and administration college about lifestyle…* |
| 5 | 5.426 | 2 | Technology & Science | productivitywithmarjan | *Hour involve focus something run community laugh about technology…* |

### Analysis

The scores (5–8) are substantially higher than Engine 1 because they aggregate similarity weights across multiple peers. The **rank-3 video is the most interesting**: it is a Gaming+Tech hybrid liked by **3 peers**, including users with 0.9+ similarity. This video never appeared in Engine 1 because Kevin's Gaming & Esports interest score is below threshold — yet his closest peers all liked it.

This is collaborative filtering's signature capability: surface content outside a user's stated interest profile by inferring latent taste alignment.

**Strength**: Discovers cross-topic surprises; signals from high-sim peers are strong.

**Weakness**: Requires dense `SIMILAR_TO` graph and active peers. New users with no peers won't get results.

---

## Engine 3 — Node2Vec Embedding Cosine Similarity

**Strategy**: The 64-dimensional Node2Vec embeddings encode each node's position in the full interaction graph (User, Video, Topic, Entity, Hashtag). Unseen videos are ranked by cosine similarity between their embedding vector and Kevin's user embedding.

**Implementation**: Embeddings are fetched from Neo4j in batches of 500, cosine similarity is computed in Python/NumPy (free-tier OOM prevents in-DB computation for full 4,000-video scan).

```python
user_emb /= np.linalg.norm(user_emb)
for video in unseen_videos:
    cos = np.dot(user_emb, video_emb / np.linalg.norm(video_emb))
```

Compared against **3,301 unseen videos**.

### Results

| Rank | Cosine | Topics | Creator | Description |
|---|---|---|---|---|
| 1 | 0.8137 | Lifestyle & Vlog | golden_relatable | *Social son land already loss really other back call between place…* |
| 2 | 0.8042 | Lifestyle & Vlog | rose_vn | *Accept guess result standard policy difference summer reflect eve…* |
| 3 | 0.8034 | Lifestyle & Vlog | patiencennamani55 | *Find teacher agency fast day heavy network use remain listen figu…* |
| 4 | 0.8010 | Lifestyle & Vlog | mystic_dramacheck | *Resource must task animal local stage high reveal investment ago…* |
| 5 | 0.8009 | Lifestyle & Vlog | marionglover_76 | *Product from across own I around finally federal about lifestyle…* |

### Analysis

All top-5 results are **Lifestyle & Vlog** — despite Kevin's dominant interest being Technology & Science. This is the most revealing result in this simulation.

Node2Vec embeds nodes based on **structural proximity in the random-walk sense**: two nodes end up close in embedding space if they appear near each other in short random walks across the graph. Kevin's embedding reflects not just his stated topics but his entire neighbourhood — who he follows, which creators his community watches, which topic nodes he co-walks with via shared videos.

The Lifestyle & Vlog results suggest Kevin's graph neighbourhood overlaps significantly with Lifestyle users, even though his explicit topic scores don't show that. This could represent:
- Accounts Kevin follows that bridge Tech and Lifestyle content
- Videos with dual topic tags (e.g., Tech tutorial styled as a vlog)
- Structural community effects: community 1007 may be a Tech-Lifestyle hybrid cluster

Cosine values around 0.80 indicate strong structural alignment. These recommendations are **serendipitous** rather than obvious — exactly what a GNN-based system adds on top of content/collaborative methods.

**Strength**: Captures latent structural signals invisible to explicit interest scores; enables genuine discovery and serendipity.

**Weakness**: Embeddings require periodic retraining (currently trained once). Cosine similarity in a 64-dim space can't distinguish topic nuance from neighbourhood effects.

---

## Engine 4 — Trending (Engagement × Completion × Like Boost)

**Strategy**: Global trending signal — not personalised. Ranks all videos by:
```
trending_score = view_count × avg(completion_rate) × (1 + 0.3 × like_count / view_count)
```

### Results

| Rank | Score | Topics | Creator | Description |
|---|---|---|---|---|
| 1 | 143.3 | Gaming & Esports, Comedy | anthony.gains | *Sister TV these social rather clear job across street themselves…* |
| 2 | 133.0 | Comedy, Art & Creativity | elizabeth.sciencefact | *Western move enough enter sometimes board actually national usual…* |
| 3 | 132.5 | Comedy & Entertainment | styleinspowithlori | *Meeting until over with officer news figure about comedy entertai…* |
| 4 | 128.2 | Comedy & Entertainment | benjaminobi_32 | *Everybody fund history decide argue record space really about com…* |
| 5 | 128.0 | Comedy & Entertainment | theworkouteater | *Three happen statement where camera sign free democratic raise ma…* |

### Analysis

Trending is dominated by Comedy & Entertainment — these are the globally high-completion, high-like videos. Almost none overlap with Kevin's primary interest (Technology & Science). This is the classic "what's viral" feed, useful as a fallback slot in a mixed feed but not personalised at all.

**Use case**: Fill ~10–15% of the feed with trending content to avoid pure echo-chamber personalisation and expose users to cross-category viral moments.

---

## Engine 5 — Creator Recommendation (PageRank + Topic Match + Social Proof)

**Strategy**: Find creators whose videos match Kevin's topic interests, boosted by social proof (his follows also follow them) and creator-level PageRank centrality:
```
score = topic_match + social_boost × 0.5 + pagerank × 0.3
```

### Results

| Rank | Creator | Top Topic | Topic Match | Social Boost | PageRank |
|---|---|---|---|---|---|
| 1 | **sergio_mx** | comedy_entertainment | 17.353 | — | 8.523 |
| 2 | **theyogaeater** | comedy_entertainment | 16.616 | — | 6.199 |
| 3 | **ludger_de** | comedy_entertainment | 16.915 | — | 2.977 |
| 4 | **thenightroutinemaker** | comedy_entertainment | 14.815 | — | 5.719 |
| 5 | **sportsclipwithguillermo** | comedy_entertainment | 12.353 | — | 7.759 |

### Analysis

The creator scores are high (12–17) because `topic_match` is a **sum across all matching topics × Kevin's interest weights**. Comedy creators with broad content libraries match many of Kevin's minor-but-nonzero interests (Lifestyle, Music, Cooking), and those small matches accumulate. `sergio_mx` tops this list despite being a comedy creator because they are the most central creator in the follow graph (PageRank 8.52) and their content library is large and diverse enough to match across multiple of Kevin's interests.

This is a valid recommendation for a "trending creator to follow" slot, but a strict tech-content filter (e.g., only recommend creators whose `top_topic = 'technology_science'`) would improve precision.

---

## Cross-Engine Comparison

| Engine | Personalised | Discovery | Interpretable | Cold-start OK |
|---|:---:|:---:|:---:|:---:|
| Content-Based | ✓ (topic scores) | ✗ (echo chamber) | ✓✓ | ✓ (new videos) |
| Collaborative Filtering | ✓ (via peers) | ✓ (peer surprise) | ✓ | ✗ (needs peers) |
| Node2Vec Cosine | ✓ (structural) | ✓✓ (serendipity) | ✗ (black box) | ✗ (needs training) |
| Trending | ✗ (global) | ✓ (viral) | ✓ | ✓ |
| Creator Rec | ✓ (topic-boosted) | ✓ (new creators) | ✓ | ✓ (new creators) |

### Overlap analysis

- **M1 and M2 share** one video (`Garden already skill good amount…` by styleinspowithjoel) — it's both topic-matched for Kevin and liked by similar peers. High confidence pick.
- **M3 (Node2Vec) has zero overlap** with M1 or M2 — it surfaces an entirely different slice of content, suggesting it genuinely adds diversity.
- **Trending has zero overlap** with personalised methods, as expected.

### Recommended feed composition

A real mixed feed for `itskevinofficial` would blend all signals:

```
Position 1–3  : Collaborative Filtering top picks   (peer-validated + topic match)
Position 4–6  : Content-Based top picks             (high topic relevance)
Position 7–8  : Node2Vec cosine top picks            (structural serendipity)
Position 9    : Trending #1                          (social awareness)
Position 10   : Creator Recommendation               (new creator to follow)
```

---

## Node2Vec Embedding Quality

The Node2Vec training converged over 5 iterations on the full interaction graph (17,700 nodes, ~912K undirected edges):

| Iteration | Loss |
|---|---|
| 1 | 24,844,859 |
| 2 | 23,827,009 |
| 3 | 23,612,416 |
| 4 | 23,544,933 |
| 5 | 23,505,054 |

Loss decreased by ~5.4% — indicating the embeddings learned meaningful structure but would benefit from more iterations (currently set to 5 for speed; 20+ iterations would improve quality at the cost of ~8× longer training time).

These embeddings are stored as `User.node2vec_embedding` and `Video.node2vec_embedding` (float[64]) and serve as input features for the GraphSAGE/RLHF stage (Method 3 in the architecture).

---

## Running This Simulation

All queries used in this report live in [queries/recommendations.cypher](queries/recommendations.cypher). To reproduce:

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Content-based — replace $uid with any user_id
with driver.session() as s:
    results = s.run("""
        MATCH (u:User {user_id:$uid})-[ti:INTERESTED_IN_TOPIC]->(t:Topic)<-[:IS_ABOUT]-(v:Video)
        WHERE NOT EXISTS { MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v) }
        WITH u, v, sum(ti.topic_score) AS ts
        ...
        ORDER BY ts DESC LIMIT 20
    """, uid="6c0f0b07-ce08-4f2b-b9f6-e323acdefc7e").data()
```

For Node2Vec cosine similarity on the free tier (278 MB transaction limit), fetch embeddings in Python and compute cosine similarity client-side (as done in this simulation) rather than using `gds.similarity.cosine()` in Cypher.
