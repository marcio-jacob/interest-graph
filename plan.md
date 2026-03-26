# TikTok Interest Graph — Complete Build Plan

## Project Overview

Build a synthetic TikTok-like social network as a Neo4j property graph, grounded in real behavioral distributions extracted from the KuaiRec 2.0 dataset. The graph will support interest scoring, community detection (GDS), content recommendation, and creator centrality analysis.

---

## Architecture

```
tiktok/
├── plan.md
├── requirements.txt
├── config/
│   ├── params.yaml           # All tuneable scale + generation parameters
│   ├── taxonomy.yaml         # Topics, Countries, Entities, Sounds, Hashtags
│   └── distributions.yaml    # Fitted distributions from KuaiRec (auto-generated)
├── analysis/
│   └── kuairec_analysis.py   # Phase 1: extract + fit distributions from KuaiRec
├── generators/
│   ├── __init__.py
│   ├── base.py               # Seeded RNG base, shared utilities
│   ├── users.py              # User nodes + Faker usernames + follow graph
│   ├── sessions.py           # UserSession nodes + session chaining
│   ├── videos.py             # Video nodes + video statistics
│   ├── taxonomy.py           # Hashtag, Topic, Entity, Sound, Country nodes
│   └── interactions.py       # VIEWED, LIKED, SKIPPED, COMMENTED, interest scores
├── llm/
│   ├── __init__.py
│   ├── client.py             # Ollama HTTP client wrapper
│   ├── prompts.py            # All prompt templates (topic × angle × tone)
│   └── generator.py          # Orchestrates LLM calls → descriptions, comments, names
├── neo4j/
│   ├── __init__.py
│   ├── connection.py         # Neo4j Aura driver + env config
│   ├── schema.py             # Constraints + indexes + Creator label DDL
│   └── loader.py             # Batched UNWIND Cypher upload for all node/rel types
└── main.py                   # Full pipeline orchestrator
```

---

## Dataset: KuaiRec 2.0

**Location**: `/home/spike/.cache/kagglehub/datasets/arashnic/kuairec-recommendation-system-data-density-100/versions/1/KuaiRec 2.0/data/`

### Files and Key Columns

| File | Key Columns We Use |
|------|--------------------|
| `big_matrix.csv` | `user_id`, `video_id`, `play_duration`, `video_duration`, `watch_ratio` |
| `user_features.csv` | `user_id`, `follow_user_num`, `fans_user_num`, `friend_user_num`, `register_days`, `is_video_author` |
| `item_daily_features.csv` | `video_id`, `video_duration`, `like_cnt`, `play_cnt`, `comment_cnt`, `share_cnt`, `download_cnt`, `follow_cnt`, `complete_play_cnt` |
| `social_network.csv` | `user_id`, `friend_list` |

### Distributions to Extract (→ distributions.yaml)

From `big_matrix.csv`:
- `watch_ratio`: Full histogram + percentiles [p10, p25, p50, p75, p90, p95] + beta-mix fit. Note: values >1 mean replay/loop.
- `video_duration`: Histogram in ms + percentiles. Convert to seconds.

From `user_features.csv`:
- `follow_user_num`: Power-law / log-normal fit (following count per user)
- `fans_user_num`: Power-law / log-normal fit (follower count)
- `register_days`: Distribution of account age

From `item_daily_features.csv` (aggregated per video):
- `like_rate = like_cnt / play_cnt`: Mean, std, percentiles
- `comment_rate = comment_cnt / play_cnt`
- `share_rate = share_cnt / play_cnt`
- `download_rate = download_cnt / play_cnt`
- `follow_rate = follow_cnt / play_cnt` (creator gains follower after video)
- `complete_play_rate = complete_play_cnt / play_cnt`

From `social_network.csv`:
- `friends_per_user`: Mean, std, max (social graph density)

---

## Content Taxonomy (Hardcoded in taxonomy.yaml)

### 12 Topics

| ID | Name | Slug |
|----|------|------|
| T01 | Cooking & Food | cooking_food |
| T02 | Gaming & Esports | gaming_esports |
| T03 | Fashion & Beauty | fashion_beauty |
| T04 | Fitness & Wellness | fitness_wellness |
| T05 | Travel & Adventure | travel_adventure |
| T06 | Music & Dance | music_dance |
| T07 | Comedy & Entertainment | comedy_entertainment |
| T08 | Technology & Science | technology_science |
| T09 | Sports & Athletics | sports_athletics |
| T10 | Education & Tutorials | education_tutorials |
| T11 | Art & Creativity | art_creativity |
| T12 | Lifestyle & Vlog | lifestyle_vlog |

### 12 Countries

| ID | Name | ISO | TikTok Affinity Topics |
|----|------|-----|------------------------|
| C01 | United States | US | All topics |
| C02 | Brazil | BR | sports_athletics (football), music_dance, comedy_entertainment |
| C03 | Indonesia | ID | cooking_food, fashion_beauty, lifestyle_vlog |
| C04 | Vietnam | VN | cooking_food, travel_adventure, lifestyle_vlog |
| C05 | Philippines | PH | comedy_entertainment, music_dance, education_tutorials |
| C06 | United Kingdom | GB | fashion_beauty, comedy_entertainment, technology_science |
| C07 | Germany | DE | technology_science, fitness_wellness, education_tutorials |
| C08 | Japan | JP | gaming_esports, art_creativity, cooking_food |
| C09 | Mexico | MX | cooking_food, music_dance, comedy_entertainment |
| C10 | India | IN | education_tutorials, music_dance, comedy_entertainment |
| C11 | South Korea | KR | music_dance (K-pop), fashion_beauty, gaming_esports |
| C12 | Nigeria | NG | music_dance (Afrobeats), comedy_entertainment, lifestyle_vlog |

### Entities per Topic (named entities for video content and NER simulation)

```yaml
cooking_food:
  - {name: "Gordon Ramsay", aliases: ["Chef Ramsay", "Hell's Kitchen"]}
  - {name: "MasterChef", aliases: ["cooking competition"]}
  - {name: "Jamie Oliver", aliases: ["Naked Chef"]}
  - {name: "Michelin Star", aliases: ["fine dining"]}
  - {name: "Street Food", aliases: ["food truck", "hawker"]}

gaming_esports:
  - {name: "Minecraft", aliases: ["MC", "Mojang"]}
  - {name: "League of Legends", aliases: ["LoL", "Riot Games"]}
  - {name: "Twitch", aliases: ["live streaming", "stream"]}
  - {name: "Valorant", aliases: ["Riot FPS"]}
  - {name: "Steam", aliases: ["Valve", "PC gaming"]}

fashion_beauty:
  - {name: "Zara", aliases: ["Inditex", "fast fashion"]}
  - {name: "Sephora", aliases: ["beauty retailer"]}
  - {name: "Vogue", aliases: ["fashion magazine"]}
  - {name: "H&M", aliases: ["sustainable fashion"]}
  - {name: "Gucci", aliases: ["luxury fashion", "GG"]}

fitness_wellness:
  - {name: "CrossFit", aliases: ["WOD", "box gym"]}
  - {name: "Nike", aliases: ["Just Do It", "swoosh"]}
  - {name: "Pilates", aliases: ["reformer", "core workout"]}
  - {name: "Adidas", aliases: ["three stripes"]}
  - {name: "Whoop", aliases: ["fitness tracker", "recovery band"]}

travel_adventure:
  - {name: "Bali", aliases: ["Bali Indonesia", "Island of Gods"]}
  - {name: "Santorini", aliases: ["Greek islands"]}
  - {name: "Tokyo", aliases: ["Japan", "Shibuya"]}
  - {name: "Airbnb", aliases: ["short term rental"]}
  - {name: "Machu Picchu", aliases: ["Peru", "Inca trail"]}

music_dance:
  - {name: "BTS", aliases: ["Bangtan", "K-pop", "HYBE"]}
  - {name: "Bad Bunny", aliases: ["El Conejo Malo", "reggaeton"]}
  - {name: "Coachella", aliases: ["music festival", "desert festival"]}
  - {name: "Drake", aliases: ["Drizzy", "OVO"]}
  - {name: "BLACKPINK", aliases: ["Blink", "YG Entertainment"]}

comedy_entertainment:
  - {name: "MrBeast", aliases: ["Jimmy Donaldson", "Beast Philanthropy"]}
  - {name: "Netflix", aliases: ["streaming", "binge-watch"]}
  - {name: "Saturday Night Live", aliases: ["SNL"]}
  - {name: "Vine", aliases: ["6 second videos"]}
  - {name: "Charli D'Amelio", aliases: ["TikTok star"]}

technology_science:
  - {name: "OpenAI", aliases: ["ChatGPT", "GPT-4"]}
  - {name: "Tesla", aliases: ["Elon Musk", "EV", "Autopilot"]}
  - {name: "Apple", aliases: ["iPhone", "iOS", "Tim Cook"]}
  - {name: "SpaceX", aliases: ["Starship", "Falcon 9"]}
  - {name: "Meta", aliases: ["Facebook", "Instagram", "Zuckerberg"]}

sports_athletics:
  - {name: "Cristiano Ronaldo", aliases: ["CR7", "Siuuu"]}
  - {name: "Lionel Messi", aliases: ["GOAT", "Leo Messi", "Inter Miami"]}
  - {name: "LeBron James", aliases: ["King James", "The King"]}
  - {name: "FIFA World Cup", aliases: ["World Cup", "Qatar 2022"]}
  - {name: "NBA", aliases: ["basketball", "playoffs"]}

education_tutorials:
  - {name: "Khan Academy", aliases: ["free education"]}
  - {name: "Duolingo", aliases: ["language app", "green owl"]}
  - {name: "TED Talk", aliases: ["TED", "ideas worth spreading"]}
  - {name: "Coursera", aliases: ["online learning"]}
  - {name: "MIT OpenCourseWare", aliases: ["MIT OCW", "free MIT"]}

art_creativity:
  - {name: "Procreate", aliases: ["digital art", "iPad art"]}
  - {name: "Banksy", aliases: ["street art", "graffiti"]}
  - {name: "Adobe Creative Suite", aliases: ["Photoshop", "Illustrator", "Adobe"]}
  - {name: "Louvre", aliases: ["Paris museum", "Mona Lisa"]}
  - {name: "Canva", aliases: ["design tool", "free design"]}

lifestyle_vlog:
  - {name: "Morning Routine", aliases: ["5am club", "productive morning"]}
  - {name: "GRWM", aliases: ["Get Ready With Me"]}
  - {name: "NYC", aliases: ["New York City", "Big Apple"]}
  - {name: "Aesthetic", aliases: ["cottagecore", "dark academia"]}
  - {name: "Manifestation", aliases: ["law of attraction", "vision board"]}
```

### 30 Sounds

```yaml
sounds:
  - {song_id: S001, song_name: "Dynamite", singer: "BTS", genre: "K-pop", country: "KR"}
  - {song_id: S002, song_name: "Pink Venom", singer: "BLACKPINK", genre: "K-pop", country: "KR"}
  - {song_id: S003, song_name: "Hype Boy", singer: "NewJeans", genre: "K-pop", country: "KR"}
  - {song_id: S004, song_name: "Tití Me Preguntó", singer: "Bad Bunny", genre: "Reggaeton", country: "MX"}
  - {song_id: S005, song_name: "Provenza", singer: "Karol G", genre: "Reggaeton", country: "CO"}
  - {song_id: S006, song_name: "God's Plan", singer: "Drake", genre: "Hip-Hop", country: "US"}
  - {song_id: S007, song_name: "SICKO MODE", singer: "Travis Scott", genre: "Hip-Hop", country: "US"}
  - {song_id: S008, song_name: "As It Was", singer: "Harry Styles", genre: "Pop", country: "GB"}
  - {song_id: S009, song_name: "Levitating", singer: "Dua Lipa", genre: "Pop", country: "GB"}
  - {song_id: S010, song_name: "drivers license", singer: "Olivia Rodrigo", genre: "Pop", country: "US"}
  - {song_id: S011, song_name: "Titanium", singer: "David Guetta ft. Sia", genre: "EDM", country: "FR"}
  - {song_id: S012, song_name: "Blinding Lights", singer: "The Weeknd", genre: "Synth-Pop", country: "CA"}
  - {song_id: S013, song_name: "Girl From Rio", singer: "Anitta", genre: "Brazilian Pop", country: "BR"}
  - {song_id: S014, song_name: "Funk Carioca", singer: "MC Bin Laden", genre: "Funk", country: "BR"}
  - {song_id: S015, song_name: "Last Last", singer: "Burna Boy", genre: "Afrobeats", country: "NG"}
  - {song_id: S016, song_name: "Essence", singer: "Wizkid ft. Tems", genre: "Afrobeats", country: "NG"}
  - {song_id: S017, song_name: "Butter", singer: "BTS", genre: "K-pop", country: "KR"}
  - {song_id: S018, song_name: "Moonlight Sonata", singer: "Beethoven", genre: "Classical", country: "DE"}
  - {song_id: S019, song_name: "Jai Ho", singer: "AR Rahman", genre: "Bollywood", country: "IN"}
  - {song_id: S020, song_name: "Gangnam Style", singer: "PSY", genre: "K-pop", country: "KR"}
  - {song_id: S021, song_name: "Con Calma", singer: "Daddy Yankee", genre: "Reggaeton", country: "MX"}
  - {song_id: S022, song_name: "Die With A Smile", singer: "Lady Gaga & Bruno Mars", genre: "Pop", country: "US"}
  - {song_id: S023, song_name: "Bohemian Rhapsody", singer: "Queen", genre: "Rock", country: "GB"}
  - {song_id: S024, song_name: "505", singer: "Arctic Monkeys", genre: "Indie Rock", country: "GB"}
  - {song_id: S025, song_name: "Less Is More", singer: "Tame Impala", genre: "Psychedelic", country: "AU"}
  - {song_id: S026, song_name: "Rasputin", singer: "Boney M", genre: "Disco", country: "DE"}
  - {song_id: S027, song_name: "Running Up That Hill", singer: "Kate Bush", genre: "Art Pop", country: "GB"}
  - {song_id: S028, song_name: "STAY", singer: "The Kid LAROI & Justin Bieber", genre: "Pop", country: "AU"}
  - {song_id: S029, song_name: "Sweet Child O' Mine", singer: "Guns N' Roses", genre: "Rock", country: "US"}
  - {song_id: S030, song_name: "Counting Stars", singer: "OneRepublic", genre: "Pop Rock", country: "US"}
```

### Topic → Hashtag Sets (10 hashtags per topic, used for video generation)

```
cooking_food:    #foodtok #cooking #recipe #mealprep #foodie #homecooking #chef #easyrecipe #foodlover #kitchenhacks
gaming_esports:  #gaming #gamer #gameplay #pcgaming #esports #streamer #gamingsetup #noob2pro #fps #rpg
fashion_beauty:  #ootd #fashiontok #grwm #makeup #skincare #styleinspo #thriftflip #nailart #aesthetic #haul
fitness_wellness:#fittok #gym #workout #gains #homeworkout #mentalhealth #yoga #proteinrecipe #fitcheck #noexcuses
travel_adventure:#traveltok #wanderlust #bucketlist #solotravel #travelvibes #hidden_gem #roadtrip #backpacker #travelcouple #explore
music_dance:     #fyp #dancechallenge #newmusic #dancecover #playlist #musiclover #singwithme #vocalcover #trending #vibes
comedy_entertainment:#funnyvideo #comedy #laughing #relatable #storytime #dramacheck #prank #roleplay #skit #blowthisup
technology_science:  #techtok #coding #aitools #programmer #techreview #sciencefact #innovation #robotics #softwareengineer #python
sports_athletics:#sports #football #basketball #training #athlete #sportsclip #matchday #skills #highlights #fitness
education_tutorials: #learntok #studywithme #tutorial #learnontiktok #factcheck #studytips #howto #explained #mindblown #dyk
art_creativity:  #arttok #digitalart #drawing #satisfying #diy #craftok #artistsoftiktok #procreate #painting #creative
lifestyle_vlog:  #dayinmylife #vlog #morningroutine #lifestyletok #manifestation #aesthetic #productivity #selfcare #grwm #nightroutine
```

### Topic Cross-Links (Entity → Topic)

```
Cristiano Ronaldo  → sports_athletics, lifestyle_vlog
BTS               → music_dance, lifestyle_vlog
MrBeast           → comedy_entertainment, education_tutorials
OpenAI / ChatGPT  → technology_science, education_tutorials
Coachella         → music_dance, lifestyle_vlog, travel_adventure
Nike              → fitness_wellness, sports_athletics, fashion_beauty
Gordon Ramsay     → cooking_food, comedy_entertainment
Bali              → travel_adventure, lifestyle_vlog, cooking_food
```

---

## Scale Parameters (config/params.yaml)

```yaml
seed: 42

scale:
  num_users: 500            # Total users (including creators)
  creator_fraction: 0.15    # 15% of users are creators (post videos)
  num_videos: 2000          # Total videos
  num_countries: 12
  num_topics: 12
  num_hashtags: 120         # 10 per topic
  num_entities: 60          # 5 per topic
  num_sounds: 30
  num_comments_per_video: [1, 15]  # range

sessions:
  sessions_per_user: [3, 25]       # range, log-normal
  videos_per_session: [5, 50]      # range, from KuaiRec
  session_duration_minutes: [3, 90]

dates:
  sim_start: "2023-01-01"
  sim_end: "2024-12-31"

# Interaction rates — overridden by distributions.yaml after KuaiRec analysis
interactions:
  like_rate: 0.35           # P(like | viewed)
  comment_rate: 0.05        # P(comment | viewed)
  repost_rate: 0.02         # P(repost | viewed)
  skip_threshold: 0.15      # watch_ratio below this = SKIPPED

# Interest score update weights (feedback loop)
interest:
  watch_positive_threshold: 0.8   # watch_ratio >= 0.8 = positive signal
  skip_decay: -0.3                 # score delta for skip
  complete_boost: +0.5             # score delta for completion
  like_boost: +0.7                 # score delta for like
  comment_boost: +0.6
  repost_boost: +0.8

# Social graph
social:
  follow_power_law_alpha: 2.1      # Zipf alpha for follower distribution
  max_following: 2000
  engagement_score_range: [0.1, 1.0]

# Country distribution for users (weights)
country_weights:
  US: 0.20
  BR: 0.10
  IN: 0.10
  ID: 0.08
  PH: 0.06
  GB: 0.08
  MX: 0.07
  JP: 0.07
  KR: 0.07
  DE: 0.06
  VN: 0.05
  NG: 0.06
```

---

## Neo4j Schema (Cypher)

### Constraints

```cypher
CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE;
CREATE CONSTRAINT session_id IF NOT EXISTS FOR (s:UserSession) REQUIRE s.session_id IS UNIQUE;
CREATE CONSTRAINT video_id IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE;
CREATE CONSTRAINT hashtag_id IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.hashtag_id IS UNIQUE;
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE;
CREATE CONSTRAINT sound_id IF NOT EXISTS FOR (s:Sound) REQUIRE s.song_id IS UNIQUE;
CREATE CONSTRAINT country_id IF NOT EXISTS FOR (c:Country) REQUIRE c.country_id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE;
CREATE CONSTRAINT comment_id IF NOT EXISTS FOR (c:Comment) REQUIRE c.comment_id IS UNIQUE;
```

### Indexes

```cypher
CREATE INDEX user_country IF NOT EXISTS FOR (u:User) ON (u.country_id);
CREATE INDEX video_topic IF NOT EXISTS FOR (v:Video) ON (v.posted_at);
CREATE INDEX session_date IF NOT EXISTS FOR (s:UserSession) ON (s.start_date);
```

---

## LLM Strategy: Generating Diverse Text with Ollama

### Axes of Variation

Each video description / comment is generated by combining:
- **Topic** (12 options)
- **Angle** (7 options): tutorial | reaction | behind_scenes | challenge | review | day_in_life | storytime
- **Tone** (5 options): casual_genz | professional | humorous | motivational | educational
- **Length**: short (15s video → 1-2 sentences) | medium (30-60s → 2-3 sentences) | long (60s+ → 3-5 sentences)
- **Country hint**: adds cultural flavor words (e.g., "saudade" for BR, "kimchi" for KR)

### Faker Username Patterns (by user persona)

```python
# Pattern pool — mixed randomly per user
USERNAME_PATTERNS = [
    "{first}{last}{nn}",            # johnsmith42 — most common
    "{first}.{topic_word}",         # maria.cooks
    "the{noun}{verb}er",            # thefoodlover
    "{topic_word}with{first}",      # cookingwithmaria
    "{adjective}_{noun}",           # cool_panda
    "{first}_{country_adj}",        # yuki_official, pedro_br
    "{noun}vibes{nn}",              # beachvibes99
    "its{first}official",           # itsmaria_official
    "xoxo.{topic_word}",            # xoxo.aesthetic (Gen-Z)
    "{first}{last}_{nn}",           # jsmith_07
]
# Faker locales mapped to countries:
# US/GB → en_US, BR → pt_BR, JP → ja_JP, KR → ko_KR,
# IN → hi_IN, DE → de_DE, MX → es_MX, ID → id_ID
```

### Comment Sentiment Distribution (per topic)

| Topic | Positive% | Neutral% | Negative% |
|-------|-----------|----------|-----------|
| cooking_food | 80 | 15 | 5 |
| gaming_esports | 55 | 25 | 20 |
| fashion_beauty | 70 | 20 | 10 |
| fitness_wellness | 85 | 10 | 5 |
| travel_adventure | 90 | 8 | 2 |
| music_dance | 75 | 15 | 10 |
| comedy_entertainment | 70 | 20 | 10 |
| technology_science | 50 | 35 | 15 |
| sports_athletics | 60 | 20 | 20 |
| education_tutorials | 80 | 15 | 5 |
| art_creativity | 85 | 12 | 3 |
| lifestyle_vlog | 65 | 25 | 10 |

---

## Execution Order: The Prompts

Send these prompts **in order**. Each builds on the previous. Do not skip ahead.

---

### PROMPT 1 — KuaiRec Distribution Analysis

> **Goal**: Load the KuaiRec 2.0 dataset, compute all behavioral distributions we need for realistic fake data, and save them to `config/distributions.yaml`.
>
> Create `analysis/kuairec_analysis.py`. The script must:
>
> **Inputs**: Read these 4 files from the KuaiRec dataset at `/home/spike/.cache/kagglehub/datasets/arashnic/kuairec-recommendation-system-data-density-100/versions/1/KuaiRec 2.0/data/`:
> - `big_matrix.csv` (columns: user_id, video_id, play_duration, video_duration, watch_ratio)
> - `user_features.csv` (columns: user_id, follow_user_num, fans_user_num, friend_user_num, register_days, is_video_author)
> - `item_daily_features.csv` (columns: video_id, date, video_duration, like_cnt, play_cnt, comment_cnt, share_cnt, download_cnt, follow_cnt, complete_play_cnt)
> - `social_network.csv` (columns: user_id, friend_list)
>
> **Computations**:
> 1. From `big_matrix.csv`:
>    - `watch_ratio` percentiles: p5, p10, p25, p50, p75, p90, p95, p99 — clip to [0, 3]
>    - Histogram bins (20 bins from 0 to 3) for watch_ratio
>    - `video_duration` (in seconds = video_duration_ms / 1000) percentiles and histogram
>    - Mean videos viewed per user (total rows / distinct users)
>
> 2. From `user_features.csv`:
>    - `follow_user_num` and `fans_user_num`: mean, std, p50, p90, p99, max
>    - `register_days`: mean, std, percentiles
>    - Fraction of users that are video authors
>
> 3. From `item_daily_features.csv` (aggregate per video_id, sum all daily rows):
>    - Compute per-video: `like_rate = like_cnt / play_cnt`, `comment_rate`, `share_rate`, `download_rate`, `follow_rate`, `complete_rate`
>    - For each rate: mean, std, p50, p75, p90 (filter out videos with play_cnt < 10 to reduce noise)
>
> 4. From `social_network.csv`:
>    - Parse friend_list (it's a string like "[2975]"), compute friends_per_user: mean, std, p90
>    - Fraction of users with at least 1 friend
>
> **Output**: Write `config/distributions.yaml` with all computed values, organized by section (watch_behavior, user_social, video_engagement, social_graph). Also write `config/` directory if it doesn't exist.
>
> Also create `requirements.txt` with: pandas, numpy, scipy, pyyaml, faker, neo4j, requests, tqdm, python-dotenv, kagglehub
>
> Run the script and confirm the YAML was written.

---

### PROMPT 2 — Project Scaffold + Taxonomy

> **Goal**: Create the full project folder structure and the two static config files.
>
> 1. Create all directories: `config/`, `generators/`, `llm/`, `neo4j/`, `analysis/`
>
> 2. Create `config/taxonomy.yaml` with the full content from the plan: all 12 topics (with their hashtag lists and entity lists), all 12 countries (with affinity topic lists), all 30 sounds, the topic cross-links (entity → topic mappings), and the username patterns list. Structure it so Python can load it with `yaml.safe_load()` and access e.g. `taxonomy['topics']`, `taxonomy['countries']`, `taxonomy['sounds']`, `taxonomy['hashtags']`, `taxonomy['entities']`.
>
> 3. Create `config/params.yaml` with all the scale and simulation parameters from the plan (seed, scale, sessions, dates, interactions, interest weights, social, country_weights). These are the master dials — nothing should be hardcoded elsewhere.
>
> 4. Create `generators/__init__.py`, `llm/__init__.py`, `neo4j/__init__.py` as empty init files.
>
> 5. Create `generators/base.py` with:
>    - A `load_config()` function that reads and merges `params.yaml` and `taxonomy.yaml` and `distributions.yaml` into a single dict
>    - A seeded `rng` using `numpy.random.default_rng(seed)` accessible as a module-level singleton
>    - A `weighted_choice(items, weights)` helper
>    - A `clamp(val, lo, hi)` helper
>    - A `date_between(start_str, end_str)` helper using `faker` that returns a `datetime` object
>
> Print a confirmation that all files and directories were created.

---

### PROMPT 3 — User & Session Generators

> **Goal**: Build `generators/users.py` and `generators/sessions.py` — the two modules that create User nodes, Creator labels, UserSession nodes, and the FOLLOWS relationships.
>
> **`generators/users.py`** must:
>
> 1. `generate_users(num_users, params, taxonomy, distributions)` → returns `list[dict]`
>    - Each user dict has all User node attributes: `user_id` (UUID), `username`, `joined_at` (datetime), `followers` (int), `following` (int), `like_count` (int), `average_watch_time` (float), `last_login` (datetime), `country_id` (str), `is_creator` (bool)
>    - `followers` and `following` drawn from log-normal using `fans_user_num` and `follow_user_num` stats from distributions.yaml
>    - `username` generated with a custom `build_username(fake, topic_words, pattern, country)` function that:
>        - Uses `Faker` with the locale matching the user's country (map: BR→pt_BR, JP→ja_JP, KR→ko_KR, IN→hi_IN, DE→de_DE, MX→es_MX, others→en_US)
>        - Picks one of the 10 username patterns from taxonomy.yaml
>        - Replaces `{first}` with `fake.first_name()`, `{last}` with `fake.last_name()`, `{nn}` with a 2-digit number, `{noun}` with a random word from topic vocabulary, `{topic_word}` with a topic slug word, `{adjective}` from a curated adjective list, `{country_adj}` with a country-based suffix (e.g., "br", "jp", "us")
>        - Lowercases, strips spaces, removes non-alphanumeric except `_` and `.`
>        - Guarantees uniqueness (retry on collision)
>    - `is_creator` = True for `creator_fraction` of users (drawn from `user_features` author fraction if available, else params)
>    - `country_id` drawn using `country_weights` from params
>    - `joined_at` and `last_login` derived from register_days distribution and sim date range
>
> 2. `generate_follows(users, params, distributions)` → returns `list[dict]` of `{follower_id, followee_id, engagement_score}`
>    - Build a directed follow graph. Follower counts are pre-assigned; now wire them up.
>    - Use preferential attachment: creators and high-follower users are more likely to be followed
>    - Engagement score: float [0.1, 1.0], positively correlated with completion rate of the followed user's videos
>    - No self-follows, no duplicate pairs
>
> **`generators/sessions.py`** must:
>
> 1. `generate_sessions(users, params)` → returns `list[dict]`
>    - Each session dict: `session_id` (UUID), `user_id`, `start_date` (datetime), `end_date` (datetime)
>    - Number of sessions per user drawn from log-normal clipped to `sessions_per_user` range
>    - Sessions are ordered in time; `PREVIOUS_SESSION` chaining metadata included as `prev_session_id` (None for first session)
>    - Session duration drawn from `session_duration_minutes` range, with heavier tails on weekends
>    - `last_login` on User should match the most recent session's end_date — return a `{user_id → last_session_id}` map as a second return value

---

### PROMPT 4 — Video & Taxonomy Node Generators

> **Goal**: Build `generators/videos.py` and `generators/taxonomy.py`.
>
> **`generators/taxonomy.py`** must:
>
> 1. `generate_topics(taxonomy)` → `list[dict]` — each has `topic_id`, `name`, `slug`
> 2. `generate_countries(taxonomy)` → `list[dict]` — each has `country_id`, `name`, `iso`
> 3. `generate_hashtags(taxonomy)` → `list[dict]` — each has `hashtag_id` (e.g. `HT001`), `name` (e.g. `#foodtok`), `topic_slug`
> 4. `generate_entities(taxonomy)` → `list[dict]` — each has `entity_id`, `name`, `aliases` (list), `topic_slug`
> 5. `generate_sounds(taxonomy)` → `list[dict]` — straight from taxonomy yaml sounds section
> 6. `entity_topic_links(entities, topics)` → `list[dict]` — `{entity_id, topic_id}` pairs including cross-topic links from the plan's entity→topic mapping table
>
> **`generators/videos.py`** must:
>
> 1. `generate_videos(num_videos, users, topics, hashtags, entities, sounds, countries, taxonomy, params, distributions)` → `list[dict]`
>    - Only `is_creator=True` users can author videos
>    - Each video dict has all Video node attributes: `video_id` (UUID), `author_id`, `video_duration` (int, seconds), `posted_at` (datetime), `description` (placeholder string `"PLACEHOLDER:{topic_slug}:{angle}:{tone}"` — filled by LLM later), `likes` (int), `downloads` (int), `shares` (int), `reposts` (int), `comments` (int)
>    - `video_duration` sampled from the video_duration distribution in distributions.yaml (use the histogram bins as a pmf)
>    - `likes`, `downloads`, `shares`, `reposts`, `comments` computed from `like_rate`, `download_rate`, `share_rate`, `comment_rate` × estimated view count, where view count is drawn from a Zipf distribution (viral videos get many more views)
>    - `posted_at` between `sim_start` and `sim_end`; creators post more videos if they have more followers (power law)
>    - Country affinity: creator's country influences which topics they post about
>
> 2. `assign_video_taxonomy(videos, topics, hashtags, entities, sounds, taxonomy, params)` → returns 4 lists of relationship dicts:
>    - `video_hashtags`: `{video_id, hashtag_id}` — 2-5 hashtags per video from the video's topic
>    - `video_entities`: `{video_id, entity_id}` — 0-3 entities per video, biased to topic's entities
>    - `video_sounds`: `{video_id, song_id}` — exactly 1 sound per video, topic-biased
>    - `video_topics`: `{video_id, topic_id}` — 1 primary + 0-1 secondary topic

---

### PROMPT 5 — LLM Text Generator (Ollama)

> **Goal**: Build the `llm/` module to generate realistic video descriptions, comments, and finalize placeholder usernames using a local Ollama model.
>
> **`llm/client.py`**:
> - `OllamaClient` class that calls `http://localhost:11434/api/generate` via `requests`
> - `generate(model, prompt, max_tokens, temperature, stop)` method — returns raw text
> - Default model: read from env `OLLAMA_MODEL` (default: `"llama3.2"`)
> - Handle HTTP errors gracefully, retry once on timeout
> - `is_available()` method — pings `/api/tags` to confirm Ollama is running
>
> **`llm/prompts.py`**:
>
> Define `DESCRIPTION_PROMPTS` as a dict keyed by `(topic_slug, angle, tone)`. The prompts must instruct the model to write a TikTok video description (NOT a caption — the `description` field = the text content/script notes for what the video is about). Each prompt must:
> - Be under 300 tokens of input
> - Produce output between 1 and 4 sentences
> - NOT use hashtags (those are separate nodes)
> - Vary meaningfully across all combinations
>
> Cover at minimum these 35 combinations (5 angles × 7 topics, then fill the rest):
>
> | topic | angle | tone | Prompt hint |
> |-------|-------|------|-------------|
> | cooking_food | tutorial | casual_genz | "write a short description of a TikTok where someone shows how to make [specific dish]. Casual Gen-Z tone, use 'real talk', mention a specific ingredient twist. No hashtags." |
> | cooking_food | reaction | humorous | "write a description of a TikTok where a food lover reacts with shock to an unusual food combo. Funny, slightly dramatic." |
> | gaming_esports | tutorial | educational | "write a description for a TikTok tutorial on getting better at [specific game mechanic]. Educational, direct, include one specific tip." |
> | gaming_esports | storytime | casual_genz | "write a description for a TikTok story about an epic gaming win or embarrassing fail. Casual, exciting, first person." |
> | fashion_beauty | review | professional | "write a description for a TikTok reviewing a specific fashion item or brand. Honest, structured, professional." |
> | fashion_beauty | challenge | casual_genz | "write a description for a TikTok outfit challenge. Trendy, energetic, reference a current aesthetic." |
> | fitness_wellness | tutorial | motivational | "write a description for a TikTok workout tutorial. Motivational, specific muscle group or exercise, 3-word power phrases." |
> | music_dance | challenge | casual_genz | "write a description for a TikTok dance challenge video. Hype, reference a specific song, tag-based community energy." |
> | technology_science | reaction | humorous | "write a description for a TikTok reacting to a mind-blowing tech fact or new AI feature. Humorous shock, relatable tech frustration or excitement." |
>
> Generate the full 84-combination matrix programmatically using `string.Template` or f-strings — do NOT write 84 individual strings. Define base templates per topic + modifiers per (angle, tone) combo. This is important for maintainability.
>
> Also define `COMMENT_PROMPTS` dict keyed by `(topic_slug, sentiment)`:
> - `positive`: enthusiastic, specific reaction to the video content
> - `neutral`: question or mild observation
> - `negative`: constructive criticism or mild complaint (never hate speech)
> Each comment prompt should produce 1 sentence, very varied vocabulary.
>
> **`llm/generator.py`**:
> 1. `fill_video_descriptions(videos, taxonomy, client, params)` → mutates videos list in place, replacing `"PLACEHOLDER:..."` with real LLM-generated text. Batch by topic/angle/tone to reuse prompts. Show tqdm progress.
> 2. `generate_comments(comment_stubs, videos, taxonomy, client)` → returns list of Comment dicts with `comment_id`, `comment_text`, `comment_sentiment` ("positive"/"neutral"/"negative")
> 3. Implement a fallback: if Ollama is not available, use `faker.sentence()` + topic vocabulary injection as a degraded-mode fallback so the rest of the pipeline still runs.

---

### PROMPT 6 — Interaction Generator (Sessions × Videos)

> **Goal**: Build `generators/interactions.py` — the most critical module. This creates the behavioral data: which videos each session watched, liked, skipped, reposted, commented on, and how interest scores are computed.
>
> **`generate_interactions(sessions, videos, users, topics, entities, hashtags, params, distributions)`** → returns a dict with keys:
> - `views`: list of `{session_id, video_id, watch_time, completion_rate}` — the VIEWED relationship
> - `likes`: list of `{session_id, video_id}` — the LIKED relationship
> - `skips`: list of `{session_id, video_id}` — the SKIPPED relationship
> - `reposts`: list of `{session_id, video_id}` — the REPOSTED relationship
> - `comment_stubs`: list of `{session_id, video_id, comment_id, sentiment}` — COMMENTED skeleton (text filled by LLM later)
> - `topic_interests`: list of `{user_id, topic_id, topic_score}` — INTERESTED_IN_TOPIC
> - `entity_interests`: list of `{user_id, entity_id, entity_score}` — INTERESTED_IN_ENTITY
> - `hashtag_interests`: list of `{user_id, hashtag_id, hashtag_score}` — INTERESTED_IN_HASHTAG
>
> **Core algorithm**:
>
> 1. For each session, determine the feed (which videos to show):
>    - First session of a user: weighted random from all videos (cold start), biased by country
>    - Subsequent sessions: 60% interest-matched videos (select by user's current topic scores), 40% random/exploratory
>    - Number of videos per session drawn from `videos_per_session` range using the distribution
>
> 2. For each (session, video) pair in the feed:
>    - Sample `watch_ratio` from the watch_ratio histogram in distributions.yaml (scipy `rv_histogram`)
>    - `watch_time = watch_ratio × video_duration` in seconds
>    - If `watch_ratio < skip_threshold` → SKIPPED (no VIEWED)
>    - Else → VIEWED with `completion_rate = min(1.0, watch_ratio)`
>    - If VIEWED: sample `is_liked` (Bernoulli with `like_rate` × boost if completion_rate > 0.8)
>    - If VIEWED: sample `is_reposted` (Bernoulli with `repost_rate`)
>    - If VIEWED: sample `is_commented` (Bernoulli with `comment_rate` × boost if is_liked)
>
> 3. After all sessions for a user, compute interest scores:
>    - For each topic T: `topic_score = Σ feedback_signal(interaction)` where:
>      - VIEWED completion ≥ 0.8 → +0.5 × completion_rate
>      - VIEWED completion < 0.8 → +0.1
>      - SKIPPED → -0.3
>      - LIKED → +0.7
>      - COMMENTED → +0.6
>      - REPOSTED → +0.8
>    - Normalize all topic scores to [0, 1] per user
>    - Only keep `topic_score > 0.05` to avoid noise edges
>    - Do the same for entities and hashtags (using video→entity and video→hashtag links)
>
> **Important**: Watch time is more important than likes. Completion rate must be stored on every VIEWED edge. This drives the recommendation system.

---

### PROMPT 7 — Neo4j Aura Loader

> **Goal**: Build the `neo4j/` module to upload all generated data to a Neo4j Aura instance in batches.
>
> **`neo4j/connection.py`**:
> - Read `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` from environment (`.env` file via python-dotenv)
> - `get_driver()` → returns a `neo4j.GraphDatabase.driver` instance
> - `test_connection()` → runs `RETURN 1` and prints server info
>
> **`neo4j/schema.py`**:
> - `apply_schema(driver)` → runs all `CREATE CONSTRAINT` and `CREATE INDEX` statements from the plan
> - Also ensures the Creator label logic: Creator is not a separate node — it's a second label added to User nodes where `is_creator=True`. The schema must add this as part of the upload, not as a separate node type.
>
> **`neo4j/loader.py`**:
>
> Write one `upload_*` function per node type and relationship type. Each function must:
> - Use `UNWIND $batch AS row` pattern for bulk writes
> - Accept `driver`, `data: list[dict]`, `batch_size=500`
> - Show tqdm progress per batch
> - Use `MERGE` (not `CREATE`) for idempotent uploads
>
> Functions to implement:
> ```
> upload_countries(driver, data)
> upload_topics(driver, data)
> upload_sounds(driver, data)
> upload_hashtags(driver, data)
> upload_entities(driver, data)
> upload_users(driver, data)           # also adds :Creator label where is_creator=True
> upload_sessions(driver, data)
> upload_videos(driver, data)
> upload_comments(driver, data)
>
> upload_rel_has_session(driver, data)        # User-[:HAS_SESSION]->UserSession
> upload_rel_last_session(driver, data)       # User-[:LAST_SESSION]->UserSession
> upload_rel_prev_session(driver, data)       # UserSession-[:PREVIOUS_SESSION]->UserSession
> upload_rel_created_by(driver, data)         # Video-[:CREATED_BY]->Creator
> upload_rel_viewed(driver, data)             # UserSession-[:VIEWED {watch_time, completion_rate}]->Video
> upload_rel_liked(driver, data)
> upload_rel_skipped(driver, data)
> upload_rel_reposted(driver, data)
> upload_rel_commented(driver, data)          # UserSession-[:COMMENTED]->Comment
> upload_rel_comment_on_video(driver, data)   # Comment-[:ON_VIDEO]->Video
> upload_rel_video_hashtag(driver, data)
> upload_rel_video_entity(driver, data)
> upload_rel_video_sound(driver, data)
> upload_rel_video_topic(driver, data)
> upload_rel_entity_topic(driver, data)
> upload_rel_user_country(driver, data)
> upload_rel_video_country(driver, data)
> upload_rel_sound_country(driver, data)
> upload_rel_follows(driver, data)            # User-[:FOLLOWS {engagement_score}]->User
> upload_rel_interested_topic(driver, data)   # User-[:INTERESTED_IN_TOPIC {topic_score}]->Topic
> upload_rel_interested_entity(driver, data)
> upload_rel_interested_hashtag(driver, data)
> ```

---

### PROMPT 8 — Main Orchestrator + Full Run

> **Goal**: Build `main.py` and run the full pipeline end-to-end.
>
> `main.py` must:
> 1. Parse CLI args: `--scale small|medium|large` (overrides num_users/num_videos in params), `--skip-llm` (use faker fallback), `--skip-upload` (generate only, no Neo4j), `--config path/to/params.yaml`
> 2. Load all configs via `generators/base.py`
> 3. Run pipeline in this exact order:
>    ```
>    Step 1: Generate taxonomy nodes (topics, countries, sounds, hashtags, entities)
>    Step 2: Generate users + follow graph
>    Step 3: Generate sessions
>    Step 4: Generate videos + assign taxonomy relationships
>    Step 5: Generate interactions (views, likes, skips, etc.) + interest scores
>    Step 6: LLM text fill (descriptions + comments) — or faker fallback
>    Step 7: Neo4j schema setup
>    Step 8: Upload all nodes (in dependency order)
>    Step 9: Upload all relationships (in dependency order)
>    ```
> 4. Print a summary table at the end:
>    ```
>    Nodes uploaded:    Users: N  Sessions: N  Videos: N  Comments: N  ...
>    Rels uploaded:     VIEWED: N  LIKED: N  FOLLOWS: N  INTERESTED_IN_*: N  ...
>    Total time: X.Xs
>    ```
>
> Also create a `.env.example` file:
> ```
> NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
> NEO4J_USER=neo4j
> NEO4J_PASSWORD=your_password
> OLLAMA_MODEL=llama3.2
> ```
>
> Test with `--scale small --skip-llm --skip-upload` first to validate generation. Then test `--skip-llm` only (with real Neo4j). Finally full run.

---

### PROMPT 9 — Recommendation Queries + GDS Analysis

> **Goal**: Create a `queries/` folder with Cypher files for recommendations, community detection, and creator centrality. These will be run against the populated Neo4j Aura instance.
>
> **`queries/recommendations.cypher`** — add explanatory comments before each query:
>
> 1. **Content-Based Recommendation** — Given a user, find videos they haven't seen, ranked by overlap between video topics/entities/hashtags and user interest scores:
> ```cypher
> // Content-based: match video taxonomy to user interest profile
> MATCH (u:User {user_id: $user_id})-[:INTERESTED_IN_TOPIC {topic_score: ts}]->(t:Topic)<-[:IS_ABOUT]-(v:Video)
> WHERE NOT EXISTS { MATCH (u)-[:HAS_SESSION]->(:UserSession)-[:VIEWED]->(v) }
> WITH v, sum(ts) AS relevance_score
> ORDER BY relevance_score DESC LIMIT 20
> RETURN v.video_id, v.description, relevance_score
> ```
>
> 2. **Collaborative Filtering (Similar Users)** — find users with similar interest vectors, return videos they liked that our user hasn't seen
>
> 3. **Trending Videos** — videos with highest completion_rate × view_count in last 7 days of sessions
>
> 4. **Creator Recommendation** — suggest creators to follow based on topic overlap and engagement score of existing follows
>
> 5. **Interest Score Update Query** — given a new VIEWED interaction, update INTERESTED_IN_TOPIC score:
> ```cypher
> // Feedback loop: update topic interest based on watch completion
> MATCH (u:User {user_id: $user_id}), (v:Video {video_id: $video_id})-[:IS_ABOUT]->(t:Topic)
> MERGE (u)-[r:INTERESTED_IN_TOPIC]->(t)
> ON CREATE SET r.topic_score = 0.0
> SET r.topic_score = r.topic_score + CASE
>   WHEN $completion_rate >= 0.8 THEN 0.5 * $completion_rate
>   WHEN $completion_rate < 0.15 THEN -0.3   // skip
>   ELSE 0.1
> END
> ```
>
> **`queries/gds_analysis.cypher`**:
>
> 1. **Project the interest graph** into GDS (users + topic interests as weighted edges)
> 2. **Louvain Community Detection** on users by shared interests — find interest clusters
> 3. **PageRank on Creators** — centrality within the creator follow network
> 4. **Node Similarity** (Jaccard) between users based on shared topics/entities — used for collaborative filtering
> 5. **Betweenness Centrality** on the follow graph — find bridge users between communities
> 6. **Weakly Connected Components** — sanity check that the graph is well-connected
>
> **`queries/exploration.cypher`**:
> - Sanity check queries: count all node types, check avg degree, verify no orphan nodes
> - Distribution checks: histogram of completion_rates, topic_score distributions
> - Sample path: shortest path between two random users through shared interests
>
> Also document: after GDS analysis, write back community IDs to User nodes so they can be used in recommendation filtering.

---

## Data Quality Invariants (Verify Before Upload)

- Every `Video` node has exactly 1 `CREATED_BY` → `Creator`
- Every `Video` has 1-3 `HAS_HASHTAG`, 1 `IS_ABOUT`, 1 `USES_SOUND`
- Every `UserSession` has exactly 1 `User` via `HAS_SESSION`
- No session's `end_date` < `start_date`
- Sessions for a user are correctly chained via `PREVIOUS_SESSION`
- `VIEWED` relationships have `completion_rate` ∈ [0, 1] and `watch_time` > 0
- `topic_score` ∈ [0, 1] after normalization
- No user follows themselves
- `followers` count on User ≈ number of incoming `FOLLOWS` edges (±5% due to sampling)
- Total `LIKED` rels per video ≈ video.likes (from generation, within 10%)

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Creator = second label on User, not separate node | Avoids duplication; Cypher still allows `MATCH (c:Creator)` |
| Interest scores stored as rel properties, not node properties | Allows per-user-per-topic granularity; GDS can project them as edge weights |
| `SKIPPED` is a rel, not an absence | Makes negative feedback explicit and queryable — critical for the loss function |
| `watch_ratio` from KuaiRec can be > 1 | Stored as `completion_rate = min(1, watch_ratio)` on VIEWED edges |
| LLM fills descriptions after graph structure is built | Decouples structure generation (fast) from text generation (slow) |
| Batch size 500 for Neo4j upload | Balances memory and transaction overhead for Aura free tier |
| Faker locale per country | Ensures Japanese users get Japanese-syllable usernames, Brazilian users get Portuguese names, etc. |

---

## Notes on Step 2 (Recommendation Engine — Future Work)

After the graph is populated:
- **Loss function**: `L = Σ -log(P(engagement | watch_ratio))` where engagement = watch, like, repost; skip = strong negative signal. Update INTERESTED_IN scores via the feedback loop query.
- **GDS Similarity**: `gds.nodeSimilarity` on (User)-[:INTERESTED_IN_TOPIC]->(Topic) gives a user-user similarity matrix. Top-K similar users → collaborative filtering.
- **Creator centrality**: `gds.pageRank` on `FOLLOWS` graph identifies influential creators per topic community. Use for ad targeting.
- **Paths**: `shortestPath` through shared topic interests reveals serendipitous recommendations.
- **Community labels**: Write back Louvain `communityId` to User nodes → use as a cheap pre-filter before scoring.
