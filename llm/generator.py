"""
llm/generator.py
================
High-level text generation pipeline using batched HuggingFace inference.

fill_video_descriptions — mutates videos in place; replaces "PLACEHOLDER"
generate_comments       — returns Comment dicts from pre-built stubs

Both functions degrade gracefully: if the LLM client is unavailable they
fall back to faker-based text so the rest of the pipeline keeps running.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from faker import Faker
from tqdm import tqdm

from generators.base import get_rng
from llm.prompts import ANGLE_TONE_PAIRS, COMMENT_PROMPTS, DESCRIPTION_PROMPTS

# Sentiments in the same order as comment_sentiment_weights lists in params.yaml
_SENTIMENTS = ["positive", "neutral", "negative"]

_BATCH_SIZE = 16  # prompts per GPU batch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_angle_tone(topic_slug: str) -> tuple[str, str]:
    """Randomly choose one of the 7 (angle, tone) pairs for a given topic."""
    rng = get_rng()
    idx = int(rng.integers(0, len(ANGLE_TONE_PAIRS)))
    return ANGLE_TONE_PAIRS[idx]


def _pick_sentiment(topic_slug: str, cfg: dict) -> str:
    """Sample a sentiment weighted by the topic's configured distribution."""
    rng = get_rng()
    weights = (
        cfg.get("comment_sentiment_weights", {})
        .get(topic_slug, [0.70, 0.20, 0.10])
    )
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(_SENTIMENTS), p=probs))
    return _SENTIMENTS[idx]


def _fallback_description(topic_slug: str, fake: Faker) -> str:
    """Generate a plausible two-sentence description without LLM."""
    topic_label = topic_slug.replace("_", " ")
    s1 = fake.sentence(nb_words=12)
    s2 = fake.sentence(nb_words=10)
    return f"{s1.rstrip('.')} about {topic_label}. {s2}"


def _fallback_comment(topic_slug: str, sentiment: str, fake: Faker) -> str:
    """Generate a short comment without LLM."""
    if sentiment == "positive":
        openers = [
            "Actually obsessed with how well this worked",
            "Nobody talks about this enough but",
            "The technique shown here genuinely changed my approach to",
            "Saving this because the detail on",
        ]
        opener = openers[int(get_rng().integers(0, len(openers)))]
        return f"{opener} {topic_slug.replace('_', ' ')} content."
    elif sentiment == "neutral":
        return fake.sentence(nb_words=10).rstrip(".") + "?"
    else:
        return (
            f"I feel like the {topic_slug.replace('_', ' ')} section "
            "could go a bit deeper — would love to see more context next time."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fill_video_descriptions(
    videos: list[dict],
    taxonomy: dict,
    client,
    params: dict,
) -> None:
    """
    Mutate each video whose description is "PLACEHOLDER" with LLM-generated text.
    Uses batched GPU inference when the client supports generate_batch().
    Falls back to faker-based text if the LLM is unavailable.
    """
    if not videos:
        return

    use_llm = client.is_available()
    fake = Faker()
    llm_cfg = params.get("llm", {})
    max_tokens = llm_cfg.get("max_new_tokens", 180)
    temperature = llm_cfg.get("temperature", 0.9)

    tid_to_slug: dict[str, str] = {
        t["topic_id"]: t["slug"] for t in taxonomy.get("topics", [])
    }

    pending = [v for v in videos if v.get("description", "") == "PLACEHOLDER"]
    if not pending:
        return

    if use_llm and hasattr(client, "generate_batch"):
        # Build prompt list upfront for batched inference
        prompts: list[str] = []
        for v in pending:
            topic_slug = tid_to_slug.get(v["topic_id"], "lifestyle_vlog")
            angle, tone = _pick_angle_tone(topic_slug)
            prompts.append(DESCRIPTION_PROMPTS.get((topic_slug, angle, tone), ""))

        # Run in batches with a single tqdm bar
        texts: list[str] = []
        with tqdm(total=len(prompts), desc="Generating descriptions", unit="video") as pbar:
            for i in range(0, len(prompts), _BATCH_SIZE):
                batch_p = prompts[i: i + _BATCH_SIZE]
                batch_t = client.generate_batch(
                    batch_p, max_tokens=max_tokens, temperature=temperature
                )
                texts.extend(batch_t)
                pbar.update(len(batch_p))

        for v, text in zip(pending, texts):
            if not text:
                topic_slug = tid_to_slug.get(v["topic_id"], "lifestyle_vlog")
                text = _fallback_description(topic_slug, fake)
            v["description"] = text
    else:
        for video in tqdm(pending, desc="Generating descriptions", unit="video"):
            topic_slug = tid_to_slug.get(video["topic_id"], "lifestyle_vlog")
            angle, tone = _pick_angle_tone(topic_slug)
            text = ""
            if use_llm:
                prompt = DESCRIPTION_PROMPTS.get((topic_slug, angle, tone), "")
                if prompt:
                    text = client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
            if not text:
                text = _fallback_description(topic_slug, fake)
            video["description"] = text


def generate_comments(
    comment_stubs: list[dict],
    videos: list[dict],
    taxonomy: dict,
    client,
) -> list[dict]:
    """
    Generate Comment node dicts from pre-built stubs.
    Uses batched GPU inference when the client supports generate_batch().

    Each stub must have: video_id, user_id.
    Optional stub keys: sentiment (str), created_at (datetime).

    Returns list of dicts with keys:
        comment_id, video_id, user_id, comment_text, comment_sentiment, created_at
    """
    if not comment_stubs:
        return []

    use_llm = client.is_available()
    fake = Faker()

    vid_map: dict[str, dict] = {v["video_id"]: v for v in videos}
    tid_to_slug: dict[str, str] = {
        t["topic_id"]: t["slug"] for t in taxonomy.get("topics", [])
    }

    # Resolve each stub's topic_slug and sentiment upfront
    resolved: list[tuple[dict, str, str]] = []  # (stub, topic_slug, sentiment)
    for stub in comment_stubs:
        video = vid_map.get(stub.get("video_id", ""))
        if not video:
            continue
        topic_slug = tid_to_slug.get(video["topic_id"], "lifestyle_vlog")
        sentiment = stub.get("sentiment") or _pick_sentiment(topic_slug, taxonomy)
        resolved.append((stub, topic_slug, sentiment))

    if use_llm and hasattr(client, "generate_batch"):
        prompts = [
            COMMENT_PROMPTS.get((slug, sent), "")
            for _, slug, sent in resolved
        ]
        texts: list[str] = []
        with tqdm(total=len(prompts), desc="Generating comments", unit="comment") as pbar:
            for i in range(0, len(prompts), _BATCH_SIZE):
                batch_p = prompts[i: i + _BATCH_SIZE]
                batch_t = client.generate_batch(batch_p, max_tokens=60, temperature=0.95)
                texts.extend(batch_t)
                pbar.update(len(batch_p))
    else:
        texts = []
        for stub, topic_slug, sentiment in tqdm(resolved, desc="Generating comments", unit="comment"):
            text = ""
            if use_llm:
                prompt = COMMENT_PROMPTS.get((topic_slug, sentiment), "")
                if prompt:
                    text = client.generate(prompt, max_tokens=60, temperature=0.95)
            texts.append(text)

    results: list[dict] = []
    for (stub, topic_slug, sentiment), text in zip(resolved, texts):
        if not text:
            text = _fallback_comment(topic_slug, sentiment, fake)
        results.append({
            "comment_id": stub.get("comment_id") or str(uuid.uuid4()),
            "video_id":   stub["video_id"],
            "user_id":    stub["user_id"],
            "comment_text":      text,
            "comment_sentiment": sentiment,
            "created_at":        stub.get("created_at", datetime.now()),
        })

    return results
