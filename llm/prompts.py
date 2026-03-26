"""
llm/prompts.py
==============
Prompt matrices for video descriptions and comments.

DESCRIPTION_PROMPTS : dict keyed by (topic_slug, angle, tone)   — 84 entries
COMMENT_PROMPTS     : dict keyed by (topic_slug, sentiment)      — 36 entries
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Description prompt matrix  (12 topics × 7 angle/tone pairs = 84 entries)
# ---------------------------------------------------------------------------

# (angle, tone) → instruction template. Placeholder: {context}
_ANGLE_TONE_TEMPLATE: dict[tuple[str, str], str] = {
    ("tutorial", "casual_genz"): (
        "Write a 2-3 sentence TikTok video caption (no hashtags) for a video showing "
        "how to {context}. "
        "Use casual Gen-Z tone — phrases like 'no cap', 'lowkey', 'hits different', 'real talk'. "
        "Mention one specific unexpected detail or technique. "
        "Each caption must be unique; vary vocabulary, structure, and the detail you highlight. "
        "Output only the caption text."
    ),
    ("reaction", "humorous"): (
        "Write a 2-3 sentence TikTok video caption (no hashtags) for a video where the creator "
        "reacts dramatically to {context}. "
        "Tone: funny and slightly over-the-top — exaggerate the shock, include a self-deprecating "
        "joke or absurd comparison. End with a punchy one-liner. "
        "Each caption must feel fresh; avoid repeating 'I can't believe' or 'mind blown'. "
        "Output only the caption text."
    ),
    ("challenge", "casual_genz"): (
        "Write a 2-3 sentence TikTok video caption (no hashtags) for a challenge built around "
        "{context}. "
        "High-energy Gen-Z voice: invite viewers to join, tease the stakes, make it sound unmissable. "
        "Vary your opener — never start with 'POV:' or 'So I tried'. "
        "Output only the caption text."
    ),
    ("storytime", "casual_genz"): (
        "Write a 2-3 sentence TikTok video caption (no hashtags) for a first-person storytime "
        "about {context}. "
        "Start mid-story to hook immediately — breathless, casual pace. "
        "End with either a cliffhanger or an ironic punchline. "
        "Each caption must open differently; vary the narrative hook each time. "
        "Output only the caption text."
    ),
    ("review", "professional"): (
        "Write 2-3 sentences (no hashtags) for a TikTok honest review of {context}. "
        "Tone: informed and balanced — one clear strength, one honest weakness, one verdict. "
        "Sound like a knowledgeable critic, not a hype person. "
        "Vary the structure and vocabulary so each review reads distinctly. "
        "Output only the review text."
    ),
    ("tutorial", "motivational"): (
        "Write 2-3 sentences (no hashtags) for a motivational TikTok tutorial on {context}. "
        "Use direct second-person address and powerful verbs. Name one concrete skill or result "
        "the viewer gains. Vary the opening word every time — never start with 'Learn' or 'Master'. "
        "Output only the caption text."
    ),
    ("reaction", "educational"): (
        "Write 2-3 sentences (no hashtags) for an educational TikTok reaction to {context}. "
        "Tone: informative and mildly enthusiastic. Include one counter-intuitive insight or "
        "surprising fact. Keep it accessible; avoid jargon. "
        "Each version must surface a different fact so no two captions feel alike. "
        "Output only the caption text."
    ),
}

# topic_slug → rich context phrase that fills {context} in every template
_TOPIC_CONTEXT: dict[str, str] = {
    "cooking_food": (
        "a viral dish with an unexpected ingredient twist that changes the entire flavour profile"
    ),
    "gaming_esports": (
        "an insane clutch play or a counterintuitive skill-building technique in a popular game"
    ),
    "fashion_beauty": (
        "a bold outfit combination or beauty hack that transformed an everyday look"
    ),
    "fitness_wellness": (
        "a deceptively simple workout or wellness habit that produces measurable results fast"
    ),
    "travel_adventure": (
        "a hidden gem destination or an off-the-beaten-path experience most travellers miss"
    ),
    "music_dance": (
        "a trending sound challenge or an original choreography that's spreading across FYP"
    ),
    "comedy_entertainment": (
        "a painfully relatable everyday situation that escalates into comedy gold"
    ),
    "technology_science": (
        "a little-known tech feature or a mind-bending scientific fact most people have never heard"
    ),
    "sports_athletics": (
        "a breathtaking athletic skill, an underdog training drill, or a match-winning highlight"
    ),
    "education_tutorials": (
        "a notoriously complex topic distilled into one clear, immediately actionable explanation"
    ),
    "art_creativity": (
        "a jaw-dropping creative process or a DIY transformation revealed step-by-step"
    ),
    "lifestyle_vlog": (
        "a brutally honest day-in-the-life moment or an aspirational routine worth stealing"
    ),
}

# Build: {(topic_slug, angle, tone): prompt_string}
DESCRIPTION_PROMPTS: dict[tuple[str, str, str], str] = {
    (topic, angle, tone): template.format(context=_TOPIC_CONTEXT[topic])
    for topic in _TOPIC_CONTEXT
    for (angle, tone), template in _ANGLE_TONE_TEMPLATE.items()
}

# Ordered list of (angle, tone) pairs — used by generator to pick randomly
ANGLE_TONE_PAIRS: list[tuple[str, str]] = list(_ANGLE_TONE_TEMPLATE.keys())

# ---------------------------------------------------------------------------
# Comment prompt matrix  (12 topics × 3 sentiments = 36 entries)
# ---------------------------------------------------------------------------

_COMMENT_SUBJECT: dict[str, str] = {
    "cooking_food":         "this cooking video",
    "gaming_esports":       "this gaming clip",
    "fashion_beauty":       "this fashion or beauty video",
    "fitness_wellness":     "this fitness video",
    "travel_adventure":     "this travel video",
    "music_dance":          "this music or dance video",
    "comedy_entertainment": "this comedy video",
    "technology_science":   "this tech or science video",
    "sports_athletics":     "this sports video",
    "education_tutorials":  "this tutorial",
    "art_creativity":       "this creative process video",
    "lifestyle_vlog":       "this vlog",
}

# sentiment → instruction template. Placeholder: {subject}
_COMMENT_SENTIMENT_TEMPLATE: dict[str, str] = {
    "positive": (
        "Write a single authentic TikTok viewer comment (no hashtags, no emojis) "
        "from someone who genuinely enjoyed {subject}. "
        "The comment can be anywhere from one short sentence to three sentences — vary the length. "
        "Be specific: mention a technique, a moment, a feeling, or a small detail that stood out. "
        "Never open with 'This is amazing', 'Loved this', or 'Great video'. "
        "Sound like a real person — conversational, specific, varied. "
        "Output only the comment text."
    ),
    "neutral": (
        "Write a single authentic TikTok viewer comment (no hashtags, no emojis) "
        "from someone who watched {subject} and has a genuine question or mild observation. "
        "The comment can be one or two sentences. "
        "Sound curious rather than enthusiastic — ask about a specific detail or note something "
        "you noticed. Vary the phrasing; never open with 'I was wondering' or 'Can you'. "
        "Output only the comment text."
    ),
    "negative": (
        "Write a single authentic TikTok viewer comment (no hashtags, no emojis) "
        "from someone who watched {subject} and has a critical but respectful opinion. "
        "The comment can be one or two sentences. "
        "Express a specific, constructive complaint or honest disagreement — no insults, "
        "just polite pushback that might help the creator improve. "
        "Vary the angle of criticism so each negative comment feels different. "
        "Output only the comment text."
    ),
}

# Build: {(topic_slug, sentiment): prompt_string}
COMMENT_PROMPTS: dict[tuple[str, str], str] = {
    (topic, sentiment): template.format(subject=_COMMENT_SUBJECT[topic])
    for topic in _COMMENT_SUBJECT
    for sentiment, template in _COMMENT_SENTIMENT_TEMPLATE.items()
}
