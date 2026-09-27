"""GoEmotions (simplified) to NRC Plutchik tag bridges for emotion-specific M3."""

from __future__ import annotations

# Must stay aligned with go_emotions simplified label order from the HF dataset.
SIMPLIFIED_GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

# Many-to-one: each GoEmotion label is linked to one or more NRC Plutchik categories
# (semantic + GoEmotions literature grouping, ch. 3 conception).
GOEMOTION_TO_NRC: dict[str, tuple[str, ...]] = {
    "admiration": ("trust", "joy"),
    "amusement": ("joy",),
    "anger": ("anger",),
    "annoyance": ("anger",),
    "approval": ("trust",),
    "caring": ("trust", "joy"),
    "confusion": ("surprise",),
    "curiosity": ("anticipation",),
    "desire": ("anticipation", "joy"),
    "disappointment": ("sadness", "surprise"),
    "disapproval": ("anger", "disgust"),
    "disgust": ("disgust",),
    "embarrassment": ("fear", "sadness"),
    "excitement": ("joy", "anticipation"),
    "fear": ("fear",),
    "gratitude": ("joy", "trust"),
    "grief": ("sadness",),
    "joy": ("joy",),
    "love": ("joy", "trust"),
    "nervousness": ("fear", "anticipation"),
    "optimism": ("anticipation", "joy"),
    "pride": ("joy", "trust"),
    "realization": ("surprise",),
    "relief": ("joy", "trust"),
    "remorse": ("sadness", "fear"),
    "sadness": ("sadness",),
    "surprise": ("surprise",),
    "neutral": (),
}


def nrc_tags_for_label(label: str) -> tuple[str, ...]:
    return GOEMOTION_TO_NRC.get(label, ())
