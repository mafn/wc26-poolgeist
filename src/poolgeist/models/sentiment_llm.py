"""Optional LLM-based news-signal classifier."""

from __future__ import annotations

from dataclasses import dataclass

from poolgeist.adapters.openai_client import OpenAIResponsesAdapter

SUPPORTED_SIGNAL_TYPES = {
    "injury",
    "suspension",
    "red_card_ban",
    "lineup_rotation",
    "tactical_change",
    "morale_positive",
    "morale_negative",
    "travel_issue",
    "weather_issue",
    "market_odds_move",
    "press_conference_hint",
    "training_absence",
    "squad_announcement",
    "coach_comment",
    "goalkeeper_issue",
    "star_player_minutes_risk",
}


@dataclass(frozen=True)
class NewsSignal:
    """Structured news signal. It is an input modifier, never a direct prediction."""

    team: str
    player: str | None
    signal_type: str
    severity: float
    confidence: float
    attack_modifier: float = 0.0
    defense_modifier: float = 0.0
    tempo_modifier: float = 0.0
    fatigue_modifier: float = 0.0
    penalty_modifier: float = 0.0
    source_url: str = ""
    source_title: str = ""
    evidence_summary: str = ""
    expires_after_match: bool = False
    notes: str = ""


def heuristic_extract_news_signal(text: str, *, team: str = "") -> NewsSignal | None:
    """Extract conservative keyword-based news signals without API keys."""

    lowered = text.lower()
    if "injur" in lowered:
        return NewsSignal(
            team, None, "injury", 0.5, 0.5, attack_modifier=-0.03, evidence_summary=text[:200]
        )
    if "suspend" in lowered or "red card" in lowered:
        return NewsSignal(
            team, None, "suspension", 0.5, 0.5, defense_modifier=-0.03, evidence_summary=text[:200]
        )
    if "rotation" in lowered:
        return NewsSignal(
            team,
            None,
            "lineup_rotation",
            0.3,
            0.4,
            fatigue_modifier=0.03,
            evidence_summary=text[:200],
        )
    return None


def llm_extract_news_signal(
    text: str, adapter: OpenAIResponsesAdapter | None = None
) -> dict | None:
    """Use optional OpenAI adapter for structured extraction; disabled unless env configured."""

    client = adapter or OpenAIResponsesAdapter()
    return client.extract_structured_signal(text)
