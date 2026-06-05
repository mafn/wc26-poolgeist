"""Explainable neutral Team Cards tactical layer."""

from __future__ import annotations

from dataclasses import dataclass, field

ABILITIES = [
    "Low Block Wall",
    "Transition Burst",
    "Set-Piece Monster",
    "Pressing Trap",
    "Possession Lock",
    "Chaos Aura",
    "Tournament Armor",
    "Golden Boot Funnel",
    "Host Buff",
    "Penalty Wizard",
    "Late Surge",
    "Rotation Machine",
]
WEAKNESSES = [
    "Weak Aerial Defense",
    "High Line Vulnerability",
    "Pressing Fragility",
    "Low Chance Conversion",
    "Goalkeeper Question",
    "Discipline Risk",
    "Thin Bench",
    "Slow Center Backs",
    "Set-Piece Fragility",
    "Over-Reliance on Star",
    "Low Block Problem",
    "Tournament Naivety",
]


@dataclass(frozen=True)
class TeamCard:
    """A neutral tactical team card with explainable numeric attributes."""

    team: str
    group: str = ""
    base_rating: float = 0.5
    attack: float = 0.5
    defense: float = 0.5
    speed: float = 0.5
    control: float = 0.5
    set_pieces: float = 0.5
    mentality: float = 0.5
    chaos: float = 0.5
    stamina: float = 0.5
    penalty_skill: float = 0.5
    clutch: float = 0.5
    star_concentration: float = 0.5
    rotation_depth: float = 0.5
    aerial_strength: float = 0.5
    goalkeeper_quality: float = 0.5
    pressing_intensity: float = 0.5
    low_block_quality: float = 0.5
    transition_threat: float = 0.5
    possession_security: float = 0.5
    defensive_line_height: float = 0.5
    discipline: float = 0.5
    tournament_experience: float = 0.5
    abilities: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    status_effects: list[str] = field(default_factory=list)
    notes: str = ""


def explainable_modifiers(card: TeamCard) -> dict[str, float | list[str]]:
    """Produce transparent xG/tempo/variance modifiers from a Team Card."""

    attack_modifier = 0.12 * (card.attack - 0.5) + 0.05 * (card.set_pieces - 0.5)
    defense_modifier = -0.12 * (card.defense - 0.5) - 0.05 * (card.goalkeeper_quality - 0.5)
    tempo_modifier = 0.10 * (card.speed + card.pressing_intensity - 1.0)
    chaos_modifier = 0.12 * (card.chaos - 0.5) + (0.04 if "Chaos Aura" in card.abilities else 0.0)
    penalty_modifier = 0.10 * (card.penalty_skill - 0.5) + (
        0.04 if "Penalty Wizard" in card.abilities else 0.0
    )
    explanations = [
        "Modifiers are linear, bounded, and derived from visible Team Card fields.",
        "No card is used by default unless user data explicitly loads it.",
    ]
    return {
        "attack_modifier": attack_modifier,
        "defense_modifier": defense_modifier,
        "tempo_modifier": tempo_modifier,
        "chaos_modifier": chaos_modifier,
        "penalty_modifier": penalty_modifier,
        "explanations": explanations,
    }
