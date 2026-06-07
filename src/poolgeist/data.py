"""Data loading helpers for neutral examples and templates."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

from poolgeist.team_cards import TeamCard

TEAM_CARDS_TEMPLATE = Path("data") / "templates" / "team_cards_template.csv"
PACKAGED_TEAM_CARDS_TEMPLATE = ("templates", "team_cards_template.csv")


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file using UTF-8."""

    return pd.read_csv(Path(path))


def load_demo_teams(repo_root: str | Path = ".") -> pd.DataFrame:
    """Load clearly marked neutral demo teams."""

    return read_csv(Path(repo_root) / "examples" / "neutral_demo_teams.csv")


def load_demo_fixtures(repo_root: str | Path = ".") -> pd.DataFrame:
    """Load clearly marked neutral demo fixtures."""

    return read_csv(Path(repo_root) / "examples" / "neutral_demo_fixtures.csv")


@lru_cache(maxsize=4)
def load_team_cards(repo_root: str | Path = ".") -> list[TeamCard]:
    """Load and parse TeamCard objects from the template CSV."""

    df = _read_team_cards_template(repo_root)
    cards = []

    def parse_list(val: Any) -> list[str]:
        if pd.isna(val) or not val:
            return []
        return [x.strip() for x in str(val).split(",") if x.strip()]

    for _, row in df.iterrows():
        card = TeamCard(
            team=str(row["team"]),
            group=str(row.get("group", "")),
            base_rating=float(row.get("base_rating", 0.5)),
            attack=float(row.get("attack", 0.5)),
            defense=float(row.get("defense", 0.5)),
            speed=float(row.get("speed", 0.5)),
            control=float(row.get("control", 0.5)),
            set_pieces=float(row.get("set_pieces", 0.5)),
            mentality=float(row.get("mentality", 0.5)),
            chaos=float(row.get("chaos", 0.5)),
            stamina=float(row.get("stamina", 0.5)),
            penalty_skill=float(row.get("penalty_skill", 0.5)),
            clutch=float(row.get("clutch", 0.5)),
            star_concentration=float(row.get("star_concentration", 0.5)),
            rotation_depth=float(row.get("rotation_depth", 0.5)),
            aerial_strength=float(row.get("aerial_strength", 0.5)),
            goalkeeper_quality=float(row.get("goalkeeper_quality", 0.5)),
            pressing_intensity=float(row.get("pressing_intensity", 0.5)),
            low_block_quality=float(row.get("low_block_quality", 0.5)),
            transition_threat=float(row.get("transition_threat", 0.5)),
            possession_security=float(row.get("possession_security", 0.5)),
            defensive_line_height=float(row.get("defensive_line_height", 0.5)),
            discipline=float(row.get("discipline", 0.5)),
            tournament_experience=float(row.get("tournament_experience", 0.5)),
            abilities=parse_list(row.get("abilities")),
            weaknesses=parse_list(row.get("weaknesses")),
            status_effects=parse_list(row.get("status_effects")),
            notes=str(row.get("notes", "")),
        )
        cards.append(card)
    return cards


def _read_team_cards_template(repo_root: str | Path) -> pd.DataFrame:
    """Read team-card template data from an installed package or source checkout."""

    searched: list[str] = []
    explicit_path = Path(repo_root) / TEAM_CARDS_TEMPLATE
    searched.append(str(explicit_path))
    if explicit_path.exists():
        return read_csv(explicit_path)

    checkout_path = Path(__file__).resolve().parents[2] / TEAM_CARDS_TEMPLATE
    searched.append(str(checkout_path))
    if checkout_path.exists():
        return read_csv(checkout_path)

    package_resource = resources.files("poolgeist").joinpath(*PACKAGED_TEAM_CARDS_TEMPLATE)
    searched.append("poolgeist/" + "/".join(PACKAGED_TEAM_CARDS_TEMPLATE))
    if package_resource.is_file():
        with package_resource.open("rb") as file:
            return pd.read_csv(file)

    searched_locations = ", ".join(searched)
    raise FileNotFoundError(
        f"Could not find team cards template CSV; searched: {searched_locations}"
    )


@lru_cache(maxsize=4)
def load_team_modifiers(repo_root: str | Path = ".") -> dict[str, dict[str, Any]]:
    """Load team cards and compute explainable modifiers for each team."""
    from poolgeist.team_cards import explainable_modifiers

    cards = load_team_cards(repo_root)
    return {card.team: explainable_modifiers(card) for card in cards}
