"""Monte Carlo tournament simulation summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from poolgeist.models.base import MatchModel
from poolgeist.simulation.knockout import simulate_knockout_match


def simulate_world_cup_2026(
    teams: Sequence[str],
    model: MatchModel,
    *,
    ratings: dict[str, float] | None = None,
    generator: np.random.Generator | None = None,
) -> str:
    """Simulate a full 48-team 2026 World Cup tournament and return the champion."""

    if len(teams) != 48:
        raise ValueError("Exactly 48 teams are required for the 2026 World Cup simulation.")

    gen = generator or np.random.default_rng()

    # 1. Group assignment
    try:
        from poolgeist.data import load_team_cards

        cards = load_team_cards()
        team_to_group = {card.team: card.group for card in cards if card.group}
    except Exception:
        team_to_group = {}

    group_names = list("ABCDEFGHIJKL")
    groups = {g: [] for g in group_names}

    # Try official grouping
    for team in teams:
        g = team_to_group.get(team)
        if g in groups:
            groups[g].append(team)

    # Check if we have exactly 4 teams in each group. If not, fallback to sequential
    if any(len(teams_in_group) != 4 for teams_in_group in groups.values()):
        groups = {g: [] for g in group_names}
        for i, team in enumerate(teams):
            g = group_names[min(i // 4, 11)]
            groups[g].append(team)

    # 2. Simulate Group Stage
    from poolgeist.schemas import Fixture
    from poolgeist.simulation.group_stage import rank_group
    from poolgeist.simulation.match import simulate_outcome

    group_tables = {}
    for g, g_teams in groups.items():
        # Generate 6 fixtures
        fixtures = [
            Fixture(
                match_id=f"{g}1", home_team=g_teams[0], away_team=g_teams[1], stage="group", group=g
            ),
            Fixture(
                match_id=f"{g}2", home_team=g_teams[2], away_team=g_teams[3], stage="group", group=g
            ),
            Fixture(
                match_id=f"{g}3", home_team=g_teams[0], away_team=g_teams[2], stage="group", group=g
            ),
            Fixture(
                match_id=f"{g}4", home_team=g_teams[1], away_team=g_teams[3], stage="group", group=g
            ),
            Fixture(
                match_id=f"{g}5", home_team=g_teams[0], away_team=g_teams[3], stage="group", group=g
            ),
            Fixture(
                match_id=f"{g}6", home_team=g_teams[1], away_team=g_teams[2], stage="group", group=g
            ),
        ]

        results = []
        for f in fixtures:
            sig = model.predict_match(f.home_team, f.away_team)
            results.append(simulate_outcome(f.home_team, f.away_team, sig, gen))

        group_tables[g] = rank_group(results, ratings=ratings)

    # 3. Extract top 2 + best 8 thirds
    from poolgeist.simulation.third_place import best_third_placed

    # Extract Group winners and runners-up
    winners = {g: group_tables[g].iloc[0]["team"] for g in group_names}
    runners_up = {g: group_tables[g].iloc[1]["team"] for g in group_names}

    thirds_df = best_third_placed(group_tables, n_qualifiers=8)
    thirds_records = thirds_df.to_dict("records")

    # 4. Map them to official 32-team knockout bracket to avoid group-stage rematches
    forbidden_groups = ["E", "I", "A", "L", "G", "D", "B", "K"]
    best_assignment = None
    min_cost = float("inf")

    def backtrack(
        idx: int, current_assignment: list[str], used_indices: set[int], cost: int
    ) -> None:
        nonlocal best_assignment, min_cost
        if idx == 8:
            if cost < min_cost:
                min_cost = cost
                best_assignment = list(current_assignment)
            return

        for i in range(len(thirds_records)):
            if i not in used_indices:
                team_group = thirds_records[i]["group"]
                forbidden = forbidden_groups[idx]
                if team_group != forbidden:
                    current_assignment.append(thirds_records[i]["team"])
                    backtrack(idx + 1, current_assignment, used_indices | {i}, cost + abs(idx - i))
                    current_assignment.pop()

    if len(thirds_records) == 8:
        backtrack(0, [], set(), 0)

    if best_assignment is not None:
        thirds = best_assignment
    else:
        thirds = [t["team"] for t in thirds_records]
        while len(thirds) < 8:
            thirds.append("")

    r32_matchups = {
        73: (runners_up["A"], runners_up["B"]),
        74: (winners["C"], runners_up["F"]),
        75: (winners["E"], thirds[0]),
        76: (winners["F"], runners_up["C"]),
        77: (runners_up["E"], runners_up["I"]),
        78: (winners["I"], thirds[1]),
        79: (winners["A"], thirds[2]),
        80: (winners["L"], thirds[3]),
        81: (winners["G"], thirds[4]),
        82: (winners["D"], thirds[5]),
        83: (winners["H"], runners_up["J"]),
        84: (runners_up["K"], runners_up["L"]),
        85: (winners["B"], thirds[6]),
        86: (runners_up["D"], runners_up["G"]),
        87: (winners["J"], runners_up["H"]),
        88: (winners["K"], thirds[7]),
    }

    # 5. Simulate Round of 32
    r32_winners = {}
    for match_id, (home, away) in r32_matchups.items():
        r32_winners[match_id] = simulate_knockout_match(
            home, away, model, generator=gen
        ).advanced_team

    # 6. Simulate Round of 16 (matches 89 to 96)
    r16_matchups = {
        89: (r32_winners[74], r32_winners[77]),
        90: (r32_winners[73], r32_winners[75]),
        91: (r32_winners[76], r32_winners[78]),
        92: (r32_winners[79], r32_winners[80]),
        93: (r32_winners[83], r32_winners[84]),
        94: (r32_winners[81], r32_winners[82]),
        95: (r32_winners[86], r32_winners[88]),
        96: (r32_winners[85], r32_winners[87]),
    }

    r16_winners = {}
    for match_id, (home, away) in r16_matchups.items():
        r16_winners[match_id] = simulate_knockout_match(
            home, away, model, generator=gen
        ).advanced_team

    # 7. Simulate Quarterfinals (matches 97 to 100)
    qf_matchups = {
        97: (r16_winners[89], r16_winners[90]),
        98: (r16_winners[93], r16_winners[94]),
        99: (r16_winners[91], r16_winners[92]),
        100: (r16_winners[95], r16_winners[96]),
    }

    qf_winners = {}
    for match_id, (home, away) in qf_matchups.items():
        qf_winners[match_id] = simulate_knockout_match(
            home, away, model, generator=gen
        ).advanced_team

    # 8. Simulate Semifinals (matches 101 and 102)
    sf_matchups = {
        101: (qf_winners[97], qf_winners[98]),
        102: (qf_winners[99], qf_winners[100]),
    }

    sf_winners = {}
    for match_id, (home, away) in sf_matchups.items():
        sf_winners[match_id] = simulate_knockout_match(
            home, away, model, generator=gen
        ).advanced_team

    # 9. Simulate Final (match 104)
    champion = simulate_knockout_match(
        sf_winners[101], sf_winners[102], model, generator=gen
    ).advanced_team
    return champion


def simulate_simple_knockout(
    teams: Sequence[str], model: MatchModel, *, seed: int | None = 2026
) -> pd.DataFrame:
    """Simulate a simple neutral knockout bracket and return champion counts.

    Odd-length rounds are handled by giving the final unpaired team a bye into the next round.
    """

    if not teams:
        raise ValueError("At least one team is required to simulate a knockout bracket.")

    if len(teams) == 48:
        ratings = None
        try:
            from poolgeist.data import load_team_cards

            cards = load_team_cards()
            ratings = {card.team: card.base_rating for card in cards}
        except Exception:
            pass
        generator = np.random.default_rng(seed)
        winner = simulate_world_cup_2026(teams, model, ratings=ratings, generator=generator)
        return pd.DataFrame([{"team": winner, "champion_count": 1}])

    generator = np.random.default_rng(seed)
    current = list(teams)
    while len(current) > 1:
        next_round = []
        for i in range(0, len(current), 2):
            if i + 1 >= len(current):
                next_round.append(current[i])
                continue
            next_round.append(
                simulate_knockout_match(
                    current[i], current[i + 1], model, generator=generator
                ).advanced_team
            )
        current = next_round
    return pd.DataFrame([{"team": current[0], "champion_count": 1}])


def monte_carlo_champions(
    teams: Sequence[str], model: MatchModel, *, n_simulations: int = 5_000, seed: int | None = 2026
) -> pd.DataFrame:
    """Run repeated simple brackets/tournaments and return champion probabilities."""

    if len(teams) == 48:
        ratings = None
        try:
            from poolgeist.data import load_team_cards

            cards = load_team_cards()
            ratings = {card.team: card.base_rating for card in cards}
        except Exception:
            pass
        counts: Counter[str] = Counter()
        rng = np.random.default_rng(seed)
        for _ in range(n_simulations):
            winner = simulate_world_cup_2026(teams, model, ratings=ratings, generator=rng)
            counts[str(winner)] += 1
        return pd.DataFrame(
            [
                {"team": team, "champion_probability": count / n_simulations}
                for team, count in counts.items()
            ]
        ).sort_values("champion_probability", ascending=False)

    counts: Counter[str] = Counter()
    rng = np.random.default_rng(seed)
    for _ in range(n_simulations):
        order = list(teams)
        rng.shuffle(order)
        winner = simulate_simple_knockout(order, model, seed=int(rng.integers(0, 2**32 - 1))).iloc[
            0
        ]["team"]
        counts[str(winner)] += 1
    return pd.DataFrame(
        [
            {"team": team, "champion_probability": count / n_simulations}
            for team, count in counts.items()
        ]
    ).sort_values("champion_probability", ascending=False)
