from poolgeist.simulation.group_stage import rank_group
from poolgeist.simulation.match import MatchResult


def test_group_stage_ranking_points_goal_difference_goals_for():
    table = rank_group(
        [
            MatchResult("A", "B", 2, 0, "A"),
            MatchResult("C", "D", 1, 1, None),
            MatchResult("A", "C", 1, 1, None),
        ]
    )
    assert table.iloc[0]["team"] == "A"
    assert table.iloc[0]["points"] == 4
