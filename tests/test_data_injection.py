from poolgeist.data import load_team_cards, load_team_modifiers
from poolgeist.models.ensemble import ModelCouncil


def test_load_team_cards_and_modifiers():
    # Load cards from the repository templates
    cards = load_team_cards()
    assert len(cards) > 0
    assert any(card.team == "Neutral Team" for card in cards)

    # Compute modifiers
    modifiers = load_team_modifiers()
    assert "Neutral Team" in modifiers
    mods = modifiers["Neutral Team"]
    assert "attack_modifier" in mods
    assert "defense_modifier" in mods
    assert "tempo_modifier" in mods
    assert "chaos_modifier" in mods
    assert "penalty_modifier" in mods


def test_load_team_cards_from_outside_repo_root(monkeypatch, tmp_path):
    import poolgeist.data as data

    data.load_team_cards.cache_clear()
    monkeypatch.chdir(tmp_path)

    cards = data.load_team_cards()

    assert any(card.team == "Neutral Team" for card in cards)
    data.load_team_cards.cache_clear()


def test_load_team_cards_cache_avoids_repeated_csv_parse(monkeypatch):
    import poolgeist.data as data

    data.load_team_cards.cache_clear()
    call_count = 0
    original_read_csv = data.read_csv

    def counting_read_csv(path):
        nonlocal call_count
        call_count += 1
        return original_read_csv(path)

    monkeypatch.setattr(data, "read_csv", counting_read_csv)

    data.load_team_cards()
    data.load_team_cards()

    assert call_count == 1
    data.load_team_cards.cache_clear()


def test_model_council_with_modifiers():
    # Create custom modifiers to test injection
    team_modifiers = {
        "Strong Attack": {
            "attack_modifier": 0.5,
            "defense_modifier": 0.0,
            "tempo_modifier": 0.0,
            "chaos_modifier": 0.0,
            "penalty_modifier": 0.0,
        },
        "Strong Defense": {
            "attack_modifier": 0.0,
            "defense_modifier": -0.5,  # negative is stronger defense
            "tempo_modifier": 0.0,
            "chaos_modifier": 0.0,
            "penalty_modifier": 0.0,
        },
    }

    # Initialize council with custom modifiers
    council = ModelCouncil(team_modifiers=team_modifiers)

    # Predict strong attack vs neutral
    pred1 = council.predict_match("Strong Attack", "Neutral Team")
    # Predict neutral vs strong defense
    pred2 = council.predict_match("Neutral Team", "Strong Defense")

    # The expected goals home in pred1 should be significantly higher than in pred2
    assert pred1.signal.expected_goals_home > pred2.signal.expected_goals_home


def test_model_council_with_frozen_model():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class CustomFrozenModel:
        default_weight: float = 0.1

        def predict_match(self, home_team: str, away_team: str):
            from poolgeist.models.poisson import PoissonGoalsModel

            return PoissonGoalsModel().predict_match(home_team, away_team)

    # ModelCouncil should not crash when initializing with a custom frozen model
    council = ModelCouncil(models=[CustomFrozenModel()], team_modifiers={})
    assert len(council.models) == 1
    assert not hasattr(council.models[0], "team_modifiers")


def test_model_council_with_slotted_model():
    class SlottedModel:
        __slots__ = ()
        default_weight = 1.0

        def predict_match(self, home_team: str, away_team: str):
            from poolgeist.models.poisson import PoissonGoalsModel

            return PoissonGoalsModel().predict_match(home_team, away_team)

    council = ModelCouncil(
        models=[SlottedModel()],
        team_modifiers={"Neutral Team": {"attack_modifier": 0.5}},
    )

    prediction = council.predict_match("Neutral Team", "Neutral Opponent")
    assert prediction.signal.home_team == "Neutral Team"
