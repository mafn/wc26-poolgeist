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
