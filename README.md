# wc26-poolgeist

**Poolgeist** — “An absurdly over-engineered Monte Carlo office-pool simulator.”

`poolgeist` is a public, reusable Python/Jupyter tournament predictor for World Cup 2026 office pools. It combines score-probability models, Monte Carlo simulation, Team Cards, optional market/news signals, an Ising-style pressure layer, a seeded Random Octopus Oracle sanity check, and exact-score strategy tooling.

> This project is for office-pool modelling and probabilistic experimentation. It is not betting advice.

## What Poolgeist is

- A neutral Python package for office-pool and tipping-pool scenario modelling.
- A model-council ensemble where each enabled model contributes a weighted signal.
- A Monte Carlo and exact-score expected-value toolkit with uncertainty, disagreement, chaos, and strategic-value outputs.
- A public-repo scaffold that uses synthetic demo data and templates rather than private picks.

## What Poolgeist is not

- It is **not** a betting, gambling, financial, or investment-advice project.
- It is **not** an official FIFA, World Cup, Kicktipp, or tournament-organizer product, and it is not endorsed by or affiliated with those organizations.
- It does **not** ship a default prediction card, recommended champion, recommended semifinalists, office-specific assumptions, or hard-coded favourite teams.
- It does **not** scrape betting sites by default. Market priors are loaded only from user-provided data.

## Data and privacy policy

This repository intentionally contains only neutral demo data and templates. Do not commit API keys, `.env`, private picks, private company data, raw search dumps containing personal data, or generated outputs from private runs. If optional caching is added, keep caches in ignored directories.

## Setup

Python 3.11 or newer is required. The project prefers `uv` when available:

```bash
uv sync --extra dev --extra openai --extra search
```

Without `uv`:

```bash
python -m pip install -e '.[dev,openai,search]'
```

You can also use:

```bash
make install
```

## Quickstart

```python
from poolgeist.models.ensemble import ModelCouncil

council = ModelCouncil()
prediction = council.predict_match("Neutral A", "Neutral B")

print(prediction.recommendation_classes)
print(prediction.chaos_index)
print(prediction.model_disagreement_index)
print(prediction.score_ev_table.head())
```

The labels above are generated from the model outputs for the supplied teams. They are not built-in default picks.

## Notebook

Run the neutral demo notebook:

```bash
uv run --extra dev jupyter notebook notebooks/wc26_poolgeist_demo.ipynb
```

For a non-interactive smoke test:

```bash
uv run --extra dev jupyter nbconvert --to notebook --execute notebooks/wc26_poolgeist_demo.ipynb --output /tmp/wc26_poolgeist_demo_executed.ipynb
```

The notebook uses neutral synthetic teams and writes example export files only when you execute the export cell.

## Adding fixtures

Use `data/templates/fixtures_template.csv` as the starting point. Required fields include match ID, stage, group, home team, away team, and a neutral venue flag. Real fixtures should be supplied by the user and reviewed for source/license suitability before publication.

## Adding Team Cards

Use `data/templates/team_cards_template.csv`. Team Cards expose visible attributes such as attack, defense, speed, control, set pieces, mentality, stamina, penalty skill, pressing, low block, transition threat, discipline, abilities, weaknesses, and status effects. The Team Card layer produces explainable modifiers; it is not opaque magic and is not loaded as a biased default card.

## Adding market odds

Use `data/templates/market_odds_template.csv` for user-provided market priors such as match 1X2 odds, group winner odds, outright winner odds, or top-scorer odds. Poolgeist can compute implied and overround-adjusted probabilities, but live scraping is not enabled by default.

## Search/news analysis

Search/news features degrade gracefully when API keys are missing. Manual CSV mode works without paid APIs through `ManualCsvSearchProvider`. Optional adapters exist for SerpAPI, Tavily, Brave Search, and Bing Search, but they return no live data unless configured.

Set feature flags explicitly if you want to use network/API features:

```bash
POOLGEIST_ENABLE_WEB_SEARCH=true
SERPAPI_API_KEY=...
TAVILY_API_KEY=...
BRAVE_API_KEY=...
BING_SEARCH_API_KEY=...
```

To disable all network/API features, leave these variables unset or set:

```bash
POOLGEIST_ENABLE_WEB_SEARCH=false
POOLGEIST_ENABLE_LLM_NEWS=false
```

## Configuring OpenAI model

LLM-assisted news extraction is optional. The LLM only extracts structured signals from source text; it must not directly output final predictions. To enable it, install the optional dependency and set both environment variables:

```bash
POOLGEIST_ENABLE_LLM_NEWS=true
OPENAI_API_KEY=...
OPENAI_MODEL=...
```

Poolgeist does not hard-code an OpenAI model name. If `OPENAI_API_KEY` or `OPENAI_MODEL` is missing, LLM news extraction is disabled with a clear warning state in the adapter.

## Interpreting safe, chaos, and strategic picks

- **Safest**: the candidate score with the highest exact-score probability.
- **Highest raw EV**: the score with the best expected points under the scoring config.
- **Highest strategic value**: raw EV adjusted for estimated popularity, uniqueness, chaos, and downside risk.
- **Highest chaos value**: a strategic candidate that benefits from uncertainty and volatility.
- **Model disagreement pick**: a candidate highlighted because council members disagree materially.

Do not treat any class as truth. Poolgeist reports uncertainty, disagreement, robustness, warnings, and limitations as first-class outputs.

## Development commands

```bash
make format
make lint
make test
```

## Limitations

- Starter model implementations are deliberately lightweight and should be calibrated before serious use.
- Demo inputs are synthetic and neutral.
- Search/news adapters are minimal and optional.
- Single-match form updates are noisy; Bayesian updates use conservative prior weights.
- Monte Carlo results depend on model assumptions and should be treated as scenario analysis, not forecasts with fake precision.

## License

MIT. See `LICENSE`.
