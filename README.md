# wc26-poolgeist

`poolgeist` is a public, probabilistic, Monte Carlo office-pool simulator for the 2026 World Cup. It is designed for neutral experimentation with match models, ensemble blending, and configurable office-pool scoring rules.

## No affiliation

Poolgeist is an independent open-source project. It is **not** an official FIFA, World Cup, Kicktipp, or tournament-organizer product, and it is not endorsed by or affiliated with those organizations. The phrase "Kicktipp-style" only describes configurable office-pool scoring mechanics.

## Not betting advice

Poolgeist is for education, entertainment, and reproducible simulation experiments. It is not betting, gambling, financial, or investment advice.

## Data policy

This repository intentionally contains only neutral demo data and templates. It does not include private predictions, hard-coded favorites, Slalom-specific data, biased default picks, or proprietary tournament forecasts.

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
from poolgeist.models.ensemble import blend_signals
from poolgeist.models.ising import IsingPressureModel
from poolgeist.models.octopus import RandomOctopusOracle
from poolgeist.models.poisson import PoissonGoalsModel
from poolgeist.models.spin_flip import SpinFlipVolatilityModel

models = [
    PoissonGoalsModel(),
    IsingPressureModel(),
    SpinFlipVolatilityModel(),
    RandomOctopusOracle(seed=2026),
]

signals = [model.predict_match("Neutral A", "Neutral B") for model in models]
ensemble_signal = blend_signals(signals)
print(ensemble_signal.probabilities)
```

## Running without API keys

Search, news, and LLM features are off by default. Copy `.env.example` to `.env` only if you want local configuration. You can run tests, simulations, and the demo notebook without any API keys:

```bash
make test
make notebook
```

## Enabling search/news

Set the feature flags and relevant provider API keys in your environment or local `.env` file:

```bash
POOLGEIST_ENABLE_WEB_SEARCH=true
SERPAPI_API_KEY=...
TAVILY_API_KEY=...
BRAVE_API_KEY=...
```

Adapters read credentials from environment variables only. Do not commit secrets.

## Configuring `OPENAI_MODEL`

LLM-assisted news summaries are optional and disabled by default. To enable an OpenAI-backed workflow, install the optional dependency and set both variables:

```bash
POOLGEIST_ENABLE_LLM_NEWS=true
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-mini
```

The code does not hard-code a model name for production use; choose a model appropriate for your budget, latency, and accuracy requirements.

## Development commands

```bash
make format
make lint
make test
```

## Limitations

- Starter models are intentionally simple placeholders, not calibrated tournament forecasts.
- Demo inputs are synthetic and neutral.
- Web/news adapters are minimal and require external API credentials.
- Monte Carlo results depend on model assumptions and should be treated as scenario analysis, not truth.
