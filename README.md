# Marketing Copy Eval Harness

LLM-as-judge for marketing copy — score variants on a configurable rubric (clarity, hook, CTA, brand fit, emotion), run pairwise comparisons with Elo ratings, get structured verdicts with reasoning.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

> ⚠️ **Disclaimer:** LLM-based evaluation is a signal, not ground truth. Use alongside human review and A/B testing.

![Eval Report](assets/eval-report.svg)

## Features

- **Rubric scoring** — 5 criteria with weights, structured JSON output, averaged over N rounds
- **Pairwise Elo** — all-pairs comparisons + Elo rating (more robust than single-shot scores)
- **Rich reports** — terminal tables with per-criterion scores, rankings, and one-line verdicts
- **Configurable** — custom criteria via rubric.py or YAML

## Install

```bash
pip install -e .
```

## Quickstart

### 1. Configure API key

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### 2. Prepare variants

Create a YAML file with your copy variants (see `examples/variants.yaml`):

```yaml
context: "Landing page hero for a B2B analytics product"
variants:
  - name: "A — Direct benefit"
    text: "Cut reporting time by 80%. One dashboard, every metric."
  - name: "B — Story opener"
    text: "Last quarter, our team spent 200 hours on reports. Then we found a better way."
  - name: "C — Question hook"
    text: "What if your entire team could access real-time metrics without a single spreadsheet?"
```

### 3. Run evaluation

```bash
# Rubric scoring — each variant scored on 5 criteria
python -m copyeval.cli score examples/variants.yaml

# Pairwise Elo — all-pairs head-to-head comparisons
python -m copyeval.cli elo examples/variants.yaml

# Dry run (show prompts without calling API)
python -m copyeval.cli score examples/variants.yaml --dry-run
```

### Example Output

The rubric scorer rates each variant on Clarity, Hook, CTA, Brand Fit, and Emotion (1-5 scale), then computes a weighted total. The Elo mode runs pairwise comparisons and produces a ranking:

```
Elo Ratings (after 6 pairwise comparisons)

  #   Variant                Elo    W-L
  1   B — Story opener      1,068   3-0
  2   A — Direct benefit    1,012   2-1
  3   D — Social proof        988   1-2
  4   C — Question hook        932   0-3
```

## How It Works

1. **Rubric scoring** sends each variant + rubric criteria to Claude, requesting structured JSON with per-criterion scores (1-5) and reasoning. Scores are averaged over N rounds for stability.

2. **Pairwise Elo** presents every pair of variants to the judge, asking "which is better and why?" Results feed into an Elo rating system (K=32, base 1000).

3. Both modes produce **structured verdicts** — the LLM must return valid JSON with scores, reasoning, and a one-line verdict. This prevents vague "both are good" responses.

## Architecture

```
src/copyeval/
├── cli.py       # Typer CLI: score / elo commands
├── client.py    # Anthropic API wrapper
├── judge.py     # Prompt construction + response parsing
├── rubric.py    # Default rubric criteria + weights
├── pairwise.py  # Pair generation, Elo computation
└── models.py    # Pydantic models for verdicts
```

## Roadmap

- [ ] Custom rubric via YAML config
- [ ] Multi-round averaging with confidence intervals
- [ ] HTML report export
- [ ] A/B test result integration

## License

MIT
