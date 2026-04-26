# ML Trader v10

Machine learning-powered cryptocurrency trading framework built on top of [Freqtrade](https://www.freqtrade.io/) and [FreqAI](https://www.freqtrade.io/en/latest/freqai/). This project aims to establish a systematic alpha exploitation system using advanced quantitative frameworks.

## Project Overview

- **Core Framework:** Freqtrade (2024.2+)
- **ML Library:** CatBoost (FreqAI integration)
- **Primary Goal:** Market-neutral trading and relative value exploitation in digital assets.
- **Key Strategy:** `CatBoostStrategy` (5m timeframe, predictive modeling).

## Directory Structure

- `user_data/`: Freqtrade data directory.
  - `strategies/`: Trading strategy implementations (e.g., `CatBoostStrategy.py`).
  - `config.json`: Main project configuration (pairs, exchange, FreqAI settings).
- `tests/`: Pytest suite for configuration and strategy validation.
- `docs/superpowers/`: Detailed research, plans, and specifications.
- `research.md`: Comprehensive theoretical foundation for the quantitative framework.
- `AGENTS.md`: Mandatory instructions and quality gates for AI agents.

## Getting Started

### Installation

Ensure you have a Python environment (3.10+ recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Bot

**Dry Run:**
```bash
freqtrade trade --config user_data/config.json --strategy CatBoostStrategy --dry-run
```

**Backtesting:**
```bash
freqtrade backtesting --config user_data/config.json --strategy CatBoostStrategy --timerange 20240101-
```

**FreqAI Training:**
Training happens automatically during backtesting or live trading when `freqai` is enabled in the config.

### Testing

Run the test suite using `pytest`:

```bash
pytest
```

## Development Conventions

### Task Tracking (Beads)

This project strictly uses **Beads (`bd`)** for issue and task tracking.
- Run `bd prime` to initialize your session context.
- Use `bd ready` to find unblocked work.
- Do NOT use markdown TODO lists or other trackers.

### AI Agent Workflow

AI agents MUST adhere to the protocol in `AGENTS.md`:
1. **Research & Plan:** Use `research.md` for context and create plans in `docs/superpowers/plans/`.
2. **Execute & Test:** Implement changes surgically and verify with `pytest`.
3. **Beads Integration:** Claim, update, and close issues using `bd`.
4. **Mandatory Push:** All work sessions MUST end with a successful `git push` including beads metadata.

### Coding Style

- Follow standard Freqtrade strategy patterns.
- Ensure all FreqAI features are prefixed with `%-` for features and `&-` for targets as per Freqtrade conventions.
- Prioritize type safety and idiomatic Python.
