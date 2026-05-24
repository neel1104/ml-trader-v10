# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Commit Walkthrough Release Note:** Every completed feature or strategy implementation must have a corresponding walkthrough release note file saved in `docs/superpowers/walkthroughs/` using the chronological naming format `YYYY-MM-DD-<description>.md`, describing what was changed, what was tested, and the exact quantitative performance impact. Stage and commit this file.
5. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
6. **Clean up** - Clear stashes, prune remote branches
7. **Verify** - All changes committed AND pushed
8. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->


## State Verification (CRITICAL)

Before proposing next steps or acting on local files (e.g., plan documents, backtest results, hyperopt outputs):
1. **Always verify the current git state** by running `git log -n 5 --oneline` and `git status`.
2. **Always verify the current beads tracker state** by running `bd list` or `bd show`.
3. **Never assume project completion state based solely on local artifact timestamps** or unchecked markdown plan checkboxes. The git repository and the beads issue tracker are the sole source of truth. Relying on outdated `backtest_results/` or `hyperopt_results/` files can lead to repeating already-completed work.

## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

### 🌟 Google Colab & Remote Execution Synchronicity (CRITICAL)

To maintain an uncompromised cloud execution pipeline for high-performance backtesting and Walk-Forward Optimizations (WFO):
1. **Pushes Are Mandatory:** Every update to the strategy, features, configurations, or dependencies MUST be committed and pushed immediately so they are pulled inside the Colab runtime.
2. **Notebook Compatibility Verification:** If you modify `requirements.txt`, install new packages, or change C-level dependencies (such as TA-Lib wrapper requirements):
   - **You MUST immediately edit `colab_notebook.ipynb`** to ensure the installation cells setup the environment correctly with the new requirements.
3. **Paths Integrity:** Ensure that all relative paths used in any script (`run_wfo.py`, config loaders, custom metrics) remain compatible with Colab's standard paths (`/content/ml-trader-v10/...`).

