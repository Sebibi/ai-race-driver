# Repository guidance

## Purpose

This repository is a high-throughput, JAX-native racing RL system. Environment physics and training hot paths must remain compilable and accelerator-friendly.

## Workflow

- Manage dependencies and commands with `uv`; do not use ad-hoc global installs.
- Run `uv run ruff check .`, `uv run mypy`, and `uv run pytest` after code changes.
- Keep the committed dependency lock CPU-only. GPU installation is a documented local override.
- Treat `artifacts/` as generated output and never commit checkpoints or benchmark runs.
- Add production dependencies only when they replace meaningful project code or provide a required capability.

## Architecture

- Preserve the boundaries between track compilation, vehicle dynamics, environment behavior, training, and CLI orchestration.
- Keep public state and parameter objects immutable PyTrees with stable shapes.
- New vehicle models implement the existing pure model contract; do not couple PPO to a concrete vehicle state.
- Keep reward terms and training hyperparameters configurable and record resolved values with checkpoints.

## Verification

- Unit tests and smoke training run on CPU with small static shapes.
- Mark expensive convergence checks with `slow`; do not weaken them to hide regressions.
- Benchmarks must report compilation separately and synchronize device work before recording elapsed time.
