# AI Race Driver

A small, fast reinforcement-learning racing stack whose environment, vehicle physics, rollouts, and PPO updates all run in JAX. The first vehicle is a heading-and-speed point mass on a closed periodic cubic-spline track.

## Why this stack

[PureJaxRL](https://github.com/luchris429/purejaxrl) demonstrates the end-to-end compiled training architecture used here, but describes itself as reference code rather than an importable modular library. This project therefore owns its PPO implementation while using [Gymnax](https://github.com/RobertTLange/gymnax) as the functional environment contract. Brax/MJX is deliberately deferred until a future vehicle model needs rigid-body or contact physics.

## Setup

The committed lockfile is CPU-only so development and CI are reproducible:

```bash
uv sync
uv run pytest
```

For a local NVIDIA environment, sync first and then install the appropriate JAX accelerator wheel into `.venv`. Current JAX guidance recommends CUDA 13 for GPUs with compute capability 7.5 or newer:

```bash
uv pip install --python .venv/bin/python --upgrade "jax[cuda13]"
uv run python -c "import jax; print(jax.devices())"
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) if CUDA 12 or a locally installed CUDA runtime is required.

## Commands

```bash
# Small CPU validation run (two PPO updates)
uv run ai-race-train --num-envs 8 --num-steps 8 --total-timesteps 128

# Accelerator-oriented defaults
uv run ai-race-train --output artifacts/oval-seed-0

# Deterministic fixed-start evaluation
uv run ai-race-eval artifacts/oval-seed-0 --episodes 3

# Environment throughput; add --ppo for full training throughput
uv run ai-race-benchmark --num-envs 2048 --num-steps 1000 --output artifacts/benchmark.json
```

Benchmark output separates compilation time from steady-state execution and calls `block_until_ready()` before timing completes.

## Architecture

- `track`: fits closed periodic cubic splines on the host and returns fixed-shape `TrackData` PyTrees. Centerline queries, projection, curvature preview, and boundary checks are device-side JAX.
- `vehicle`: exposes a pure `VehicleModel` contract. `PointMassModel` implements bounded acceleration and yaw-rate controls with semi-implicit integration.
- `envs`: provides the Gymnax-compatible `RacingEnv`. Its 14-value ego observation contains normalized speed/lateral error, heading-error sine/cosine, previous action, and eight curvature previews.
- `training`: contains a fully scanned PPO pipeline using a tanh-squashed Gaussian policy, GAE, clipped objectives, Flax, Optax, and explicit PRNG keys.
- `cli`: training, checkpoint evaluation, and reproducible benchmark entry points.

Tracks use SI units and a constant full width. Training resets randomize the location and pose; evaluation starts at the centerline start. Episodes finish after a forward lap, an off-track departure, or the time limit.

## Quality checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
```

The longer learning acceptance test is opt-in:

```bash
AI_RACE_RUN_SLOW=1 uv run pytest -m slow
```
