# Repository guidance

## Purpose and current state

This repository is a high-throughput, single-agent racing RL system. Vehicle physics,
track queries, environment transitions, rollout collection, GAE, and PPO optimizer updates
run in JAX and are designed to compile together for accelerator execution.

The implemented vertical slice includes:

- A heading-and-speed point vehicle with normalized longitudinal and lateral acceleration actions.
- Closed tracks fitted as periodic cubic splines, with constant-width boundaries.
- A Gymnax-compatible racing environment with randomized training resets and fixed-start
  evaluation.
- An in-repository continuous-action PPO trainer using Flax, Optax, Distrax, `vmap`, and
  nested `lax.scan` loops.
- Checkpoint save/load, deterministic evaluation, synchronized throughput benchmarks,
  live chunked progress logging, periodic fixed/randomized evaluation, lightweight fixed-start
  evaluation videos, CPU CI, and an opt-in three-seed learning acceptance test.

## Repository map

```text
.
├── AGENTS.md                         # Repository-wide instructions (this file)
├── README.md                         # Setup, architecture, and command examples
├── pyproject.toml                    # Package metadata, dependencies, tools, CLI scripts
├── uv.lock                           # Reproducible CPU dependency lock
├── Dockerfile                        # Shared CPU/CUDA 13 image targets
├── .dockerignore                     # Excludes secrets, Git data, and generated state
├── .devcontainer/{cpu,gpu}/          # Selectable local development containers
├── .github/workflows/ci.yml          # CPU lint, type-check, and test workflow
├── .github/workflows/containers.yml  # Container validation and GHCR publication
├── src/ai_race_driver/
│   ├── AGENTS.md                     # JAX-specific implementation constraints
│   ├── __init__.py                   # Supported top-level public exports
│   ├── configuration.py              # Environment validation and dotenv fallback
│   ├── metadata.py                   # Host-side Git metadata for experiment tracking
│   ├── py.typed                      # Typed-package marker
│   ├── vehicle/
│   │   ├── base.py                   # Generic pure `VehicleModel` protocol
│   │   └── point_mass.py             # PointMass state, params, dynamics, angle wrapping
│   ├── track/
│   │   └── spline.py                 # Host spline compiler and device geometry queries
│   ├── envs/
│   │   └── racing.py                 # Gymnax env, observations, rewards, reset/termination
│   ├── training/
│   │   ├── evaluation.py             # Deterministic metric and video trajectory scans
│   │   └── ppo.py                    # Actor-critic, squashed policy, PPO, checkpoints
│   ├── visualization/
│   │   └── video.py                  # Host telemetry, track/model drawing, MP4 encoding
│   └── cli/
│       ├── train.py                  # `ai-race-train`
│       ├── evaluate.py               # `ai-race-eval`
│       └── benchmark.py              # `ai-race-benchmark`
└── tests/
    ├── AGENTS.md                     # Test-specific guidance
    ├── test_vehicle.py               # Dynamics and JIT invariants
    ├── test_track.py                 # Spline, projection, boundary, seam behavior
    ├── test_environment.py           # Gymnax API, reset, vectorization, termination
    └── test_ppo.py                   # PPO smoke and slow learning acceptance
```

Package `__init__.py` files are intentionally thin. Put behavior in the owning module above,
then export it only when it is part of the supported public API.

## Runtime data flow

1. `compile_closed_track` runs on the host with SciPy and returns fixed-shape `TrackData`
   arrays containing spline coefficients and arc-length lookup geometry.
2. `RacingEnvParams` combines `TrackData`, `PointMassParams`, reset ranges, reward weights,
   and the episode limit. It is passed explicitly to Gymnax `reset` and `step` calls.
3. `RacingEnvState` carries the vehicle state, wrapped centerline location, signed accumulated
   progress, tracking errors, previous action, nearest sample index, return, and termination.
4. Each environment step advances the vehicle, projects it locally onto the centerline,
   computes a 14-element ego observation and shaped progress reward, and detects off-track,
   lap-complete, or time-limit termination.
5. `make_train` vectorizes environments with `vmap` and scans both rollouts and PPO updates.
   The training CLI uses the same update scan in resumable compiled chunks, keeping runner state
   on-device while console/W&B logging, evaluation, and checkpoints run at host boundaries. Do
   not insert callbacks or other host work into a compiled chunk.
6. Training serializes Flax parameters plus JSON configuration/metrics. Evaluation rebuilds
   the same network shape, restores parameters, and uses `tanh(mean)` deterministically.
7. Optional evaluation videos use a separate compiled fixed-start trajectory scan. Only selected
   trajectories cross to the host; Pillow/PyAV rendering and W&B media logging never enter a JAX
   transform or PPO chunk.

## Public contracts and invariants

- Actions have shape `(2,)`, are clipped to `[-1, 1]`, and mean normalized `(ax, ay)` demands.
  Combined demands outside the point mass's traction ellipse are projected onto its boundary.
- Point-mass state is `(position[2], heading, speed)` in SI units; speed stays within model
  limits and heading is wrapped to `[-pi, pi)`.
- Tracks are closed, periodic, constant-width, and require fixed sample counts for compiled
  workloads. SciPy is allowed only during host-side track compilation.
- Observations have shape `(14,)`: normalized speed/lateral error, heading-error sine/cosine,
  previous action, and eight curvature previews.
- New vehicle models implement the pure `VehicleModel` contract. Keep the environment and PPO
  independent of concrete model internals; a new state shape may recompile but must remain a
  PyTree.
- PPO uses a tanh-squashed diagonal Gaussian. Preserve the change-of-variables correction in
  both sampling and recomputed log probabilities.
- PRNG keys, parameters, and state are explicit. Public state/config objects remain immutable
  PyTrees with stable array ranks and dtypes.
- `randomize_reset=True` is the training behavior. Evaluation uses a fixed start via params
  replacement; do not fork environment logic for evaluation.

## Development workflow

- Use Python 3.13 and manage environments, dependencies, and commands with `uv`.
- Start with `uv sync`. Do not use ad-hoc global installs.
- Keep the committed `uv.lock` CPU-only. The README documents the local NVIDIA JAX override;
  do not lock platform-specific CUDA wheels into the project.
- Keep the Docker `cpu` target locked to `uv.lock`. The `cuda13` target may overlay only the
  CUDA extra matching the JAX version already installed from that lock.
- Never copy `.env`, `.git`, checkpoints, W&B state, or host virtual environments into an image.
  Container credentials are runtime-only and cloud outputs must use mounted persistent storage.
- Treat `artifacts/` as generated output. Never commit checkpoints or benchmark JSON files.
- Add production dependencies only when they replace meaningful project code or provide a
  required capability. Keep the training hot path small.
- Keep plotting and video encoding in `visualization/`; do not add host rendering methods or
  dependencies to vehicle, track, environment, or compiled training objects.
- Preserve subsystem boundaries. Host orchestration belongs in `cli/`; reusable computation
  belongs in `track/`, `vehicle/`, `envs/`, or `training/`.
- Update the README and this file when commands, module ownership, public contracts, or required
  verification change.

## Commands

```bash
uv sync
uv run ruff check .
uv run mypy
uv run pytest

# Small end-to-end training/checkpoint smoke run
uv run ai-race-train --num-envs 8 --num-steps 8 --total-timesteps 128

# Select compiled chunk and live evaluation/logging cadence in PPO updates
uv run ai-race-train --log-every-updates 4 --eval-episodes 32

# Opt-in fixed-start videos at initial, periodic, and final evaluation boundaries
uv run ai-race-train --video-every-evals 5

# W&B experiment tracking uses required environment variables or the local .env
uv run ai-race-train

# Deterministic checkpoint evaluation
uv run ai-race-eval --checkpoint artifacts/latest --episodes 3

# Synchronized environment benchmark; add --ppo for trainer throughput
uv run ai-race-benchmark --num-envs 2048 --num-steps 1000

# Build the unified development/job images
docker build --platform linux/amd64 --target cpu --tag ai-race-driver:cpu .
docker build --platform linux/amd64 --target cuda13 --tag ai-race-driver:cuda13 .

# Expensive convergence acceptance: three seeds must finish a fixed-start oval lap
AI_RACE_RUN_SLOW=1 uv run pytest -m slow

# The default suite creates and verifies a real W&B portal run
uv run pytest
```

## Pre-PR verification

Before opening or updating a pull request, fetch and merge the latest `origin/master` into the
feature branch:

```bash
git fetch origin master
git merge --no-edit origin/master
git diff --name-only --diff-filter=U
```

Resolve every merge conflict before continuing. The final command must produce no output; inspect
the merged diff and do not open or update the pull request while any unmerged paths remain. Run
the local validation sequence below after the merge so it verifies the exact state proposed for
the pull request.

Before opening or updating a pull request, run every command from the pull-request validation
job in `.github/workflows/ci.yml` locally, in workflow order, and fix all failures:

```bash
uv sync --locked
uv run ruff check .
uv run mypy
uv run pytest
```

Treat the validation workflow as the source of truth if its commands change. The merge-only
`.github/workflows/train-after-merge.yml` performance run is intentionally excluded from
pre-PR verification. Do not open or update the pull request until this local CI sequence passes.

After opening a pull request or pushing new commits to an existing pull request, wait for all
GitHub checks to finish with `gh pr checks --watch`. If any check fails, inspect its Actions
logs, fix the underlying issue, rerun the local CI sequence, push the fix, and watch the new
checks. Do not report the pull request as ready while checks are pending or failing.

## Verification by change type

- Vehicle or track math: run the relevant unit file plus the environment tests; cover plain,
  `jit`, and `vmap` execution and periodic seam behavior.
- Observation, reward, reset, or termination changes: run environment and PPO tests because
  policy input semantics and learning stability are coupled.
- PPO network, distribution, loss, batching, or optimizer changes: run the default suite and
  the slow three-seed acceptance test.
- CLI or checkpoint changes: run a small training command, reload its checkpoint with
  `ai-race-eval`, and keep the generated artifact untracked.
- Performance changes: run `ai-race-benchmark`; report compile and steady-state time separately
  and synchronize device work with `block_until_ready()`.
- Visualization changes: encode and decode a real headless MP4, run a small training command with
  video enabled, and confirm disabled video does not alter PPO compilation or throughput metrics.
- Dependency or packaging changes: run `uv lock`, `uv sync --locked`, all static checks, and the
  default tests. Confirm the lock remains CPU-portable.
- Container changes: build both targets, run a tiny CPU train/evaluate round trip, verify the
  CUDA packages without a GPU, and run training plus PPO benchmarking on a real CUDA 13 node
  before treating GPU execution as verified.

Do not weaken assertions, reduce the slow acceptance requirement, or move device work to Python
to make checks pass. Fix the underlying numerical, compilation, or interface regression.
