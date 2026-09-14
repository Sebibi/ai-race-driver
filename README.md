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

## Containers

The repository contains two selectable development-container configurations backed by one
multi-target Dockerfile:

- `AI Race Driver (CPU)` builds the `cpu` target for tests, debugging, and CPU baselines.
- `AI Race Driver (CUDA 13)` builds the `cuda13` target and starts Docker with `--gpus all`.

With the VS Code Dev Containers extension, run **Dev Containers: Reopen in Container** and
choose the required configuration. Both images use Python 3.13, install the locked project and
development tools into `/opt/venv`, and run as an unprivileged `coder` user. The mounted checkout
is synchronized in editable mode when the container is first created.

The CPU container only requires Docker Engine with Buildx. The GPU container additionally
requires an NVIDIA GPU with compute capability 7.5 or newer, Linux driver 580 or newer, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
JAX's CUDA 13 pip wheels provide the CUDA and cuDNN user-space libraries; do not inject a
different CUDA installation through `LD_LIBRARY_PATH`.

Build the same images without a devcontainer as follows:

```bash
docker build --platform linux/amd64 --target cpu --tag ai-race-driver:cpu .
docker build --platform linux/amd64 --target cuda13 --tag ai-race-driver:cuda13 .

docker run --rm ai-race-driver:cpu \
  python -c "import jax; print(jax.devices())"
docker run --rm --gpus all ai-race-driver:cuda13 \
  python -c "import jax; assert jax.default_backend() == 'gpu'; print(jax.devices())"
```

Secrets are supplied only at runtime. For example, the ignored local `.env` file can be passed
to a CPU smoke run while checkpoints are written to a host directory:

```bash
mkdir -p artifacts/container-cpu
docker run --rm \
  --env-file .env \
  --mount type=bind,source="$(pwd)/artifacts/container-cpu",target=/outputs \
  ai-race-driver:cpu \
  ai-race-train \
    --num-envs 8 \
    --num-steps 8 \
    --total-timesteps 128 \
    --eval-episodes 1 \
    --video-every-evals 0 \
    --output /outputs/run
```

The benchmark CLI does not require W&B credentials:

```bash
docker run --rm ai-race-driver:cpu \
  ai-race-benchmark --num-envs 2048 --num-steps 1000 --ppo
docker run --rm --gpus all ai-race-driver:cuda13 \
  ai-race-benchmark --num-envs 2048 --num-steps 1000 --ppo
```

GitHub Actions validates both targets on relevant pull requests. Merges to `master` publish
public images to `ghcr.io/sebibi/ai-race-driver` with moving `cpu` and `cuda13` tags plus immutable
`cpu-git-<full-commit>` and `cuda13-git-<full-commit>` tags. The workflow can also be dispatched
for an explicit branch, tag, or commit; manual publications receive only immutable tags. After
the first publication, set the package visibility to public in GitHub's package settings.

Use the digest reported in the workflow summary for cloud jobs:

```bash
docker pull ghcr.io/sebibi/ai-race-driver@sha256:<digest>
```

A provider-neutral GPU job needs one NVIDIA GPU, the immutable image reference, an
`ai-race-train ... --output /outputs/<run>` command, runtime-injected `WANDB_*` variables, and
persistent storage mounted at `/outputs`. The current trainer uses one JAX device; requesting
multiple GPUs does not yet distribute training across them.

## Commands

```bash
# Small CPU validation run (two PPO updates)
uv run ai-race-train --num-envs 8 --num-steps 8 --total-timesteps 128

# Accelerator-oriented defaults
uv run ai-race-train --output artifacts/oval-seed-0

# Live metrics every PPO update, with evaluation at the same cadence
uv run ai-race-train --log-every-updates 1 --output artifacts/oval-seed-0

# Higher-throughput default: four compiled PPO updates between host synchronizations
uv run ai-race-train --log-every-updates 4 --eval-episodes 32

# Fixed-start videos at step zero, every fifth evaluation, and the final evaluation
uv run ai-race-train --video-every-evals 5 --output artifacts/oval-seed-0

# Deterministic fixed-start evaluation
uv run ai-race-eval --checkpoint artifacts/oval-seed-0 --episodes 3

# Environment throughput; add --ppo for full training throughput
uv run ai-race-benchmark --num-envs 2048 --num-steps 1000 --output artifacts/benchmark.json
```

Benchmark output separates compilation time from steady-state execution and calls `block_until_ready()` before timing completes.
All commands write color-coded logs to stderr when attached to a terminal. Pass `--log-level`
with `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` to control their verbosity.
W&B configuration is environment-only. Training requires non-empty `WANDB_API_KEY`,
`WANDB_PROJECT`, and `WANDB_ENTITY`; `WANDB_NAME` and `WANDB_MODE` remain optional W&B settings.
The reusable `setup_environment` helper first uses variables exported by the process (for
example, an HPC scheduler), then loads missing values from `.env` with `python-dotenv`. Exported
variables take precedence. Training exits with a clear error if required values are still
missing. Generated W&B state is kept under the ignored `wandb/` directory.
Every W&B run records the current Git branch, commit message, and commit hash in its config.

After a pull request is merged into `master`, the `Training performance` GitHub Actions
workflow checks out the exact merge commit, installs the locked dependencies, and runs
`uv run ai-race-train` without command-line overrides. Its Actions and W&B run names use
`master@<merge-commit-sha>`, making default-training performance directly traceable over time.

Training executes static groups of PPO updates in compiled `lax.scan` chunks. The
`--log-every-updates` option controls both the chunk size and live console/W&B/evaluation
cadence; larger values reduce synchronization overhead. W&B still receives one training point
per PPO update after each chunk. At step zero and every chunk boundary, the deterministic policy
is evaluated once from the canonical fixed start and over a reproducible vectorized batch of
randomized starts controlled by `--eval-episodes` and `--eval-seed`. The final policy is saved at
the requested output path and the best fixed-start policy is saved below its `best/` directory.

Evaluation videos are disabled by default, so normal training has no trajectory capture, device
transfer, rendering, or media-upload overhead. Setting `--video-every-evals N` records the
deterministic fixed-start policy at step zero, every Nth evaluation, and the final evaluation.
The video trajectory is produced by a separate compiled scan and transferred to the host only at
those boundaries. Pillow draws the track, driven path, heading marker, speed, longitudinal and
lateral acceleration, traction-ellipse utilization, yaw rate, and lateral error with labeled
y-axis scales; PyAV streams at
most 300 frames directly into an H.264 MP4. A single background worker overlaps rendering with
subsequent training. If it is
still busy when another video is due, training waits rather than dropping the requested video.
Videos are written below `<output>/videos/` and logged to W&B as `eval/fixed/video`. Video capture,
render, wait, and logging durations are reported separately from PPO throughput.

Training episode metrics describe the stochastic policy and randomized resets used to collect
PPO rollouts. `eval/fixed/*` describes the canonical deterministic deployment trajectory, while
`eval/randomized/*` measures deterministic-policy robustness across starting states. Optimizer
loss, approximate KL, clipping fraction, entropy, explained variance, learning rate, throughput,
and evaluation time are logged as diagnostics rather than treated as policy performance.

## Architecture

- `track`: fits closed periodic cubic splines on the host and returns fixed-shape `TrackData` PyTrees. Centerline queries, projection, curvature preview, and boundary checks are device-side JAX.
- `vehicle`: exposes a pure `VehicleModel` contract. `PointMassModel` implements bounded longitudinal and lateral acceleration controls, coupled by a traction ellipse, with semi-implicit integration.
- `envs`: provides the Gymnax-compatible `RacingEnv`. Its 14-value ego observation contains normalized speed/lateral error, heading-error sine/cosine, previous action, and eight curvature previews.
- `training`: contains resumable compiled PPO chunks and deterministic evaluation scans using a tanh-squashed Gaussian policy, GAE, clipped objectives, Flax, Optax, and explicit PRNG keys.
- `visualization`: converts selected deterministic trajectories to host telemetry and renders headless evaluation MP4s without entering the environment or PPO JIT paths.
- `configuration`: validates required process variables and falls back to `.env` for local runs.
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

The default test suite includes an online W&B acceptance test that creates a real portal run,
verifies its uploaded metric and Git metadata through the W&B API, and prints the run URL. Its
run name combines the current branch and commit message in normalized lowercase `snake_case`
form. It requires the same W&B environment variables as training:

```bash
uv run pytest -s
```
