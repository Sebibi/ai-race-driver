"""End-to-end JIT-compiled PPO for continuous racing control."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from ai_race_driver.envs.racing import RacingEnv, RacingEnvParams


@dataclass(frozen=True)
class PPOConfig:
    """Static PPO shapes and hyperparameters."""

    total_timesteps: int = 10_000_000
    num_envs: int = 2_048
    num_steps: int = 128
    num_minibatches: int = 4
    update_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    anneal_learning_rate: bool = True
    hidden_size: int = 128

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        if self.batch_size % self.num_minibatches:
            raise ValueError("num_envs * num_steps must be divisible by num_minibatches")
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        transitions_per_update = self.batch_size
        updates, remainder = divmod(self.total_timesteps, transitions_per_update)
        if remainder:
            raise ValueError("total_timesteps must be divisible by num_envs * num_steps")
        if updates < 1:
            raise ValueError("total_timesteps must contain at least one PPO update")
        return updates

    @classmethod
    def cpu_smoke(cls) -> "PPOConfig":
        """Small, fast configuration for tests and local validation."""

        return cls(
            total_timesteps=128,
            num_envs=8,
            num_steps=8,
            num_minibatches=2,
            update_epochs=2,
            hidden_size=32,
        )

    def __str__(self) -> str:
        fields = asdict(self)
        return "PPOConfig(\n" + "\n".join(
            f"  {key}={value}" for key, value in fields.items()
        ) + "\n)"




class ActorCritic(nn.Module):
    """Separate actor and critic MLP towers."""

    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor = observation
        for _ in range(2):
            actor = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(actor)
            actor = nn.tanh(actor)
        mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        log_std = jnp.broadcast_to(log_std, mean.shape)

        critic = observation
        for _ in range(2):
            critic = nn.Dense(
                self.hidden_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(critic)
            critic = nn.tanh(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return mean, log_std, jnp.squeeze(value, axis=-1)


class Transition(NamedTuple):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_probability: jax.Array
    observation: jax.Array
    lap_complete: jax.Array
    returned_episode: jax.Array
    returned_episode_return: jax.Array


class UpdateMetrics(NamedTuple):
    loss: jax.Array
    mean_reward: jax.Array
    completed_laps: jax.Array
    completed_episodes: jax.Array
    mean_completed_return: jax.Array


class TrainOutput(NamedTuple):
    train_state: TrainState
    environment_state: Any
    final_observation: jax.Array
    metrics: UpdateMetrics


def _distribution(mean: jax.Array, log_std: jax.Array) -> distrax.Distribution:
    return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))


def sample_squashed_action(
    mean: jax.Array, log_std: jax.Array, key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Sample a tanh-squashed Gaussian action and its corrected log probability."""

    distribution = _distribution(mean, log_std)
    raw_action = distribution.sample(seed=key)
    action = jnp.tanh(raw_action)
    correction = jnp.sum(jnp.log(1.0 - jnp.square(action) + 1e-6), axis=-1)
    return action, distribution.log_prob(raw_action) - correction


def squashed_log_probability(mean: jax.Array, log_std: jax.Array, action: jax.Array) -> jax.Array:
    """Evaluate log probability under a tanh-squashed Gaussian."""

    clipped_action = jnp.clip(action, -1.0 + 1e-6, 1.0 - 1e-6)
    raw_action = jnp.arctanh(clipped_action)
    correction = jnp.sum(jnp.log(1.0 - jnp.square(clipped_action) + 1e-6), axis=-1)
    return _distribution(mean, log_std).log_prob(raw_action) - correction


def deterministic_action(model: ActorCritic, params: Any, observation: jax.Array) -> jax.Array:
    """Return the bounded mean action used for evaluation."""

    network_output: Any = model.apply(params, observation)
    return jnp.tanh(network_output[0])


def make_train(
    env: RacingEnv,
    env_params: RacingEnvParams,
    config: PPOConfig,
):
    """Build a pure training function suitable for one outer ``jax.jit``."""

    num_updates = config.num_updates
    minibatch_size = config.minibatch_size
    model = ActorCritic(action_dim=2, hidden_size=config.hidden_size)
    vector_reset = jax.vmap(env.reset, in_axes=(0, None))
    vector_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def learning_rate_schedule(count: jax.Array) -> jax.Array:
        updates_completed = count // (config.num_minibatches * config.update_epochs)
        fraction_remaining = 1.0 - updates_completed / jnp.maximum(num_updates, 1)
        return config.learning_rate * fraction_remaining

    learning_rate: float | optax.Schedule
    learning_rate = learning_rate_schedule if config.anneal_learning_rate else config.learning_rate
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    def train(key: jax.Array) -> TrainOutput:
        key, initialization_key, reset_key = jax.random.split(key, 3)
        initial_observation = jnp.zeros(env.observation_space(env_params).shape, dtype=jnp.float32)
        parameters = model.init(initialization_key, initial_observation)
        train_state = TrainState.create(apply_fn=model.apply, params=parameters, tx=optimizer)

        reset_keys = jax.random.split(reset_key, config.num_envs)
        observations, environment_state = vector_reset(reset_keys, env_params)
        runner_state = (train_state, environment_state, observations, key)

        def update_step(runner_state, _):
            def environment_step(runner_state, _):
                train_state, environment_state, observation, key = runner_state
                key, action_key, step_key = jax.random.split(key, 3)
                mean, log_std, value = model.apply(train_state.params, observation)
                action, log_probability = sample_squashed_action(mean, log_std, action_key)
                step_keys = jax.random.split(step_key, config.num_envs)
                next_observation, environment_state, reward, done, info = vector_step(
                    step_keys, environment_state, action, env_params
                )
                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_probability=log_probability,
                    observation=observation,
                    lap_complete=info["lap_complete"],
                    returned_episode=info["returned_episode"],
                    returned_episode_return=info["returned_episode_return"],
                )
                return (
                    train_state,
                    environment_state,
                    next_observation,
                    key,
                ), transition

            runner_state, trajectory = jax.lax.scan(
                environment_step, runner_state, None, length=config.num_steps
            )
            train_state, environment_state, last_observation, key = runner_state
            _, _, last_value = model.apply(train_state.params, last_observation)

            def calculate_gae(carry, transition):
                advantage, next_value = carry
                not_done = 1.0 - transition.done.astype(jnp.float32)
                delta = transition.reward + config.gamma * next_value * not_done - transition.value
                advantage = delta + config.gamma * config.gae_lambda * not_done * advantage
                return (advantage, transition.value), advantage

            _, advantages = jax.lax.scan(
                calculate_gae,
                (jnp.zeros_like(last_value), last_value),
                trajectory,
                reverse=True,
            )
            targets = advantages + trajectory.value

            flat_trajectory = jax.tree.map(
                lambda value: value.reshape((config.batch_size,) + value.shape[2:]), trajectory
            )
            flat_advantages = advantages.reshape(config.batch_size)
            flat_targets = targets.reshape(config.batch_size)

            def update_epoch(update_state, _):
                train_state, key = update_state
                key, permutation_key = jax.random.split(key)
                permutation = jax.random.permutation(permutation_key, config.batch_size)
                shuffled = (
                    jax.tree.map(lambda value: value[permutation], flat_trajectory),
                    flat_advantages[permutation],
                    flat_targets[permutation],
                )
                minibatches = jax.tree.map(
                    lambda value: value.reshape(
                        (config.num_minibatches, minibatch_size) + value.shape[1:]
                    ),
                    shuffled,
                )

                def update_minibatch(train_state, batch):
                    batch_trajectory, batch_advantages, batch_targets = batch

                    def loss_function(parameters):
                        mean, log_std, value = model.apply(parameters, batch_trajectory.observation)
                        log_probability = squashed_log_probability(
                            mean, log_std, batch_trajectory.action
                        )
                        value_clipped = batch_trajectory.value + jnp.clip(
                            value - batch_trajectory.value,
                            -config.clip_epsilon,
                            config.clip_epsilon,
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(
                                jnp.square(value - batch_targets),
                                jnp.square(value_clipped - batch_targets),
                            ).mean()
                        )
                        normalized_advantage = (batch_advantages - batch_advantages.mean()) / (
                            batch_advantages.std() + 1e-8
                        )
                        ratio = jnp.exp(log_probability - batch_trajectory.log_probability)
                        unclipped_actor_loss = -ratio * normalized_advantage
                        clipped_actor_loss = (
                            -jnp.clip(
                                ratio,
                                1.0 - config.clip_epsilon,
                                1.0 + config.clip_epsilon,
                            )
                            * normalized_advantage
                        )
                        actor_loss = jnp.maximum(unclipped_actor_loss, clipped_actor_loss).mean()
                        entropy_estimate = -log_probability.mean()
                        total_loss = (
                            actor_loss
                            + config.value_coefficient * value_loss
                            - config.entropy_coefficient * entropy_estimate
                        )
                        return total_loss

                    loss, gradients = jax.value_and_grad(loss_function)(train_state.params)
                    return train_state.apply_gradients(grads=gradients), loss

                train_state, losses = jax.lax.scan(update_minibatch, train_state, minibatches)
                return (train_state, key), losses.mean()

            (train_state, key), epoch_losses = jax.lax.scan(
                update_epoch,
                (train_state, key),
                None,
                length=config.update_epochs,
            )
            completed_episodes = trajectory.returned_episode.sum()
            completed_return_sum = trajectory.returned_episode_return.sum()
            metrics = UpdateMetrics(
                loss=epoch_losses.mean(),
                mean_reward=trajectory.reward.mean(),
                completed_laps=trajectory.lap_complete.sum(),
                completed_episodes=completed_episodes,
                mean_completed_return=completed_return_sum / jnp.maximum(completed_episodes, 1),
            )
            return (
                train_state,
                environment_state,
                last_observation,
                key,
            ), metrics

        runner_state, metrics = jax.lax.scan(update_step, runner_state, None, length=num_updates)
        train_state, environment_state, final_observation, _ = runner_state
        return TrainOutput(
            train_state=train_state,
            environment_state=environment_state,
            final_observation=final_observation,
            metrics=metrics,
        )

    return train


def save_checkpoint(
    directory: str | Path,
    params: Any,
    config: PPOConfig,
    summary: dict[str, Any],
) -> Path:
    """Save model parameters and human-readable run metadata."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / "params.msgpack").write_bytes(serialization.to_bytes(params))
    metadata = {"ppo": asdict(config), "summary": summary}
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return path


def load_checkpoint(directory: str | Path, template_params: Any) -> Any:
    """Restore parameters using a matching initialized parameter tree."""

    payload = (Path(directory) / "params.msgpack").read_bytes()
    return serialization.from_bytes(template_params, payload)
