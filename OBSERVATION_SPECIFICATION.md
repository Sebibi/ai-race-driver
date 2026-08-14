# Observation Specification for AI Race Driver

This document describes the observation space and the 14-element observation vector provided to the policy at each timestep.

## Overview

The observation space is a continuous 14-dimensional vector that provides the agent with:
- **Ego state**: Current speed, lateral position, heading alignment (4 elements)
- **Action history**: Previous action for temporal context (2 elements)
- **Track geometry preview**: Lookahead curvature samples (8 elements)

The observation is normalized to the range $[-1, 1]$ (approximately) to promote stable learning and reduce the policy's sensitivity to absolute scale.

## Observation Components

### Group 1: Ego State (Elements 0-3)

#### Element 0: Normalized Speed

```
obs[0] = vehicle_speed / vehicle.max_speed
```

**Definition:**
- `vehicle_speed`: Current speed in m/s (from point-mass model)
- `vehicle.max_speed`: Maximum achievable speed (vehicle parameter)

**Range:** [0, 1] when obeying speed limits; can exceed 1.0 if accelerating beyond max_speed

**Purpose:** Allows the policy to regulate speed via longitudinal action. Knowledge of current speed is essential for:
- Adapting steering behavior (high-speed turns require earlier planning)
- Deciding when to brake or accelerate
- Understanding available traction margins

---

#### Element 1: Normalized Lateral Error

```
obs[1] = lateral_error / half_width
```

**Definition:**
- `lateral_error`: Perpendicular distance from track centerline
  - Positive = vehicle right of centerline
  - Negative = vehicle left of centerline
- `half_width`: Half the track width (boundaries at ±half_width)

**Range:** [-1, 1] when on-track; ±∞ when off-track (but episodes terminate)

**Purpose:** Provides immediate feedback on lateral position. The policy uses this to:
- Steer toward centerline via lateral/turn action
- Avoid boundaries
- Maintain stable racing line

---

#### Element 2: Heading Error Sine

```
obs[2] = sin(heading_error)
```

**Definition:**
- `heading_error`: Angular difference between vehicle heading and track tangent at current position
- Both angles wrapped to [-π, π)

**Range:** [-1, 1] by construction (sine range)

**Purpose:** Sine encoding allows smooth learning of angular relationships and avoids discontinuities at ±π. The sine component encodes which direction to rotate to align with the track.

---

#### Element 3: Heading Error Cosine

```
obs[3] = cos(heading_error)
```

**Definition:**
- Same `heading_error` as Element 2

**Range:** [-1, 1] by construction (cosine range)

**Purpose:** Combined with sine (Element 2), cosine/sine encoding uniquely represents the full angle [-π, π) without discontinuities. Provides better gradients for angular alignment learning than raw angle values.

**Note:** Together, `obs[2:4] = [sin(ε), cos(ε)]` fully encodes heading error. The policy learns to rotate until both approach zero simultaneously.

---

### Group 2: Action History (Elements 4-5)

#### Element 4: Previous Longitudinal Action

```
obs[4] = previous_action[0]
```

**Definition:**
- `previous_action`: The 2D action taken in the previous timestep
- `action[0]`: Longitudinal component (acceleration/braking)

**Range:** [-1, 1] (clipped in environment step)

**Purpose:** Provides temporal context to the policy. The agent can:
- Decide whether to continue or change acceleration
- Learn smooth control sequences (aided by action_change_penalty)
- Understand its own momentum and control lag

---

#### Element 5: Previous Turning Action

```
obs[5] = previous_action[1]
```

**Definition:**
- `previous_action[1]`: Lateral/turning component (yaw rate control)

**Range:** [-1, 1] (clipped in environment step)

**Purpose:** Provides temporal context for steering decisions. The agent can:
- Adjust steering gradually for smooth turns
- Anticipate curves based on sustained steering
- Learn coordinated speed-steering behaviors

---

### Group 3: Track Geometry Preview (Elements 6-13)

#### Elements 6-13: Preview Curvature Array

```
obs[6:14] = clip(preview_curvature * half_width, -1.0, 1.0)
```

**Definition:**
- `preview_curvature`: Array of 8 curvature samples at lookahead distances
- Sampled distances (in meters): `[0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0]`
- `half_width`: Track half-width (normalization scale)

**Range:** [-1, 1] after clipping

**Curvature Details:**
- Curvature κ = 1/R where R is turn radius
- **Straight sections:** κ = 0.0
- **Left turns (counterclockwise):** κ > 0 (positive curvature)
- **Right turns (clockwise):** κ < 0 (negative curvature)
- **Tighter curves:** |κ| larger

**Normalization:** Multiplying by `half_width` converts from geometric curvature to an interpretable scale related to track width. Clipping to [-1, 1] ensures bounded features.

**Purpose:** Provides lookahead information about upcoming track geometry. The policy uses this to:
- Prepare steering inputs in advance (predictive control)
- Adjust speed for upcoming turns
- Avoid sharp, unprepared turns
- Learn anticipatory racing strategies

**Lookahead Structure:**
| Element | Distance | Purpose |
|---------|----------|---------|
| 6 | 0 m | Current position (immediate feedback) |
| 7 | 2 m | Near-term planning (< 0.5 seconds at typical speed) |
| 8 | 5 m | Short-term planning |
| 9 | 10 m | Medium-term planning |
| 10 | 15 m | Medium-term planning |
| 11 | 20 m | Long-term planning |
| 12 | 30 m | Strategic planning |
| 13 | 40 m | Distant planning (deep foresight) |

---

## Complete Observation Vector Structure

```python
obs = [
    # Ego state (normalized)
    normalized_speed,           # obs[0]  ∈ [0, 1] typical
    normalized_lateral_error,   # obs[1]  ∈ [-1, 1] on-track
    sin(heading_error),         # obs[2]  ∈ [-1, 1]
    cos(heading_error),         # obs[3]  ∈ [-1, 1]
    
    # Action history
    previous_longitudinal_action,  # obs[4]  ∈ [-1, 1]
    previous_turning_action,       # obs[5]  ∈ [-1, 1]
    
    # Track geometry lookahead (8 curvature samples)
    preview_curvature[0],        # obs[6]  @ 0 m
    preview_curvature[1],        # obs[7]  @ 2 m
    preview_curvature[2],        # obs[8]  @ 5 m
    preview_curvature[3],        # obs[9]  @ 10 m
    preview_curvature[4],        # obs[10] @ 15 m
    preview_curvature[5],        # obs[11] @ 20 m
    preview_curvature[6],        # obs[12] @ 30 m
    preview_curvature[7],        # obs[13] @ 40 m
]
```

---

## Observation Space Properties

### Shape
```python
shape = (14,)
dtype = jnp.float32
```

### Range (Approximate)
Most elements bounded in $[-1, 1]$:
- **Speed**: typically [0, 1.0], can exceed if over-accelerating
- **Lateral error**: [-1, 1] when on-track; terminates if exceeds
- **Heading error (sin/cos)**: exactly [-1, 1]
- **Previous actions**: exactly [-1, 1]
- **Preview curvatures**: exactly [-1, 1] (clipped)

### Space Definition (Gymnax)
```python
spaces.Box(
    low=-jnp.inf,
    high=jnp.inf,
    shape=(14,),
    dtype=jnp.float32,
)
```

The space is unbounded to accommodate occasional speed overshoots or off-track states during training.

---

## Design Rationale

### Why Normalized Speed?
Speed normalization ensures the policy receives consistent signals regardless of vehicle max_speed. A policy trained with normalized speed generalizes better to different dynamics.

### Why Sine/Cosine Heading?
Raw angle values create discontinuities at ±π (wrapping). Sine/cosine encoding:
- Avoids discontinuities (smooth gradients across wrap-around)
- Provides unique representation of ±π angles
- Enables the network to learn circular distance naturally

### Why Separate Action History?
Including previous actions allows the policy to:
- Learn smooth control sequences (useful with action_change_penalty)
- Account for control lag and momentum
- Develop consistent behavior across timesteps
- Benefit from temporal correlations in optimal control

### Why Curvature Preview?
Curvature lookahead enables anticipatory control:
- **Reactive policies** (only current state): slow, oscillatory, reactive steering
- **Predictive policies** (with preview): smooth, anticipatory, efficient steering
- Typical race drivers look ahead multiple car-lengths; the 8-sample preview mimics this.

### Why These Distances?
Lookahead distances scale with typical vehicle speed:
- **0-2 m**: Immediate (< 0.1 seconds at max speed)
- **2-10 m**: Tactical (0.1-0.5 seconds)
- **10-40 m**: Strategic (0.5-2 seconds)
- Maximum 40 m provides ~2 seconds lookahead, matching human reaction time horizons

### Why Clip Curvature to [-1, 1]?
Unbounded curvatures can exceed the policy network's learned value scales, leading to large prediction errors. Clipping normalizes the signal while preserving the direction and approximate magnitude of curvature.

---

## Observation Usage in Learning

### Policy Network Input
The 14-element observation is passed directly to the policy network (actor) at each step:
```python
action_mean, action_log_std = policy_network(obs)
action ~ Tanh(Normal(action_mean, exp(action_log_std)))
```

### Value Network Input
The observation is also passed to the value network (critic) for advantage estimation:
```python
value_estimate = value_network(obs)
```

### Temporal Sequences
Since PPO uses rollout collection without recurrence, the observation provides all temporal information needed. The previous action history (obs[4:6]) enables the network to infer control intent across timesteps.

---

## Observation Extraction Process (Code Flow)

1. **State update**: Vehicle step, track projection, error computation
2. **Normalization**: Speed, lateral error, heading error computed and normalized
3. **Preview sampling**: Curvature sampled at 8 lookahead distances via `vmap`
4. **Concatenation**: All components assembled into 14-element array
5. **Clipping & casting**: Curvature clipped to [-1, 1], cast to float32

See `get_obs()` method in [racing.py](../src/ai_race_driver/envs/racing.py) for implementation.

---

## Testing and Validation

### Unit Tests
See `tests/test_environment.py` for:
- Observation shape and dtype correctness
- Normalization bounds (on-track and edge cases)
- Curvature preview computation (periodic spline behavior)
- Action history tracking

### Integration Tests
See `tests/test_ppo.py` for:
- Learning with observations (policy convergence)
- Gradient flow through normalization

### Behavioral Validation
Use evaluation scripts to:
- Inspect raw observation traces during episode playback
- Visualize agent perception (speed, errors, preview curvature)
- Compare observation distributions in successful vs. failed trajectories
- Identify outliers or saturation in any component

---

## Common Modifications

### Adding Lateral Acceleration Feedback
```python
# obs[6] = lateral_acceleration / max_lateral_acceleration
# Shifts curvature preview to obs[7:15] (increases size to 15)
```
**Rationale:** Provides immediate feedback on control effectiveness.

### Adding Speed Rate-of-Change
```python
# obs[?] = speed_delta_from_last_step
# Useful for aggressive acceleration/braking learning
```

### Adding Distance-to-Boundary
```python
# obs[?] = lateral_error / half_width (already obs[1])
# Could add along-track distance to nearest turn
```

### Reducing Lookahead for Faster Inference
```python
# Use 4 curvature samples instead of 8 (obs size 10)
# Trade: reduced planning horizon for faster execution
```

---

## References

- **Observation Engineering:** François-Lavet et al., "An Introduction to Deep Reinforcement Learning" (2018)
- **Normalization & Scaling:** OpenAI Spinning Up in Deep RL
- **Curvature-based Control:** Coulter et al., "Implementation of the Smooth Pursuit Algorithms for Autonomous Vehicle Path Tracking" (1992)
- **Action History in RL:** Hausknecht & Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs" (2015)
