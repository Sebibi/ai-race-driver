# Reward Specification for AI Race Driver

This document describes the reward components and shaping strategy used to train the PPO racing agent.

## Overview

The total reward per step is a shaped combination of:
1. **Primary objective**: Forward progress on the track
2. **Behavior shaping**: Penalties for deviations from ideal driving
3. **Terminal events**: Bonuses/penalties for lap completion and off-track crashes

## Reward Components

### 1. Progress Reward (Primary Objective)

```
R_progress = normalized_progress
```

**Definition:**
- `progress_delta`: Signed distance traveled along the track centerline in one timestep
- `normalized_progress`: `progress_delta / (max_speed × dt)`

**Purpose:** Incentivizes the agent to move forward as fast as possible. This is the primary reward signal that drives learning.

**Normalization:** By dividing by `max_speed × dt`, the reward scales consistently regardless of vehicle dynamics. A single timestep of forward motion at maximum speed yields approximately +1.0 reward.

**Clipping:** Progress delta is clipped to `[-max_step_progress, +max_step_progress]` where `max_step_progress = max_speed × dt × 1.5` to prevent large unrealistic jumps from numerical issues or extreme resets.

---

### 2. Lateral Deviation Penalty

```
R_lateral = -lateral_penalty × (lateral_error / half_width)²
```

**Definition:**
- `lateral_error`: Perpendicular distance from track centerline (positive = right, negative = left)
- `half_width`: Half the track width (boundaries are at ±half_width)
- `normalized_lateral`: `lateral_error / half_width` (range: [-1, 1] when on-track)

**Purpose:** Discourages the agent from drifting away from the centerline. Driving straight is more stable and safer than weaving.

**Quadratic Form:** The squared term means:
- Small deviations incur small penalties (smooth learning)
- Large deviations near the boundary incur increasingly steep penalties
- Example: 0.5 half-widths off-center costs 0.25× the penalty weight

**Default Value:** `lateral_penalty = 0.0` (disabled by default to allow natural lane-finding behavior)

---

### 3. Heading Alignment Penalty

```
R_heading = -heading_penalty × (heading_error / π)²
```

**Definition:**
- `heading_error`: Angular difference between vehicle heading and track tangent
- `normalized_heading`: `heading_error / π` (range: [-1, 1] when at ±π radians)
- Both angles are wrapped to [-π, π)

**Purpose:** Encourages the vehicle to face along the track direction rather than driving sideways or backward. Improves steering efficiency and control.

**Quadratic Form:** Similar to lateral penalty, squared error provides smooth shaping.

**Default Value:** `heading_penalty = 0.0` (disabled by default)

---

### 4. Action Smoothness Penalty

```
R_action_change = -action_change_penalty × Σ(action - previous_action)²
```

**Definition:**
- `action`: Current 2D action vector [longitudinal, turn] in range [-1, 1]
- `previous_action`: Action from the previous timestep
- Summed over both action dimensions

**Purpose:** Penalizes abrupt changes in acceleration or steering. Encourages smooth, continuous control that is more physically realistic and often more efficient.

**Effect:**
- A sudden +1.0 change in any action dimension costs `action_change_penalty × 1.0`
- Gradual changes accumulate less penalty over time
- Essential for preventing jitter and oscillatory behavior

**Default Value:** `action_change_penalty = 0.0` (disabled by default)

---

### 5. Off-Track Penalty

```
R_off_track = -off_track_penalty × off_track_flag
```

**Definition:**
- `off_track`: Boolean, True if `|lateral_error| > half_width`
- `off_track_penalty`: Scalar multiplier

**Timing:** Applied when termination occurs (immediate upon exiting track bounds)

**Purpose:** Strongly discourages driving off the track. Defines the constraint boundary.

**Default Value:** `off_track_penalty = 1.0` (enabled, -1.0 reward on crash)

**Effect:** An off-track crash immediately terminates the episode and applies a fixed -1.0 penalty, forming a hard constraint.

---

### 6. Lap Completion Bonus

```
R_lap_bonus = lap_bonus × lap_complete_flag
```

**Definition:**
- `lap_complete`: Boolean, True if `accumulated_progress >= track.length`
- `lap_bonus`: Scalar multiplier

**Timing:** Applied when the agent completes the first lap (or multiple laps if the episode continues)

**Purpose:** Provides a large reward signal for reaching the goal. Acts as a shaped objective that doesn't require trial-and-error discovery.

**Default Value:** `lap_bonus = 1.0` (enabled, +1.0 reward per lap)

**Effect:** Completing a lap provides a substantial one-time reward. Combined with progress reward throughout the lap, this creates a clear gradient toward the goal.

---

## Total Reward Equation

At each timestep, the agent receives:

```
R_total = R_progress 
        + R_lateral 
        + R_heading 
        + R_action_change 
        + R_off_track × [1 if off_track else 0]
        + R_lap_bonus × [1 if lap_complete else 0]
```

Simplified:
```
R_total = normalized_progress
        - lateral_penalty × (normalized_lateral)²
        - heading_penalty × (normalized_heading)²
        - action_change_penalty × Σ(Δaction)²
        - off_track_penalty × [off_track]
        + lap_bonus × [lap_complete]
```

---

## Configuration via RacingEnvParams

All reward weights are configurable through `RacingEnvParams`:

```python
@struct.dataclass
class RacingEnvParams(environment.EnvParams):
    # ... other fields ...
    lateral_penalty: float = 0.0          # Lateral deviation penalty weight
    heading_penalty: float = 0.0          # Heading misalignment penalty weight
    action_change_penalty: float = 0.0    # Action smoothness penalty weight
    off_track_penalty: float = 1.0        # Off-track crash penalty
    lap_bonus: float = 1.0                # Lap completion bonus
```

### Usage Example

```python
from ai_race_driver.envs.racing import RacingEnv, RacingEnvParams

# Default config: pure progress-based reward
params = RacingEnvParams()

# Add behavior shaping
params = RacingEnvParams(
    lateral_penalty=0.1,        # Gently discourage lateral drift
    heading_penalty=0.05,       # Mildly discourage misalignment
    action_change_penalty=0.01, # Small smoothness bonus
    off_track_penalty=1.0,      # Hard constraint (unchanged)
    lap_bonus=1.0,              # Large goal bonus (unchanged)
)

env = RacingEnv(params)
```

---

## Reward Shaping Strategy

### Phase 1: Progress-Based Learning (Default)
- Enable: `progress_reward` (always enabled)
- Disable: all penalties and bonuses
- **Rationale:** Pure progress creates a simple gradient. The agent learns to move forward without premature behavioral constraints.

### Phase 2: Constraint Introduction
- Add: `off_track_penalty` to define boundary
- Add: `lap_bonus` to emphasize goal
- **Rationale:** Once basic forward motion works, hard constraints and goal signals guide behavior toward valid solutions.

### Phase 3: Behavior Refinement (Optional)
- Add: `lateral_penalty`, `heading_penalty`, `action_change_penalty`
- **Rationale:** Fine-tune driving style for stability, efficiency, or smoothness once the agent has learned the core task.

---

## Numerical Ranges and Scales

### Typical Reward Magnitudes (per timestep)

| Component | Min | Typical | Max | Note |
|-----------|-----|---------|-----|------|
| Progress | -1.5 | 0.3-0.7 | +1.0 | Backward speed penalized by clipping |
| Lateral penalty | 0 | -0.025 | -0.1 | Quadratic; worse toward boundary |
| Heading penalty | 0 | -0.01 | -0.1 | Quadratic; worst at ±π misalignment |
| Action change | 0 | -0.001 | -0.1 | Depends on action volatility |
| Off-track | 0 | 0 | -1.0 | Only on termination |
| Lap bonus | 0 | 0 | +1.0 | Only once per lap |

### Episode-Level Cumulative

- **Perfect lap (no penalties):** ~400-600 (2000-step episode × 0.2-0.3 average reward)
- **With penalized deviation:** -50 to -200 additional cost
- **Off-track crash:** -1.0 (plus loss of future progress rewards)

---

## Design Rationale

### Why Quadratic Penalties?
Quadratic error terms provide smooth learning curves. Linear penalties (|error|) create discontinuous gradients at zero error. Quadratic penalties encourage smooth convergence to the ideal trajectory.

### Why Normalize by Vehicle Dynamics?
The normalized progress reward scales consistently across different vehicle speeds and timestep sizes. Without normalization, a slow vehicle receives lower rewards, biasing learning incorrectly.

### Why Separate Progress and Lap Bonus?
- **Progress reward:** Provides dense, continuous feedback at every step
- **Lap bonus:** Provides a discrete milestone signal
- Together, they create both intrinsic motivation (make progress) and extrinsic goals (complete lap)

### Why Default to Zero for Shaping Penalties?
The environment defaults to pure progress reward (shaping penalties disabled). This:
- Simplifies the learning problem initially
- Allows the agent to discover its own efficient style
- Reduces hyperparameter search space
- Enables easy ablation studies

Shaping penalties can be enabled when and if specific behaviors need to be encouraged.

---

## Testing and Validation

### Unit Tests
See `tests/test_environment.py` for:
- Reward calculation correctness
- Boundary condition behavior (at-track-limit, off-track transitions)
- Terminal reward application (lap complete, off-track)
- Cumulative episode return tracking

### Integration Tests
See `tests/test_ppo.py` for:
- Learning curves with different reward configurations
- Convergence behavior with and without shaping penalties
- Policy behavior (does lap completion happen? are crashes minimized?)

### Behavioral Validation
Use evaluation scripts (`ai-race-eval`) to:
- Inspect generated trajectories and reward traces
- Visualize progress, lateral error, and heading error over time
- Compare policies trained with different reward weights

---

## References

- **Reward Shaping:** Ng et al., "Policy Invariance Under Reward Transformations" (1999)
- **PPO Training:** Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
- **Continuous Control:** Lillicrap et al., "Continuous Control with Deep Reinforcement Learning" (2016)
