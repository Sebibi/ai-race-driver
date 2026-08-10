# JAX implementation guidance

- Functions transformed by `jit`, `vmap`, or `scan` must be pure and use JAX operations only.
- Thread PRNG keys explicitly; never hide global random state in an environment or trainer.
- Keep array ranks and batch sizes static inside compiled loops. Avoid data-dependent Python control flow.
- Do not call NumPy, SciPy, logging, file I/O, or host callbacks from a compiled hot path.
- Avoid implicit device synchronization inside rollouts and optimizer scans.
- Preserve corrected log probabilities when changing the bounded action distribution.
- Keep host-side track fitting separate from device-side geometry queries.
