# Test guidance

- Default tests must be deterministic, CPU-friendly, and use small batch/rollout shapes.
- Test uncompiled, `jit`, and `vmap` behavior for new physics or geometry operations.
- Use fixed PRNG seeds and assert finite values, shapes, state invariants, and terminal semantics.
- Mark full convergence or hardware throughput tests `slow`; keep them opt-in locally.
