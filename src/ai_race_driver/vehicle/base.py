"""Interface shared by vehicle dynamics implementations."""

from typing import Protocol, TypeVar

import jax

StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT", contravariant=True)


class VehicleModel(Protocol[StateT, ParamsT]):
    """Pure, JAX-transformable vehicle model contract."""

    def reset(
        self,
        position: jax.Array,
        heading: jax.Array,
        speed: jax.Array,
        params: ParamsT,
    ) -> StateT: ...

    def step(self, state: StateT, action: jax.Array, params: ParamsT) -> StateT: ...

    def position(self, state: StateT) -> jax.Array: ...

    def heading(self, state: StateT) -> jax.Array: ...

    def speed(self, state: StateT) -> jax.Array: ...
