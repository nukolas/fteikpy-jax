from .__about__ import __version__
from ._grid import Grid2D, Grid3D, TraveltimeGrid2D, TraveltimeGrid3D
from ._helpers import get_num_threads, set_num_threads
from ._io import grid_to_meshio, ray_to_meshio
from ._solver import Eikonal2D, Eikonal3D

# Optional JAX backend
try:  # pragma: no cover - optional dependency
    from ._jax import (
        solve3d_jax,
        eikonal3d_jax,
        misfit_l2_jax,
        grad_misfit_l2_jax,
    )
except Exception:  # noqa: BLE001
    solve3d_jax = None  # type: ignore
    eikonal3d_jax = None  # type: ignore
    misfit_l2_jax = None  # type: ignore
    grad_misfit_l2_jax = None  # type: ignore

__all__ = [
    "Eikonal2D",
    "Eikonal3D",
    "Grid2D",
    "Grid3D",
    "TraveltimeGrid2D",
    "TraveltimeGrid3D",
    "get_num_threads",
    "set_num_threads",
    "grid_to_meshio",
    "ray_to_meshio",
    "__version__",
    # JAX backend (if available)
    "solve3d_jax",
    "eikonal3d_jax",
    "misfit_l2_jax",
    "grad_misfit_l2_jax",
]
