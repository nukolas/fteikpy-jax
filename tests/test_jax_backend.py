from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

import fteikpy

solve3d_jax = getattr(fteikpy, "solve3d_jax", None)
misfit_l2_jax = getattr(fteikpy, "misfit_l2_jax", None)
grad_misfit_l2_jax = getattr(fteikpy, "grad_misfit_l2_jax", None)

if solve3d_jax is None or misfit_l2_jax is None or grad_misfit_l2_jax is None:  # pragma: no cover - optional dependency
    pytest.skip("JAX backend not available", allow_module_level=True)


def analytic_time(indices: np.ndarray, gridsize: tuple[float, float, float], velocity: float) -> np.ndarray:
    """Return analytical traveltime at grid indices for constant velocity."""
    dz, dx, dy = gridsize
    z = indices[..., 0] * dz
    x = indices[..., 1] * dx
    y = indices[..., 2] * dy
    return np.sqrt(z * z + x * x + y * y) / velocity


def test_jax_solver_constant_velocity_matches_analytic():
    velocity = 2000.0  # m/s
    gridsize = (10.0, 10.0, 10.0)
    nz, nx, ny = 3, 3, 3

    slow = np.full((nz, nx, ny), 1.0 / velocity, dtype=np.float64)
    source = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    tt = solve3d_jax(slow, *gridsize, source, nsweep=4)
    tt_host = np.asarray(tt)

    grid_indices = np.stack(np.meshgrid(
        np.arange(nz + 1, dtype=np.float64),
        np.arange(nx + 1, dtype=np.float64),
        np.arange(ny + 1, dtype=np.float64),
        indexing="ij",
    ), axis=-1)
    expected = analytic_time(grid_indices, gridsize, velocity)

    # Only compare interior nodes (excluding the extra padding layer at the far boundary)
    assert np.allclose(tt_host[: nz + 1, : nx + 1, : ny + 1], expected, atol=1.0e-6)


def test_jax_misfit_gradient_zero_for_exact_data():
    velocity = jnp.ones((4, 4, 4), dtype=jnp.float64) * 2500.0
    gridsize = (20.0, 20.0, 20.0)
    sources = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float64)
    receivers = jnp.array(
        [
            [20.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
            [0.0, 0.0, 20.0],
            [20.0, 20.0, 20.0],
        ],
        dtype=jnp.float64,
    )

    # Analytical travel times for homogeneous medium
    distances = jnp.linalg.norm(receivers, axis=1)
    t_obs = (distances / velocity[0, 0, 0]).reshape(1, -1)

    misfit = misfit_l2_jax(velocity, gridsize, sources, receivers, t_obs, nsweep=4)
    grad = grad_misfit_l2_jax(velocity, gridsize, sources, receivers, t_obs, nsweep=4)

    assert pytest.approx(float(misfit), abs=1.0e-8) == 0.0
    assert np.allclose(np.asarray(grad), 0.0, atol=1.0e-8)

