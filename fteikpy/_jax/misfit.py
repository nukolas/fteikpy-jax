from __future__ import annotations

from typing import Tuple

import jax
from jax import numpy as jnp, lax

from .eikonal3d import solve3d_jax


def _trilinear_sample(tt: jnp.ndarray, gridsize: Tuple[float, float, float], points_local: jnp.ndarray) -> jnp.ndarray:
    """Trilinear interpolation of traveltime grid at local grid coords.

    Parameters
    - tt: traveltime grid (nz+1, nx+1, ny+1)
    - gridsize: (dz, dx, dy)
    - points_local: array (..., 3) of physical coords relative to origin
    Returns: array (...) of interpolated traveltimes
    """
    dz, dx, dy = gridsize
    z = points_local[..., 0] / dz
    x = points_local[..., 1] / dx
    y = points_local[..., 2] / dy

    nz1, nx1, ny1 = tt.shape
    # Clamp to interior voxels for base corner
    iz = jnp.clip(jnp.floor(z).astype(jnp.int32), 0, nz1 - 2)
    ix = jnp.clip(jnp.floor(x).astype(jnp.int32), 0, nx1 - 2)
    iy = jnp.clip(jnp.floor(y).astype(jnp.int32), 0, ny1 - 2)

    wz = z - iz
    wx = x - ix
    wy = y - iy

    # Gather 8 corners
    def g(di, dj, dk):
        return tt[iz + di, ix + dj, iy + dk]

    c000 = g(0, 0, 0)
    c100 = g(1, 0, 0)
    c010 = g(0, 1, 0)
    c110 = g(1, 1, 0)
    c001 = g(0, 0, 1)
    c101 = g(1, 0, 1)
    c011 = g(0, 1, 1)
    c111 = g(1, 1, 1)

    c00 = c000 * (1 - wz) + c100 * wz
    c01 = c001 * (1 - wz) + c101 * wz
    c10 = c010 * (1 - wz) + c110 * wz
    c11 = c011 * (1 - wz) + c111 * wz
    c0 = c00 * (1 - wx) + c10 * wx
    c1 = c01 * (1 - wx) + c11 * wx
    c = c0 * (1 - wy) + c1 * wy
    return c


def misfit_l2_jax(
    velocity: jnp.ndarray,
    gridsize: Tuple[float, float, float],
    sources: jnp.ndarray,
    receivers: jnp.ndarray,
    t_obs: jnp.ndarray,
    origin: jnp.ndarray | None = None,
    nsweep: int = 2,
) -> jnp.ndarray:
    """L2 traveltime misfit summed over sources and receivers.

    Assumes all sources share the same receiver set. Shapes:
    - velocity: (nz, nx, ny)
    - sources: (Ns, 3)
    - receivers: (Nr, 3)
    - t_obs: (Ns, Nr)
    Returns scalar 0.5 * ||t_pred - t_obs||^2
    """
    velocity = jnp.asarray(velocity, dtype=jnp.float64)
    dz, dx, dy = gridsize
    origin = jnp.zeros(3, dtype=jnp.float64) if origin is None else jnp.asarray(origin, dtype=jnp.float64)
    sources = jnp.asarray(sources, dtype=jnp.float64)
    receivers = jnp.asarray(receivers, dtype=jnp.float64)
    t_obs = jnp.asarray(t_obs, dtype=jnp.float64)

    slow = 1.0 / velocity

    src_local_all = sources - origin
    rec_local = receivers - origin

    def body_fun(acc, i):
        src_local = src_local_all[i]
        tt = solve3d_jax(slow, dz, dx, dy, src_local, nsweep)
        t_pred = _trilinear_sample(tt, (dz, dx, dy), rec_local)
        mis = 0.5 * jnp.sum((t_pred - t_obs[i]) ** 2.0)
        return acc + mis, None

    Ns = sources.shape[0]
    misfit, _ = lax.scan(body_fun, 0.0, jnp.arange(Ns))
    return misfit


grad_misfit_l2_jax = jax.grad(misfit_l2_jax)
