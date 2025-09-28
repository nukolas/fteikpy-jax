from __future__ import annotations

from typing import Tuple

import jax
from jax import numpy as jnp, lax

# Enable 64-bit for numerical parity with NumPy version
jax.config.update("jax_enable_x64", True)

Big = 1.0e5
eps = 1.0e-15


def _min4(a, b, c, d):
    return jnp.minimum(jnp.minimum(a, b), jnp.minimum(c, d))


def _t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero):
    di = dz * (i - zsa)
    dj = dx * (j - xsa)
    dk = dy * (k - ysa)
    return vzero * jnp.sqrt(di * di + dj * dj + dk * dk)


def _initialize_tt(slow, dz, dx, dy, zsrc, xsrc, ysrc) -> Tuple[jnp.ndarray, float]:
    nz, nx, ny = slow.shape

    # Convert src to grid position and guard edges
    zsa = zsrc / dz
    xsa = xsrc / dx
    ysa = ysrc / dy
    zsa = jnp.where(zsa >= nz, zsa - eps, zsa)
    xsa = jnp.where(xsa >= nx, xsa - eps, xsa)
    ysa = jnp.where(ysa >= ny, ysa - eps, ysa)

    zsi = jnp.minimum(jnp.floor(zsa).astype(jnp.int32), nz - 1)
    xsi = jnp.minimum(jnp.floor(xsa).astype(jnp.int32), nx - 1)
    ysi = jnp.minimum(jnp.floor(ysa).astype(jnp.int32), ny - 1)

    vzero = slow[zsi, xsi, ysi]

    # Work array is +1 along each axis (same convention as NumPy backend)
    tt = jnp.full((nz + 1, nx + 1, ny + 1), Big, dtype=jnp.float64)

    # Corners around source
    corners = jnp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=jnp.int32,
    )

    def set_corner(carry, idx):
        tt = carry
        di, dj, dk = corners[idx]
        i = zsi + di
        j = xsi + dj
        k = ysi + dk
        tval = _t_ana(i, j, k, dz, dx, dy, zsa, xsa, ysa, vzero)
        tt = tt.at[i, j, k].set(tval)
        return tt, None

    tt, _ = lax.scan(set_corner, tt, jnp.arange(corners.shape[0]))
    return tt, vzero


def _sweep_direction(tt, slow, dz, dx, dy, sgnvz, sgnvx, sgnvy, sgntz, sgntx, sgnty):
    """One directional sweep using fast sweeping update.

    The loop order is determined by sgnv* flags (0 means reverse along that axis
    for the loop variable). The finite difference upwind signs are sgnt*.
    """
    nz1, nx1, ny1 = tt.shape
    nz = nz1
    nx = nx1
    ny = ny1

    dz2i = 1.0 / (dz * dz)
    dx2i = 1.0 / (dx * dx)
    dy2i = 1.0 / (dy * dy)
    dz2dx2 = dz2i * dx2i
    dz2dy2 = dz2i * dy2i
    dx2dy2 = dx2i * dy2i
    dsum = dz2i + dx2i + dy2i

    # Loop ranges (1..N-1 inclusive for interior)
    def rng(n, reverse):
        lo = 1
        hi = n  # exclusive in fori_loop (so stop at n-1)
        idxs = jnp.arange(lo, hi)
        return jnp.flip(idxs) if reverse else idxs

    ks = rng(ny, sgnvy == 0)
    js = rng(nx, sgnvx == 0)
    is_ = rng(nz, sgnvz == 0)

    def body_k(tt, k):
        def body_j(tt, j):
            def body_i(tt, i):
                i1 = i - sgnvz
                j1 = j - sgnvx
                k1 = k - sgnvy

                # Neighbor indices guarded within valid range for slow
                im = jnp.clip(i1, 0, slow.shape[0] - 1)
                jm = jnp.clip(j1, 0, slow.shape[1] - 1)
                km = jnp.clip(k1, 0, slow.shape[2] - 1)

                tv = tt[i - sgntz, j, k]
                te = tt[i, j - sgntx, k]
                tn = tt[i, j, k - sgnty]
                tev = tt[i - sgntz, j - sgntx, k]
                ten = tt[i, j - sgntx, k - sgnty]
                tnv = tt[i - sgntz, j, k - sgnty]
                tnve = tt[i - sgntz, j - sgntx, k - sgnty]

                # 1D operators (refracted times)
                vref1 = _min4(
                    slow[im, jnp.clip(j - 1, 0, slow.shape[1] - 2), jnp.clip(k - 1, 0, slow.shape[2] - 2)],
                    slow[im, jnp.clip(j - 1, 0, slow.shape[1] - 2), jnp.clip(k, 0, slow.shape[2] - 2)],
                    slow[im, jnp.clip(j, 0, slow.shape[1] - 2), jnp.clip(k - 1, 0, slow.shape[2] - 2)],
                    slow[im, jnp.clip(j, 0, slow.shape[1] - 2), jnp.clip(k, 0, slow.shape[2] - 2)],
                )
                t1d1 = tv + dz * vref1

                vref2 = _min4(
                    slow[jnp.clip(i - 1, 0, slow.shape[0] - 2), jm, jnp.clip(k - 1, 0, slow.shape[2] - 2)],
                    slow[jnp.clip(i, 0, slow.shape[0] - 2), jm, jnp.clip(k - 1, 0, slow.shape[2] - 2)],
                    slow[jnp.clip(i - 1, 0, slow.shape[0] - 2), jm, jnp.clip(k, 0, slow.shape[2] - 2)],
                    slow[jnp.clip(i, 0, slow.shape[0] - 2), jm, jnp.clip(k, 0, slow.shape[2] - 2)],
                )
                t1d2 = te + dx * vref2

                vref3 = _min4(
                    slow[jnp.clip(i - 1, 0, slow.shape[0] - 2), jnp.clip(j - 1, 0, slow.shape[1] - 2), km],
                    slow[jnp.clip(i - 1, 0, slow.shape[0] - 2), jnp.clip(j, 0, slow.shape[1] - 2), km],
                    slow[jnp.clip(i, 0, slow.shape[0] - 2), jnp.clip(j - 1, 0, slow.shape[1] - 2), km],
                    slow[jnp.clip(i, 0, slow.shape[0] - 2), jnp.clip(j, 0, slow.shape[1] - 2), km],
                )
                t1d3 = tn + dy * vref3

                t1d = jnp.minimum(t1d1, jnp.minimum(t1d2, t1d3))

                # 2D operators using conditional formulas
                t2d1_vref = jnp.minimum(
                    slow[im, jm, jnp.clip(k - 1, 0, slow.shape[2] - 2)],
                    slow[im, jm, jnp.clip(k, 0, slow.shape[2] - 2)],
                )
                cond1 = jnp.logical_and(tv < te + dx * t2d1_vref, te < tv + dz * t2d1_vref)
                ta1 = tev + te - tv
                tb1 = tev - te + tv
                disc1 = 4.0 * t2d1_vref * t2d1_vref * (dz2i + dx2i) - dz2i * dx2i * (ta1 - tb1) ** 2.0
                t2d1 = (tb1 * dz2i + ta1 * dx2i + jnp.sqrt(jnp.maximum(0.0, disc1))) / (dz2i + dx2i)
                t2d1 = jnp.where(cond1, t2d1, Big)

                t2d2_vref = jnp.minimum(
                    slow[im, jnp.clip(j - 1, 0, slow.shape[1] - 2), km],
                    slow[im, jnp.clip(j, 0, slow.shape[1] - 2), km],
                )
                cond2 = jnp.logical_and(tv < tn + dy * t2d2_vref, tn < tv + dz * t2d2_vref)
                ta2 = tv - tn + tnv
                tb2 = tn - tv + tnv
                disc2 = 4.0 * t2d2_vref * t2d2_vref * (dz2i + dy2i) - dz2i * dy2i * (ta2 - tb2) ** 2.0
                t2d2 = (ta2 * dz2i + tb2 * dy2i + jnp.sqrt(jnp.maximum(0.0, disc2))) / (dz2i + dy2i)
                t2d2 = jnp.where(cond2, t2d2, Big)

                t2d3_vref = jnp.minimum(
                    slow[jnp.clip(i - 1, 0, slow.shape[0] - 2), jm, km],
                    slow[jnp.clip(i, 0, slow.shape[0] - 2), jm, km],
                )
                cond3 = jnp.logical_and(te < tn + dy * t2d3_vref, tn < te + dx * t2d3_vref)
                ta3 = te - tn + ten
                tb3 = tn - te + ten
                disc3 = 4.0 * t2d3_vref * t2d3_vref * (dx2i + dy2i) - dx2i * dy2i * (ta3 - tb3) ** 2.0
                t2d3 = (ta3 * dx2i + tb3 * dy2i + jnp.sqrt(jnp.maximum(0.0, disc3))) / (dx2i + dy2i)
                t2d3 = jnp.where(cond3, t2d3, Big)

                t2d = jnp.minimum(t2d1, jnp.minimum(t2d2, t2d3))

                # 3D operator
                vref3d = slow[im, jm, km]
                cond3d = jnp.minimum(t1d, t2d) > jnp.maximum(jnp.maximum(tv, te), tn)
                ta = te - 0.5 * tn + 0.5 * ten - 0.5 * tv + 0.5 * tev - tnv + tnve
                tb = tv - 0.5 * tn + 0.5 * tnv - 0.5 * te + 0.5 * tev - ten + tnve
                tc = tn - 0.5 * te + 0.5 * ten - 0.5 * tv + 0.5 * tnv - tev + tnve

                t2 = vref3d * vref3d * dsum * 9.0
                t3 = dz2dx2 * (ta - tb) ** 2.0 + dz2dy2 * (tb - tc) ** 2.0 + dx2dy2 * (ta - tc) ** 2.0
                disc3d = jnp.maximum(0.0, t2 - t3)
                t1 = tb * dz2i + ta * dx2i + tc * dy2i
                t3d = (t1 + jnp.sqrt(disc3d)) / dsum
                t3d = jnp.where(cond3d, t3d, Big)

                tnew = jnp.minimum(jnp.minimum(tt[i, j, k], t1d), jnp.minimum(t2d, t3d))
                tt = tt.at[i, j, k].set(tnew)
                return tt

            return lax.fori_loop(0, is_.shape[0], lambda _, t: body_i(t, is_[_]), tt)

        return lax.fori_loop(0, js.shape[0], lambda _, t: body_j(t, js[_]), tt)

    return lax.fori_loop(0, ks.shape[0], lambda _, t: body_k(t, ks[_]), tt)


@jax.jit
def _sweep8(tt, slow, dz, dx, dy):
    # Eight canonical sweep directions from the NumPy backend
    # Top->Bottom; West->East; South->North
    tt = _sweep_direction(tt, slow, dz, dx, dy, 1, 1, 1, 1, 1, 1)
    # Top->Bottom; East->West; South->North
    tt = _sweep_direction(tt, slow, dz, dx, dy, 1, 0, 1, 1, -1, 1)
    # Top->Bottom; West->East; North->South
    tt = _sweep_direction(tt, slow, dz, dx, dy, 1, 1, 0, 1, 1, -1)
    # Top->Bottom; East->West; North->South
    tt = _sweep_direction(tt, slow, dz, dx, dy, 1, 0, 0, 1, -1, -1)
    # Bottom->Top; West->East; South->North
    tt = _sweep_direction(tt, slow, dz, dx, dy, 0, 1, 1, -1, 1, 1)
    # Bottom->Top; East->West; South->North
    tt = _sweep_direction(tt, slow, dz, dx, dy, 0, 0, 1, -1, -1, 1)
    # Bottom->Top; West->East; North->South
    tt = _sweep_direction(tt, slow, dz, dx, dy, 0, 1, 0, -1, 1, -1)
    # Bottom->Top; East->West; North->South
    tt = _sweep_direction(tt, slow, dz, dx, dy, 0, 0, 0, -1, -1, -1)
    return tt


def solve3d_jax(slow: jnp.ndarray, dz: float, dx: float, dy: float, src: jnp.ndarray, nsweep: int = 2) -> jnp.ndarray:
    """Solve 3D Eikonal using a JAX fast-sweeping implementation.

    Parameters
    - slow: slowness grid (nz, nx, ny) [s/m]
    - dz, dx, dy: grid spacings
    - src: source location in physical units (z, x, y)
    - nsweep: number of 8-direction sweep cycles

    Returns
    - tt: traveltime grid with shape (nz+1, nx+1, ny+1)
    """
    nz, nx, ny = slow.shape
    condz = jnp.logical_and(0.0 <= src[0], src[0] <= dz * nz)
    condx = jnp.logical_and(0.0 <= src[1], src[1] <= dx * nx)
    condy = jnp.logical_and(0.0 <= src[2], src[2] <= dy * ny)
    ok = jnp.logical_and(jnp.logical_and(condz, condx), condy)
    def _raise(_):
        return (_ for _ in ()).throw(ValueError("source out of bound"))
    if not bool(ok):
        _raise(None)

    tt, _ = _initialize_tt(slow, dz, dx, dy, src[0], src[1], src[2])

    def one_cycle(tt, _):
        return _sweep8(tt, slow, dz, dx, dy), None

    tt, _ = lax.scan(one_cycle, tt, jnp.arange(nsweep))
    return tt


def eikonal3d_jax(velocity: jnp.ndarray, gridsize: Tuple[float, float, float], source: jnp.ndarray, origin: jnp.ndarray | None = None, nsweep: int = 2) -> jnp.ndarray:
    """Convenience wrapper taking velocity to traveltime via JAX.

    Parameters
    - velocity: velocity model (nz, nx, ny)
    - gridsize: (dz, dx, dy)
    - source: (z, x, y) in world coords
    - origin: grid origin (z0, x0, y0). If None, assumes zeros.
    - nsweep: number of sweep cycles
    """
    dz, dx, dy = gridsize
    origin = jnp.zeros(3, dtype=jnp.float64) if origin is None else jnp.asarray(origin, dtype=jnp.float64)
    slow = 1.0 / jnp.asarray(velocity, dtype=jnp.float64)
    src_local = jnp.asarray(source, dtype=jnp.float64) - origin
    return solve3d_jax(slow, dz, dx, dy, src_local, nsweep)

