"""Flatten a 3D surface patch to 2D with a Least Squares Conformal Map.

This is a from-scratch implementation of LSCM (Levy et al. 2002),
kept deliberately small and explicit:

Each triangle is expressed in its own 2D frame (see meshutil.triangle_frames).
A linear map from that frame to (u, v) is conformal exactly when, writing
uv as the complex number z = u + iv and the frame corners as complex
numbers p_j, the weighted sum over the triangle's corners vanishes:

    sum_j  e_j * z_j = 0        with  e_j = p_{j+2} - p_{j+1}

(e_j is the edge opposite corner j). LSCM minimises the total failure of
that condition, sum over triangles of |sum_j e_j z_j|^2 / A_t, which is a
sparse linear least-squares problem once two vertices are pinned to fix
translation, rotation and scale.

The result is validated in tests against developable surfaces (which must
flatten with ~zero distortion) and against libigl's reference LSCM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import trimesh

from flatpack.distortion import DistortionReport, distortion_report
from flatpack.meshutil import (
    boundary_loops,
    farthest_pair,
    triangle_frames,
    unique_edges,
)


@dataclass
class FlattenResult:
    """Flattened panel: per-vertex uv coordinates plus distortion metrics."""

    uv: np.ndarray  # (n, 2), same units as the input mesh
    distortion: DistortionReport
    pins: tuple[int, int]  # vertex indices that anchored the map


def lscm(
    vertices: np.ndarray,
    faces: np.ndarray,
    pins: tuple[int, int] | None = None,
) -> np.ndarray:
    """Least Squares Conformal Map of a disk-like surface patch.

    vertices: (n, 3) float array.
    faces: (t, 3) int array. The patch must have a boundary (not be closed).
    pins: two vertex indices to anchor; defaults to the two most distant
        boundary vertices.

    Returns (n, 2) uv coordinates, uniformly rescaled so total uv area
    equals total 3D area (a conformal map only fixes scale up to the pins).
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    n = len(vertices)

    require_disk(vertices, faces)
    if pins is None:
        pins = default_pins(vertices, faces)
    pin_a, pin_b = pins
    if pin_a == pin_b:
        raise ValueError("the two pinned vertices must differ")

    x1, x2, y2, area = triangle_frames(vertices, faces)

    # Complex corner positions in the local frame: p0 = 0, p1 = x1, p2 = x2 + i y2.
    # Opposite edges e_j = p_{j+2} - p_{j+1}, weighted by 1/sqrt(A) so every
    # triangle's conformal energy counts per unit area.
    w = 1.0 / np.sqrt(area)
    e = np.empty((len(faces), 3), dtype=complex)
    e[:, 0] = (x2 - x1) + 1j * y2
    e[:, 1] = -x2 - 1j * y2
    e[:, 2] = x1
    e *= w[:, None]

    # Real least-squares system over unknowns [u_0..u_{n-1}, v_0..v_{n-1}]:
    # each triangle contributes two rows (real and imaginary part of the
    # conformality condition).
    t = len(faces)
    rows = np.repeat(np.arange(t) * 2, 3)
    cols_u = faces.ravel()
    er = e.real.ravel()
    ei = e.imag.ravel()
    coo_rows = np.concatenate([rows, rows, rows + 1, rows + 1])
    coo_cols = np.concatenate([cols_u, cols_u + n, cols_u, cols_u + n])
    coo_vals = np.concatenate([er, -ei, ei, er])
    full = scipy.sparse.coo_matrix(
        (coo_vals, (coo_rows, coo_cols)), shape=(2 * t, 2 * n)
    ).tocsc()

    # Pin vertex a at (0, 0) and vertex b at (d, 0) with d their 3D distance.
    d = float(np.linalg.norm(vertices[pin_b] - vertices[pin_a]))
    pinned_cols = np.array([pin_a, pin_b, pin_a + n, pin_b + n])
    pinned_vals = np.array([0.0, d, 0.0, 0.0])
    free_mask = np.ones(2 * n, dtype=bool)
    free_mask[pinned_cols] = False

    a_free = full[:, free_mask]
    rhs = -full[:, pinned_cols] @ pinned_vals

    # Normal equations: small, symmetric positive definite with two pins.
    ata = (a_free.T @ a_free).tocsc()
    solution = scipy.sparse.linalg.spsolve(ata, a_free.T @ rhs)

    unknowns = np.empty(2 * n)
    unknowns[free_mask] = solution
    unknowns[pinned_cols] = pinned_vals
    uv = np.column_stack([unknowns[:n], unknowns[n:]])

    # If the map came out mirrored (negative uv areas), flip it back.
    uv_signed = _signed_uv_areas(uv, faces)
    if uv_signed.sum() < 0:
        uv[:, 1] = -uv[:, 1]
        uv_signed = -uv_signed

    # Uniform rescale so total area is preserved; the pins only set an
    # arbitrary scale.
    scale = np.sqrt(area.sum() / uv_signed.sum())
    center = uv.mean(axis=0)
    return (uv - center) * scale + center


def require_disk(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Reject patches that cannot flatten because of their topology.

    A conformal map to the plane needs disk topology: Euler characteristic
    V - E + F == 1. A closed surface (sphere-like shell, chi == 2) or a
    tube that still wraps around (chi == 0, e.g. a pack body whose seam
    didn't open it) would flatten to overlapping garbage; better to say
    why up front.
    """
    used = np.unique(faces)
    chi = len(used) - len(unique_edges(faces)) + len(faces)
    if chi == 1:
        return
    if not boundary_loops(faces):
        raise ValueError(
            "patch is a closed surface; cut it with seams before flattening"
        )
    raise ValueError(
        f"panel is not a flattenable patch (Euler characteristic {chi}, "
        "expected 1): it still wraps around like a tube or has a hole. "
        "Add a seam connecting its boundaries (e.g. one seam up the side "
        "of a pack body)."
    )


def default_pins(vertices: np.ndarray, faces: np.ndarray) -> tuple[int, int]:
    """The two most distant vertices on the longest boundary loop."""
    loops = boundary_loops(faces)
    if not loops:
        raise ValueError(
            "patch is a closed surface; cut it with seams before flattening"
        )
    border = loops[0]
    i, j = farthest_pair(np.asarray(vertices)[border])
    return int(border[i]), int(border[j])


def _signed_uv_areas(uv: np.ndarray, faces: np.ndarray) -> np.ndarray:
    d1 = uv[faces[:, 1]] - uv[faces[:, 0]]
    d2 = uv[faces[:, 2]] - uv[faces[:, 0]]
    return 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])


def flatten(mesh: trimesh.Trimesh, pins: tuple[int, int] | None = None) -> FlattenResult:
    """Flatten a mesh patch and report per-triangle distortion."""
    if pins is None:
        pins = default_pins(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    uv = lscm(mesh.vertices, mesh.faces, pins=pins)
    report = distortion_report(mesh.vertices, mesh.faces, uv)
    return FlattenResult(uv=uv, distortion=report, pins=pins)
