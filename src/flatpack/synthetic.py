"""Synthetic test surfaces with known curvature.

These are used to validate the flattening step before touching real
backpack geometry:

- a flat plane (zero curvature: flattening must be exact),
- a cylinder patch (developable: flattens with ~zero distortion),
- a sphere patch (positive Gaussian curvature: cannot flatten without
  distortion, like the domed back panel of a pack),
- a saddle patch (negative Gaussian curvature, like a strap junction).

All are height fields z = f(x, y) over a regular grid on a square,
which keeps the triangulation simple and free of degenerate triangles.
"""

from __future__ import annotations

import numpy as np
import trimesh


def _grid_mesh(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> trimesh.Trimesh:
    """Triangulate a height field given by 2D arrays x, y, z of shape (n, m)."""
    n, m = z.shape
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    idx = np.arange(n * m).reshape(n, m)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[:-1, 1:].ravel()
    d = idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, d]), np.column_stack([a, d, c])])

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _square_grid(half_width: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    s = np.linspace(-half_width, half_width, n)
    return np.meshgrid(s, s, indexing="ij")


def make_plane(half_width: float = 50.0, n: int = 15) -> trimesh.Trimesh:
    """Flat square patch. Flattening must reproduce it exactly."""
    x, y = _square_grid(half_width, n)
    return _grid_mesh(x, y, np.zeros_like(x))


def make_cylinder_patch(
    radius: float = 100.0, half_width: float = 60.0, n: int = 21
) -> trimesh.Trimesh:
    """Developable patch: cylinder of given radius, curved along x only.

    half_width must be < radius.
    """
    if half_width >= radius:
        raise ValueError("half_width must be smaller than radius")
    x, y = _square_grid(half_width, n)
    z = np.sqrt(radius**2 - x**2)
    return _grid_mesh(x, y, z)


def make_sphere_patch(
    radius: float = 100.0, half_width: float = 60.0, n: int = 21
) -> trimesh.Trimesh:
    """Doubly-curved patch cut from a sphere (positive Gaussian curvature).

    half_width must be < radius / sqrt(2) so the patch stays on the sphere.
    """
    if half_width >= radius / np.sqrt(2):
        raise ValueError("half_width must be smaller than radius / sqrt(2)")
    x, y = _square_grid(half_width, n)
    z = np.sqrt(radius**2 - x**2 - y**2)
    return _grid_mesh(x, y, z)


def make_saddle_patch(
    half_width: float = 60.0, curvature: float = 0.005, n: int = 21
) -> trimesh.Trimesh:
    """Saddle z = curvature * (x^2 - y^2): negative Gaussian curvature."""
    x, y = _square_grid(half_width, n)
    z = curvature * (x**2 - y**2)
    return _grid_mesh(x, y, z)
