"""Small mesh helpers shared by the flattening and export steps."""

from __future__ import annotations

import numpy as np


def triangle_frames(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Express each triangle in a local orthonormal 2D frame.

    The frame puts corner 0 at the origin and corner 1 on the +x axis, so
    the three corners become (0, 0), (x1, 0) and (x2, y2) with y2 > 0.

    Returns (x1, x2, y2, area), each an array of length len(faces).
    """
    p0 = vertices[faces[:, 0]]
    p1 = vertices[faces[:, 1]]
    p2 = vertices[faces[:, 2]]

    e1 = p1 - p0
    x1 = np.linalg.norm(e1, axis=1)
    if np.any(x1 < 1e-12):
        raise ValueError("mesh contains a degenerate triangle (zero-length edge)")
    u = e1 / x1[:, None]

    d2 = p2 - p0
    normal = np.cross(e1, d2)
    norm = np.linalg.norm(normal, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("mesh contains a degenerate (zero-area) triangle")
    v = np.cross(normal / norm[:, None], u)

    x2 = np.einsum("ij,ij->i", d2, u)
    y2 = np.einsum("ij,ij->i", d2, v)
    area = 0.5 * x1 * y2
    return x1, x2, y2, area


def boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """Ordered boundary vertex loops of a triangle mesh.

    Boundary edges are the ones used by exactly one face. They are chained
    in face-winding order into closed loops; loops are returned longest
    first. A closed (watertight) mesh returns an empty list.
    """
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    boundary = directed[counts[inverse] == 1]

    successor = {int(a): int(b) for a, b in boundary}
    loops: list[np.ndarray] = []
    while successor:
        start, nxt = successor.popitem()
        loop = [start]
        while nxt != start:
            loop.append(nxt)
            nxt = successor.pop(nxt)
        loops.append(np.array(loop))
    loops.sort(key=len, reverse=True)
    return loops


def farthest_pair(points: np.ndarray, max_samples: int = 1500) -> tuple[int, int]:
    """Indices of (approximately) the two most distant points.

    Exact for up to max_samples points; beyond that a random subsample is
    used, which is plenty for picking LSCM pin vertices.
    """
    n = len(points)
    if n < 2:
        raise ValueError("need at least two points")
    if n > max_samples:
        rng = np.random.default_rng(0)
        candidates = rng.choice(n, size=max_samples, replace=False)
    else:
        candidates = np.arange(n)
    sub = points[candidates]
    d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return int(candidates[i]), int(candidates[j])


def unique_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges of the mesh, shape (m, 2), each sorted."""
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(directed, axis=1), axis=0)
