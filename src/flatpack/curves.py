"""Curved seams: a smooth seam through three clicked points.

A straight seam between two clicks follows the shortest surface path (a
geodesic). A curved seam is defined by three points — start, a middle
point it bows through, and end — fitted with a quadratic Bezier that
passes through all three. The curve is sampled, each sample snapped to the
nearest mesh vertex, and consecutive snaps stitched together with shortest
edge paths, so the result is an ordinary seam (a vertex path along mesh
edges) that curves through the middle point instead of cornering at it.

Collinear points give a straight line, so this never fails degenerately.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
from scipy.spatial import cKDTree

from flatpack.meshutil import unique_edges


def bezier3_points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, samples: int
) -> np.ndarray:
    """Sample a quadratic Bezier that passes through a, b, c (b at t=0.5)."""
    control = 2.0 * b - 0.5 * (a + c)  # so the curve hits b exactly at t=0.5
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * control + t**2 * c


def edge_length_graph(
    vertices: np.ndarray, faces: np.ndarray
) -> scipy.sparse.csr_matrix:
    """Vertex adjacency weighted by edge length (for shortest paths)."""
    edges = unique_edges(np.asarray(faces, dtype=np.int64))
    v = np.asarray(vertices, dtype=float)
    lengths = np.linalg.norm(v[edges[:, 1]] - v[edges[:, 0]], axis=1)
    n = len(v)
    return scipy.sparse.coo_matrix(
        (lengths, (edges[:, 0], edges[:, 1])), shape=(n, n)
    ).tocsr()


def curve_seam_path(
    vertices: np.ndarray,
    faces: np.ndarray,
    start: int,
    mid: int,
    end: int,
    samples: int = 24,
    edge_graph: scipy.sparse.csr_matrix | None = None,
) -> list[int]:
    """Vertex path of a curved seam through (start, mid, end).

    edge_graph may be passed in (cached) to avoid rebuilding it.
    """
    vertices = np.asarray(vertices, dtype=float)
    n = len(vertices)
    for name, idx in (("start", start), ("mid", mid), ("end", end)):
        if not 0 <= idx < n:
            raise ValueError(f"{name} vertex {idx} out of range (mesh has {n})")
    if start == end:
        raise ValueError("a curved seam needs distinct start and end points")

    if edge_graph is None:
        edge_graph = edge_length_graph(vertices, faces)
    tree = cKDTree(vertices)

    curve = bezier3_points(vertices[start], vertices[mid], vertices[end], samples)
    _, snapped = tree.query(curve)

    # Waypoints: the snapped vertices, de-duplicated, endpoints pinned exactly.
    waypoints: list[int] = []
    for v in snapped:
        v = int(v)
        if not waypoints or waypoints[-1] != v:
            waypoints.append(v)
    if waypoints[0] != start:
        waypoints.insert(0, start)
    if waypoints[-1] != end:
        waypoints.append(end)

    # Stitch consecutive waypoints with shortest edge paths so the result is
    # a continuous edge path (each pair is close, so it hugs the curve).
    path = [waypoints[0]]
    for u, w in zip(waypoints, waypoints[1:]):
        if w == u:
            continue
        leg = _shortest_path(edge_graph, u, w)
        for v in leg[1:]:
            # Drop an immediate backtrack (…x, y, x…) to keep the seam clean.
            if len(path) >= 2 and path[-2] == v:
                path.pop()
            else:
                path.append(v)
    return path


def _shortest_path(
    edge_graph: scipy.sparse.csr_matrix, source: int, target: int
) -> list[int]:
    _, predecessors = scipy.sparse.csgraph.dijkstra(
        edge_graph, directed=False, indices=source, return_predecessors=True
    )
    if predecessors[target] < 0 and source != target:
        raise ValueError("curve crosses a gap in the mesh; pick closer points")
    path = [target]
    while path[-1] != source:
        path.append(int(predecessors[path[-1]]))
    return path[::-1]
