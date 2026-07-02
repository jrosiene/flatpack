"""Plane cutting across faces (diagonal seams)."""

import numpy as np
import pytest
import trimesh

from flatpack.cut import cut_between
from flatpack.meshutil import unique_edges
from flatpack.synthetic import make_plane, make_sphere_patch

N = 11


def is_edge_path(faces, path):
    edges = {tuple(e) for e in unique_edges(np.asarray(faces, dtype=np.int64))}
    return all(
        (min(a, b), max(a, b)) in edges for a, b in zip(path, path[1:])
    )


def test_diagonal_cut_on_flat_grid_is_straight():
    """Cut corner-to-corner across a flat grid: the path must be the
    straight diagonal, not a staircase along existing edges.

    The grid's triangulation runs its diagonals the other way, so this
    cut genuinely crosses faces.
    """
    mesh = make_plane(half_width=50.0, n=N)
    start, end = N - 1, (N - 1) * N  # anti-diagonal corners
    result = cut_between(mesh, start, end)

    a = mesh.vertices[start]
    b = mesh.vertices[end]
    direction = (b - a) / np.linalg.norm(b - a)
    for v in result.path:
        p = result.mesh.vertices[v] - a
        off_line = p - (p @ direction) * direction
        assert np.linalg.norm(off_line) < 1e-9, "cut point strays off the diagonal"

    assert result.path[0] == start and result.path[-1] == end
    assert is_edge_path(result.mesh.faces, result.path)
    assert len(result.mesh.vertices) > len(mesh.vertices)  # actually cut faces


def test_cut_preserves_surface_area_and_quality():
    mesh = make_sphere_patch(radius=100.0, half_width=60.0, n=N)
    result = cut_between(mesh, 0, N * N - 1)
    assert result.mesh.area == pytest.approx(mesh.area, rel=1e-9)
    assert result.mesh.area_faces.min() > 1e-9, "cut created degenerate triangles"
    # No T-junctions: every interior edge is shared by exactly two faces.
    edges = np.sort(
        np.concatenate(
            [result.mesh.faces[:, [0, 1]], result.mesh.faces[:, [1, 2]], result.mesh.faces[:, [2, 0]]]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() <= 2


def test_cut_keeps_existing_vertices_stable():
    mesh = make_plane(half_width=50.0, n=N)
    result = cut_between(mesh, N - 1, (N - 1) * N)
    assert len(result.mesh.vertices) > len(mesh.vertices)
    assert np.allclose(
        result.mesh.vertices[: len(mesh.vertices)], mesh.vertices
    ), "cutting must only append vertices, never reorder them"


def test_cut_along_existing_edges_adds_no_vertices():
    """Cutting straight along a grid column needs no new geometry."""
    mesh = make_plane(half_width=50.0, n=N)
    start, end = 5, (N - 1) * N + 5  # same column, opposite sides
    result = cut_between(mesh, start, end)
    assert len(result.mesh.vertices) == len(mesh.vertices)
    assert result.path[0] == start and result.path[-1] == end
    assert is_edge_path(result.mesh.faces, result.path)


def test_snapping_avoids_slivers():
    """A cut passing very close to a vertex snaps to it instead of
    creating a needle triangle."""
    mesh = make_plane(half_width=50.0, n=N)
    # Nudge one interior vertex to sit almost exactly on the diagonal.
    v = mesh.vertices.copy()
    row, col = 3, 3
    v[row * N + col, :2] = v[row * N + col, :2] + 0.01
    mesh = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False)
    result = cut_between(mesh, 0, N * N - 1, snap=0.05)
    ratio = result.mesh.area_faces.min() / mesh.area_faces.mean()
    assert ratio > 1e-4, "sliver triangle produced despite snapping"


def test_diagonal_seam_flattens_cleanly():
    """End to end: diagonal cut -> seam -> split -> flatten. On a flat
    sheet the two triangular panels must flatten with zero distortion."""
    from flatpack.flatten import flatten
    from flatpack.seams import spec_from_dict, split_mesh

    mesh = make_plane(half_width=50.0, n=N)
    result = cut_between(mesh, N - 1, (N - 1) * N)

    spec = spec_from_dict({"seams": [{"name": "diag", "path": result.path}]})
    panels = split_mesh(result.mesh, spec)
    assert len(panels) == 2
    for panel in panels:
        flat = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        assert np.allclose(flat.distortion.sigma1, 1.0, atol=1e-8)


def test_degenerate_requests_are_rejected():
    mesh = make_plane(half_width=50.0, n=N)
    with pytest.raises(ValueError, match="two different"):
        cut_between(mesh, 3, 3)
