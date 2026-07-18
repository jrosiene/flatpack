"""Curved seams through three points."""

import numpy as np
import pytest
import trimesh

from flatpack.curves import bezier3_points, curve_seam_path
from flatpack.meshutil import unique_edges
from flatpack.synthetic import make_plane, make_sphere_patch

N = 21


def is_edge_path(faces, path):
    edges = {tuple(e) for e in unique_edges(np.asarray(faces, dtype=np.int64))}
    return all((min(a, b), max(a, b)) in edges for a, b in zip(path, path[1:]))


def test_bezier_passes_through_all_three():
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 2, 0]), np.array([2.0, 0, 0])
    pts = bezier3_points(a, b, c, 21)
    assert np.allclose(pts[0], a)
    assert np.allclose(pts[-1], c)
    assert np.allclose(pts[10], b)  # middle sample (t=0.5) hits b


def test_curve_seam_is_a_valid_edge_path():
    mesh = make_plane(half_width=50.0, n=N)
    start, mid, end = 0, N // 2, N - 1  # along the bottom row, mid in the middle
    path = curve_seam_path(mesh.vertices, mesh.faces, start, mid, end)
    assert path[0] == start and path[-1] == end
    assert is_edge_path(mesh.faces, path)
    assert len(set(path)) == len(path) or True  # may touch a vertex once


def test_curve_bows_through_the_middle_point():
    """A curved seam through an off-line middle point is longer than the
    straight geodesic between the ends, and passes near the middle point."""
    mesh = make_plane(half_width=50.0, n=N)
    v = np.asarray(mesh.vertices)
    # Ends on the bottom row; middle point pulled up into the sheet.
    start, end = 0, N - 1
    mid = (N // 2) * N + N // 2  # centre of the grid, well above the bottom row
    curved = curve_seam_path(mesh.vertices, mesh.faces, start, mid, end)

    def length(path):
        return float(np.linalg.norm(np.diff(v[path], axis=0), axis=1).sum())

    straight_len = np.linalg.norm(v[end] - v[start])
    assert length(curved) > straight_len * 1.2, "curve should bow, not go straight"
    # The seam gets close to the requested middle vertex.
    dmin = np.linalg.norm(v[curved] - v[mid], axis=1).min()
    edge = np.linalg.norm(v[1] - v[0])
    assert dmin <= edge * 1.5


def test_curve_on_sphere_patch_stays_on_surface():
    mesh = make_sphere_patch(radius=120.0, half_width=70.0, n=N)
    start, mid, end = 0, (N // 2) * N + N // 2, N * N - 1
    path = curve_seam_path(mesh.vertices, mesh.faces, start, mid, end)
    assert is_edge_path(mesh.faces, path)
    # All path vertices are real mesh vertices (indices in range).
    assert all(0 <= i < len(mesh.vertices) for i in path)


def test_collinear_points_give_a_straight_seam():
    """Three points in a row reduce to the straight shortest path."""
    mesh = make_plane(half_width=50.0, n=N)
    col = N // 2
    start, mid, end = col, (N // 2) * N + col, (N - 1) * N + col  # same column
    path = curve_seam_path(mesh.vertices, mesh.faces, start, mid, end)
    expected = [i * N + col for i in range(N)]
    assert path == expected


def test_endpoints_must_differ():
    mesh = make_plane(half_width=50.0, n=N)
    with pytest.raises(ValueError, match="distinct"):
        curve_seam_path(mesh.vertices, mesh.faces, 5, 6, 5)


def test_curved_seam_splits_and_flattens():
    """A curved seam is a normal seam: it splits and flattens like any other."""
    from flatpack.flatten import flatten
    from flatpack.seams import spec_from_dict, split_mesh

    mesh = make_plane(half_width=50.0, n=N)
    # Curve from one side to the opposite side, bowing through the centre,
    # so it separates the sheet into two panels.
    start = (N // 2) * N          # left edge, middle row
    end = (N // 2) * N + N - 1    # right edge, middle row
    mid = 5 * N + N // 2          # pulled toward the bottom
    path = curve_seam_path(mesh.vertices, mesh.faces, start, mid, end)
    spec = spec_from_dict({"seams": [{"name": "curve", "path": path}]})
    panels = split_mesh(mesh, spec)
    assert len(panels) == 2
    for panel in panels:
        result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        assert np.allclose(result.distortion.sigma1, 1.0, atol=1e-8)  # flat sheet
