"""Open-ended containers (tube-like pack bodies) through the seam pipeline."""

import numpy as np
import pytest
import trimesh

from flatpack.flatten import flatten
from flatpack.meshutil import boundary_loops
from flatpack.seams import open_seams, spec_from_dict, split_mesh
from flatpack.synthetic import make_open_tube

N_AROUND, N_HEIGHT = 24, 9
SIDE_SEAM = [j for j in range(N_HEIGHT)]  # straight up ring 0


@pytest.fixture
def tube():
    return make_open_tube(radius=80.0, height=300.0, n_around=N_AROUND, n_height=N_HEIGHT)


def test_tube_without_seam_is_rejected_clearly(tube):
    with pytest.raises(ValueError, match="wraps around"):
        flatten(tube)


def test_side_seam_opens_the_tube(tube):
    """One seam up the side must actually open the tube: the seam's
    interior vertices get duplicated and the seam becomes boundary."""
    spec = spec_from_dict({"seams": [{"name": "side", "path": SIDE_SEAM}]})
    panels = split_mesh(tube, spec)
    assert len(panels) == 1  # a non-separating seam still yields one panel

    panel = panels[0]
    # Interior seam vertices are doubled (endpoints sit on the end rings,
    # where the two sides also separate, so they double too).
    assert len(panel.vertices) == len(tube.vertices) + len(SIDE_SEAM)
    # Both copies map back to the original seam vertices.
    for v in SIDE_SEAM:
        assert np.count_nonzero(panel.orig_vertex_index == v) == 2
    # The opened panel is a single disk-like sheet: one boundary loop.
    assert len(boundary_loops(panel.faces)) == 1


def test_opened_tube_flattens_like_the_cylinder_it_is(tube):
    """A cylinder is developable: after one side seam it must flatten
    nearly perfectly, and its uv should be ~circumference x height."""
    spec = spec_from_dict({"seams": [{"name": "side", "path": SIDE_SEAM}]})
    panel = split_mesh(tube, spec)[0]
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))

    strain = np.maximum(result.distortion.sigma1 - 1, 1 - result.distortion.sigma2)
    assert strain.max() < 0.01, "opened cylinder should flatten near-perfectly"
    assert not result.distortion.flipped.any()

    # The sheet comes out rotated (pins set an arbitrary orientation), so
    # measure its extents along its own principal axes.
    centered = result.uv - result.uv.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    extents = np.ptp(centered @ axes.T, axis=0)
    circumference = 2 * np.pi * 80.0
    assert sorted(extents) == pytest.approx(
        sorted([circumference, 300.0]), rel=0.02
    )


def test_two_seams_split_tube_into_two_sheets(tube):
    """Seams up opposite sides make front and back panels."""
    other_side = [(N_AROUND // 2) * N_HEIGHT + j for j in range(N_HEIGHT)]
    spec = spec_from_dict(
        {
            "seams": [
                {"name": "left", "path": SIDE_SEAM},
                {"name": "right", "path": other_side},
            ]
        }
    )
    panels = split_mesh(tube, spec)
    assert len(panels) == 2
    for panel in panels:
        result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        assert (result.distortion.sigma1 - 1).max() < 0.01


def test_open_seams_noop_without_seams(tube):
    vertices, faces, orig_map = open_seams(tube, set())
    assert len(vertices) == len(tube.vertices)
    assert np.array_equal(faces, np.asarray(tube.faces))
    assert np.array_equal(orig_map, np.arange(len(vertices)))


def test_partial_seam_does_not_tear_extra_vertices():
    """A seam ending mid-surface duplicates only strictly interior seam
    vertices; the innermost endpoint stays single (the cut just stops)."""
    from flatpack.synthetic import make_plane

    n = 11
    mesh = make_plane(half_width=50.0, n=n)
    # Cut inward from the boundary, stopping at the middle of the sheet.
    path = [5 * n + col for col in range(0, n // 2 + 1)]
    spec = spec_from_dict({"seams": [{"name": "slit", "path": path}]})
    panels = split_mesh(mesh, spec)
    assert len(panels) == 1
    panel = panels[0]
    # All path vertices except the interior endpoint are duplicated.
    assert len(panel.vertices) == n * n + len(path) - 1
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    assert np.allclose(result.distortion.sigma1, 1.0, atol=1e-8)
