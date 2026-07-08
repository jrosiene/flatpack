"""Automatic seam alignment ticks."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from flatpack.export import layout_panel, write_dxf, write_svg
from flatpack.flatten import flatten
from flatpack.seams import spec_from_dict, split_mesh
from flatpack.synthetic import make_open_tube, make_sphere_patch

N = 21


def flatten_panel(panel):
    return flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))


def split_sphere_two_panels():
    mesh = make_sphere_patch(radius=150.0, half_width=90.0, n=N)
    seam = [i * N + N // 2 for i in range(N)]
    spec = spec_from_dict(
        {
            "seams": [{"name": "c", "path": seam}],
            "panels": [
                {"name": "L", "anchor_face": 0},
                {"name": "R", "anchor_face": len(mesh.faces) - 1},
            ],
        }
    )
    return split_mesh(mesh, spec)


def test_seam_edges_detected_only_on_the_seam():
    panels = split_sphere_two_panels()
    for panel in panels:
        assert panel.seam_edges, "the shared seam should be flagged on each panel"
        # Every flagged edge maps back to the seam column (j == N//2 vertices).
        for a, b in panel.seam_edges:
            oa = int(panel.orig_vertex_index[a])
            ob = int(panel.orig_vertex_index[b])
            assert oa % N == N // 2 and ob % N == N // 2


def test_free_edge_panel_has_no_seam_ticks():
    """A single flat panel with no seams gets no alignment ticks."""
    from flatpack.synthetic import make_plane

    mesh = make_plane(half_width=50.0, n=N)
    spec = spec_from_dict({"seams": []})
    panel = split_mesh(mesh, spec)[0]
    layout = layout_panel(panel, flatten_panel(panel).uv)
    assert layout.seam_ticks == []


def test_mating_panels_get_matching_tick_counts():
    """The two panels sharing a seam get the same number of ticks (so they
    can be matched one-to-one when sewn)."""
    left, right = split_sphere_two_panels()
    ll = layout_panel(left, flatten_panel(left).uv)
    rl = layout_panel(right, flatten_panel(right).uv)
    assert len(ll.seam_ticks) >= 1
    assert len(ll.seam_ticks) == len(rl.seam_ticks)


def test_ticks_align_by_arc_fraction():
    """Corresponding ticks sit at the same fraction along each panel's seam,
    so they meet when the seam is sewn."""
    left, right = split_sphere_two_panels()

    def tick_fractions(panel):
        layout = layout_panel(panel, flatten_panel(panel).uv)
        # Project ticks onto the seam run and report their arc fractions.
        pts = np.array([p for p, _ in layout.seam_ticks])
        # Order along the seam by distance from the run's first tick.
        seam_pts = pts[np.argsort(pts[:, 1])]
        d = np.linalg.norm(np.diff(seam_pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(d)])
        return cum / cum[-1] if cum[-1] > 0 else cum

    fl = tick_fractions(left)
    fr = tick_fractions(right)
    assert len(fl) == len(fr)
    # Same fractional spacing on both panels (mirror seam), within tolerance.
    assert np.allclose(fl, fr, atol=0.05)


def test_tube_seam_gets_ticks_on_both_sides():
    """One seam up a tube leaves both sides on a single panel; both sides
    get matching ticks so the tube can be closed."""
    tube = make_open_tube(radius=80.0, height=300.0, n_around=24, n_height=9)
    spec = spec_from_dict({"seams": [{"name": "side", "path": list(range(9))}]})
    panel = split_mesh(tube, spec)[0]
    layout = layout_panel(panel, flatten_panel(panel).uv)
    # Two seam sides, each with the same interior tick count -> even total.
    assert len(layout.seam_ticks) >= 2
    assert len(layout.seam_ticks) % 2 == 0


def test_markers_toggle_in_svg_and_dxf(tmp_path):
    left, _ = split_sphere_two_panels()
    layout = layout_panel(left, flatten_panel(left).uv)

    on = tmp_path / "on.svg"
    write_svg([layout], str(on), seam_markers=True)
    assert "seam-tick" in on.read_text()

    off = tmp_path / "off.svg"
    write_svg([layout], str(off), seam_markers=False)
    assert "seam-tick" not in off.read_text()

    import ezdxf

    dxf = tmp_path / "p.dxf"
    write_dxf([layout], str(dxf), seam_markers=True)
    layers = {e.dxf.layer for e in ezdxf.readfile(str(dxf)).modelspace()}
    assert "SEAMTICK" in layers


def test_spec_field_default_and_parse():
    assert spec_from_dict({}).seam_markers is True
    assert spec_from_dict({"seam_markers": False}).seam_markers is False
