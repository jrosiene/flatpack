"""Darts (slit seams that absorb distortion) and sewing marks."""

import xml.etree.ElementTree as ET

import ezdxf
import numpy as np
import pytest
import trimesh

from flatpack.export import layout_panel, write_dxf, write_svg
from flatpack.flatten import flatten
from flatpack.seams import spec_from_dict, split_mesh
from flatpack.synthetic import make_sphere_patch

N = 21
# Dart: from the middle of one boundary edge straight in to the centre.
DART = [i * N + N // 2 for i in range(N // 2 + 1)]


def sphere():
    return make_sphere_patch(radius=100.0, half_width=65.0, n=N)


def split_with(data):
    return split_mesh(sphere(), spec_from_dict(data))


def test_spec_parses_darts_and_marks():
    spec = spec_from_dict(
        {
            "darts": [{"name": "d", "path": DART}],
            "marks": [
                {"vertex": 7, "type": "bartack", "label": "strap", "toward": 8},
                {"vertex": 12},
            ],
        }
    )
    assert spec.darts == [DART]
    assert spec.marks[0].type == "bartack" and spec.marks[0].toward == 8
    assert spec.marks[1].type == "attach" and spec.marks[1].toward is None


def test_dart_is_a_slit_not_a_split():
    panels = split_with({"darts": [{"name": "d", "path": DART}]})
    assert len(panels) == 1
    panel = panels[0]
    # Every dart vertex except the interior apex is duplicated.
    assert len(panel.vertices) == N * N + len(DART) - 1
    assert panel.darts == [DART]


def test_dart_reduces_distortion():
    """The whole point: a dart lets doubly-curved fabric lie flatter."""
    plain = flatten(sphere())
    panel = split_with({"darts": [{"name": "d", "path": DART}]})[0]
    darted = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))

    worst = lambda r: float(
        np.maximum(r.distortion.sigma1 - 1, 1 - r.distortion.sigma2).max()
    )
    assert worst(darted) < worst(plain) * 0.75, (
        f"dart should cut worst strain substantially: "
        f"{worst(plain):.4f} -> {worst(darted):.4f}"
    )


def test_dart_layout_geometry():
    """Legs share the apex, and the mouth opens by the dart intake."""
    panel = split_with({"darts": [{"name": "d", "path": DART}]})[0]
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    layout = layout_panel(panel, result.uv)

    assert len(layout.darts) == 1
    leg1, leg2 = layout.darts[0]
    assert len(leg1) == len(leg2) == len(DART)
    assert np.allclose(leg1[0], leg2[0])  # apex first, shared
    intake = np.linalg.norm(leg1[-1] - leg2[-1])
    assert intake > 1.0, "flattening should open the slit into a V"
    # Legs are (almost) mirror images: equal length.
    length = lambda leg: np.linalg.norm(np.diff(leg, axis=0), axis=1).sum()
    assert length(leg1) == pytest.approx(length(leg2), rel=0.05)


def test_interior_fisheye_dart_is_rejected_clearly():
    """A fully interior slit makes an annulus, which cannot conformally
    flatten - the error should say so up front."""
    interior = [i * N + N // 2 for i in range(4, N - 4)]  # both ends interior
    with pytest.raises(ValueError, match="boundary or seam"):
        split_with({"darts": [{"name": "fish", "path": interior}]})


def test_dart_reversed_path_is_accepted():
    """Mouth may be given as either end of the path."""
    panels = split_with({"darts": [{"name": "d", "path": DART[::-1]}]})
    assert len(panels) == 1


def test_marks_land_on_panels_and_files(tmp_path):
    center = (N // 2) * N + N // 2
    panels = split_with(
        {
            "marks": [
                {"vertex": center, "type": "bartack", "label": "strap", "toward": center + 1},
                {"vertex": center + N, "type": "attach", "label": "buckle"},
            ],
        }
    )
    panel = panels[0]
    assert len(panel.marks) == 2

    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    layout = layout_panel(panel, result.uv)
    assert len(layout.marks) == 2
    assert np.linalg.norm(layout.marks[0].direction) == pytest.approx(1.0)

    svg = tmp_path / "p.svg"
    write_svg([layout], str(svg))
    body = ET.tostring(ET.parse(svg).getroot(), encoding="unicode")
    assert "mark-bartack" in body and "mark-attach" in body
    assert "strap" in body and "buckle" in body

    dxf = tmp_path / "p.dxf"
    write_dxf([layout], str(dxf))
    layers = {e.dxf.layer for e in ezdxf.readfile(str(dxf)).modelspace()}
    assert "MARK" in layers


def test_dart_in_dxf(tmp_path):
    panel = split_with({"darts": [{"name": "d", "path": DART}]})[0]
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    layout = layout_panel(panel, result.uv)
    dxf = tmp_path / "p.dxf"
    write_dxf([layout], str(dxf))
    msp = ezdxf.readfile(str(dxf)).modelspace()
    dart_lines = [e for e in msp if e.dxf.layer == "DART"]
    assert len(dart_lines) == 3  # two legs + apex circle
