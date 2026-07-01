"""SVG/DXF export: files parse, geometry is sane."""

import xml.etree.ElementTree as ET

import ezdxf
import numpy as np
import pytest
import trimesh
import yaml
from shapely.geometry import Point, Polygon

from flatpack.export import layout_panel, pack_layouts, write_dxf, write_svg
from flatpack.flatten import flatten
from flatpack.seams import load_seam_spec, split_mesh
from flatpack.synthetic import make_sphere_patch


@pytest.fixture(scope="module")
def layout_and_panel():
    n = 15
    mesh = make_sphere_patch(radius=150.0, half_width=80.0, n=n)
    center = [i * n + n // 2 for i in range(n)]
    spec = load_seam_spec_from_dict(
        {
            "seams": [{"name": "cut", "path": center}],
            "panels": [
                {
                    "name": "left",
                    "anchor_face": 0,
                    "fabric": "silnylon",
                    "grain": [0, (n - 1) * n],  # panel-side edge, along i
                    "notches": [center[n // 2]],
                }
            ],
            "seam_allowance": 10,
        }
    )
    panels = split_mesh(mesh, spec)
    panel = next(p for p in panels if p.name == "left")
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    layout = layout_panel(panel, result.uv, seam_allowance=10.0)
    return layout, panel


def load_seam_spec_from_dict(data, tmp_path=None):
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False
    ) as handle:
        yaml.safe_dump(data, handle)
        name = handle.name
    return load_seam_spec(Path(name))


def test_cut_line_offsets_outward(layout_and_panel):
    layout, _ = layout_and_panel
    stitch = Polygon(layout.stitch)
    cut = Polygon(layout.cut)
    assert cut.area > stitch.area
    assert cut.contains(stitch)
    # Offset distance is respected (mitred corners can exceed it slightly).
    assert cut.exterior.distance(stitch) == pytest.approx(10.0, abs=0.5)


def test_notches_sit_on_stitch_line(layout_and_panel):
    layout, _ = layout_and_panel
    assert len(layout.notches) == 1
    point, outward = layout.notches[0]
    assert Polygon(layout.stitch).exterior.distance(Point(point)) < 1e-6
    assert np.linalg.norm(outward) == pytest.approx(1.0)
    # Outward means: stepping along it leaves the stitch polygon.
    probe = point + outward * 1.0
    assert not Polygon(layout.stitch).contains(Point(probe))


def test_layout_is_in_positive_quadrant(layout_and_panel):
    layout, _ = layout_and_panel
    assert layout.cut.min() >= 0


def test_svg_writes_and_parses(layout_and_panel, tmp_path):
    layout, _ = layout_and_panel
    pack_layouts([layout])
    path = tmp_path / "pattern.svg"
    write_svg([layout], str(path))

    tree = ET.parse(path)
    root = tree.getroot()
    assert root.tag.endswith("svg")
    assert root.get("width").endswith("mm")
    body = ET.tostring(root, encoding="unicode")
    assert "panel-left" in body
    assert "silnylon" in body


def test_dxf_round_trips(layout_and_panel, tmp_path):
    layout, _ = layout_and_panel
    path = tmp_path / "pattern.dxf"
    write_dxf([layout], str(path))

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()
    layers = {e.dxf.layer for e in msp}
    assert {"CUT", "STITCH", "NOTCH", "ANNOT"} <= layers
    cut_polys = [e for e in msp if e.dxf.layer == "CUT"]
    assert len(cut_polys) == 1
    assert cut_polys[0].closed


def test_pack_layouts_separates_panels():
    n = 15
    mesh = make_sphere_patch(radius=150.0, half_width=80.0, n=n)
    center = [i * n + n // 2 for i in range(n)]
    spec = load_seam_spec_from_dict(
        {"seams": [{"name": "cut", "path": center}], "seam_allowance": 10}
    )
    panels = split_mesh(mesh, spec)
    layouts = []
    for panel in panels:
        result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        layouts.append(layout_panel(panel, result.uv))
    pack_layouts(layouts)
    a, b = (Polygon(layout.cut) for layout in layouts)
    assert not a.intersects(b)
