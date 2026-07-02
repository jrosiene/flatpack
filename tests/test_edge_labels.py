"""Edge-length labels along the straight edges of a pattern."""

import xml.etree.ElementTree as ET

import ezdxf
import numpy as np
import pytest
import trimesh

from flatpack.export import format_edge_length, layout_panel, write_dxf, write_svg
from flatpack.flatten import flatten
from flatpack.seams import spec_from_dict, split_mesh
from flatpack.synthetic import make_open_tube

NA, NH = 24, 9


@pytest.fixture(scope="module")
def rectangle_layout():
    """Unrolled tube: a rectangle ~503 x 300 mm with four straight edges."""
    tube = make_open_tube(radius=80.0, height=300.0, n_around=NA, n_height=NH)
    spec = spec_from_dict({"seams": [{"name": "side", "path": list(range(NH))}]})
    panel = split_mesh(tube, spec)[0]
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    return layout_panel(panel, result.uv)


def test_formatting():
    assert format_edge_length(503.0, "cm") == "50.3 cm"
    assert format_edge_length(254.0, "in") == '10.00"'
    with pytest.raises(ValueError):
        format_edge_length(100.0, "furlongs")


def test_rectangle_gets_four_edge_labels(rectangle_layout):
    labels = rectangle_layout.edge_labels
    assert len(labels) == 4
    lengths = sorted(label.length_mm for label in labels)
    circumference = 2 * np.pi * 80.0
    # Two short edges (height) and two long ones (circumference); the
    # unrolled polygon edge is a chord-polygon of the circle, slightly
    # shorter than the true circumference.
    assert lengths[0] == pytest.approx(300.0, rel=0.01)
    assert lengths[1] == pytest.approx(300.0, rel=0.01)
    assert lengths[2] == pytest.approx(circumference, rel=0.01)
    assert lengths[3] == pytest.approx(circumference, rel=0.01)
    # Labels sit inside the stitch outline.
    from shapely.geometry import Point, Polygon

    poly = Polygon(rectangle_layout.stitch)
    for label in labels:
        assert poly.contains(Point(label.pos))


def test_svg_labels_toggle(rectangle_layout, tmp_path):
    off = tmp_path / "off.svg"
    write_svg([rectangle_layout], str(off))
    assert "edge-length" not in off.read_text()

    cm = tmp_path / "cm.svg"
    write_svg([rectangle_layout], str(cm), edge_units="cm")
    body = cm.read_text()
    assert body.count('class="edge-length"') == 4
    assert "30.0 cm" in body

    inches = tmp_path / "in.svg"
    write_svg([rectangle_layout], str(inches), edge_units="in")
    assert '11.81"' in inches.read_text()  # 300 mm


def test_dxf_labels_toggle(rectangle_layout, tmp_path):
    path = tmp_path / "p.dxf"
    write_dxf([rectangle_layout], str(path), edge_units="cm")
    msp = ezdxf.readfile(str(path)).modelspace()
    texts = [e.dxf.text for e in msp if e.dxftype() == "TEXT"]
    assert sum("cm" in t for t in texts) == 4


def test_spec_edge_labels_field():
    assert spec_from_dict({}).edge_labels == "none"
    assert spec_from_dict({"edge_labels": "in"}).edge_labels == "in"
    with pytest.raises(ValueError, match="edge_labels"):
        spec_from_dict({"edge_labels": "yards"})
