"""Page tiling for home printing."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

from flatpack.export import layout_panel, sheet_bbox
from flatpack.flatten import flatten
from flatpack.seams import Panel, PanelSpec
from flatpack.synthetic import make_plane
from flatpack.tiling import PAGE_SIZES_MM, page_windows, write_tiled_svgs


def make_layout(half_width):
    """A simple square panel of the requested size (mm half-width)."""
    mesh = make_plane(half_width=half_width, n=5)
    panel = Panel(
        name="square",
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        orig_vertex_index=np.arange(len(mesh.vertices)),
        spec=PanelSpec(name="square"),
    )
    result = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    return layout_panel(panel, result.uv)


def test_small_panel_fits_one_page():
    layout = make_layout(half_width=40.0)  # ~110 mm wide with allowance
    windows = page_windows(sheet_bbox([layout]), page="letter")
    assert len(windows) == 1
    assert windows[0].label == "A1"


def test_large_panel_needs_a_grid_of_pages():
    layout = make_layout(half_width=200.0)  # ~430 mm square: > 2 pages each way
    bbox = sheet_bbox([layout])
    windows = page_windows(bbox, page="a4", printer_margin=10.0, overlap=15.0)
    labels = {w.label for w in windows}
    assert len(windows) >= 4
    assert "A1" in labels and "B2" in labels

    # Adjacent columns overlap by exactly the glue strip.
    a1 = next(w for w in windows if w.label == "A1")
    b1 = next(w for w in windows if w.label == "B1")
    assert (a1.x0 + a1.width) - b1.x0 == pytest.approx(15.0)

    # Together the windows cover the whole sheet.
    assert min(w.x0 for w in windows) <= bbox[0]
    assert max(w.x0 + w.width for w in windows) >= bbox[2]
    assert min(w.y0 for w in windows) <= bbox[1]
    assert max(w.y0 + w.height for w in windows) >= bbox[3]


def test_pages_are_printable_size_and_marked(tmp_path):
    layout = make_layout(half_width=150.0)
    paths = write_tiled_svgs([layout], tmp_path, page="letter", printer_margin=10.0)
    assert len(paths) > 1

    for path in paths:
        root = ET.parse(path).getroot()
        width_mm = float(root.get("width").removesuffix("mm"))
        assert width_mm == pytest.approx(PAGE_SIZES_MM["letter"][0] - 20.0)
        body = ET.tostring(root, encoding="unicode")
        assert "page-marks" in body
        assert "page " in body  # label text


def test_overlap_strip_repeats_on_next_page(tmp_path):
    """The same pattern geometry near a page seam appears on both pages."""
    layout = make_layout(half_width=150.0)
    paths = write_tiled_svgs([layout], tmp_path, page="letter")
    bodies = {p.name: ET.tostring(ET.parse(p).getroot(), encoding="unicode") for p in paths}
    # Every page embeds the full pattern group (clipped by the viewport),
    # so the panel outline is present in each file.
    for body in bodies.values():
        assert "panel-square" in body
