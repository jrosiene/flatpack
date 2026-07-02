"""Turn flattened panels into printable pattern geometry and SVG/DXF files.

For each panel we build a PanelLayout in final 2D coordinates (mm):

- the stitch line (the flattened boundary itself),
- the cut line (stitch line offset outward by the seam allowance),
- notches (ticks crossing the seam allowance at marked vertices),
- a grainline arrow and a text label.

Panels are rotated grain-vertical, then arranged left-to-right on one
sheet. SVG is written with millimetre units so it prints at true scale;
DXF uses layers CUT / STITCH / NOTCH / ANNOT.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import ezdxf
import numpy as np
from shapely.geometry import Polygon

from flatpack.meshutil import boundary_loops
from flatpack.seams import Panel

SVG_NS = "http://www.w3.org/2000/svg"


@dataclass
class PanelLayout:
    """One panel's printable geometry, in sheet coordinates (mm, y up)."""

    name: str
    stitch: np.ndarray  # (n, 2) closed boundary (stitch line)
    cut: np.ndarray  # (m, 2) closed boundary (cut line, offset outward)
    notches: list[tuple[np.ndarray, np.ndarray]]  # (point on stitch, unit outward dir)
    grain_arrow: tuple[np.ndarray, np.ndarray]  # (start, end) points
    label: str
    label_pos: np.ndarray
    seam_allowance: float


def layout_panel(
    panel: Panel,
    uv: np.ndarray,
    seam_allowance: float = 10.0,
) -> PanelLayout:
    """Build printable geometry for one flattened panel.

    Rotates the layout so the grainline (panel.spec.grain vertex pair, if
    given) points straight up, and translates it to the positive quadrant.
    """
    faces = np.asarray(panel.faces, dtype=np.int64)
    loop = boundary_loops(faces)[0]
    stitch = uv[loop].astype(float)

    stitch = _rotate_grain_vertical(panel, uv, stitch)

    polygon = Polygon(stitch)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    cut_polygon = polygon.buffer(
        seam_allowance, join_style="mitre", mitre_limit=4.0
    )
    cut = np.asarray(cut_polygon.exterior.coords)[:-1]

    # Shift everything to the positive quadrant with a small margin.
    shift = seam_allowance + 5.0 - np.min(cut, axis=0)
    stitch = stitch + shift
    cut = cut + shift
    polygon = Polygon(stitch)

    notches = _notches(panel, uv, stitch, loop, shift)

    centroid = np.asarray(polygon.centroid.coords[0])
    height = float(np.ptp(stitch[:, 1]))
    grain_len = min(60.0, 0.5 * height)
    grain_arrow = (
        centroid + np.array([0.0, -grain_len / 2]),
        centroid + np.array([0.0, grain_len / 2]),
    )

    label = f"{panel.name}  ({panel.spec.fabric})"
    return PanelLayout(
        name=panel.name,
        stitch=stitch,
        cut=cut,
        notches=notches,
        grain_arrow=grain_arrow,
        label=label,
        label_pos=centroid + np.array([5.0, 0.0]),
        seam_allowance=seam_allowance,
    )


def _rotate_grain_vertical(
    panel: Panel, uv: np.ndarray, stitch: np.ndarray
) -> np.ndarray:
    """Rotate the boundary so the panel's grainline points +y.

    Without a grainline, the flattening's orientation is arbitrary (set by
    the LSCM pins), so fall back to the boundary's principal axis: the
    panel comes out upright instead of tilted, which also tiles onto fewer
    pages.
    """
    if panel.spec.grain is not None:
        a = uv[panel.local_index(panel.spec.grain[0])]
        b = uv[panel.local_index(panel.spec.grain[1])]
        direction = b - a
    else:
        centered = stitch - stitch.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
    angle = np.arctan2(direction[1], direction[0])
    rot = np.pi / 2 - angle
    c, s = np.cos(rot), np.sin(rot)
    matrix = np.array([[c, -s], [s, c]])
    return stitch @ matrix.T


def _loop_is_ccw(loop: np.ndarray) -> bool:
    x, y = loop[:, 0], loop[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def _notches(
    panel: Panel,
    uv: np.ndarray,
    stitch: np.ndarray,
    loop: np.ndarray,
    shift: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Notch ticks at the panel's marked boundary vertices.

    Each notch is (point on the stitch line, unit direction pointing
    outward across the seam allowance).
    """
    if not _loop_is_ccw(stitch):
        stitch = stitch[::-1]
        loop = loop[::-1]

    loop_list = list(loop)
    notches = []
    for orig_vertex in panel.spec.notches:
        try:
            local = panel.local_index(orig_vertex)
        except KeyError:
            continue
        if local not in loop_list:
            continue  # notch vertex ended up interior; nothing to mark
        k = loop_list.index(local)
        point = stitch[k]
        tangent = stitch[(k + 1) % len(stitch)] - stitch[k - 1]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        outward = np.array([tangent[1], -tangent[0]])  # right of travel on a CCW loop
        notches.append((point, outward))
    return notches


def pack_layouts(layouts: list[PanelLayout], gap: float = 20.0) -> None:
    """Arrange layouts side by side (in place): simple left-to-right shelf."""
    x = 0.0
    for layout in layouts:
        shift = np.array([x - layout.cut[:, 0].min(), 0.0])
        _shift_layout(layout, shift)
        x = layout.cut[:, 0].max() + gap


def _shift_layout(layout: PanelLayout, shift: np.ndarray) -> None:
    layout.stitch = layout.stitch + shift
    layout.cut = layout.cut + shift
    layout.notches = [(p + shift, d) for p, d in layout.notches]
    layout.grain_arrow = (layout.grain_arrow[0] + shift, layout.grain_arrow[1] + shift)
    layout.label_pos = layout.label_pos + shift


def sheet_bbox(layouts: list[PanelLayout], margin: float = 5.0) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] over all cut lines, padded by margin."""
    pts = np.concatenate([layout.cut for layout in layouts])
    return np.concatenate([pts.min(axis=0) - margin, pts.max(axis=0) + margin])


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def svg_content_group(layouts: list[PanelLayout], flip_height: float) -> ET.Element:
    """All panels as one SVG <g>. flip_height maps our y-up mm coordinates
    to SVG's y-down convention."""
    group = ET.Element("g", {"id": "pattern"})
    for layout in layouts:
        group.append(_panel_group(layout, flip_height))
    return group


def _panel_group(layout: PanelLayout, h: float) -> ET.Element:
    def pts(a: np.ndarray) -> str:
        return " ".join(f"{x:.3f},{h - y:.3f}" for x, y in a)

    g = ET.Element("g", {"id": f"panel-{layout.name}"})
    ET.SubElement(
        g,
        "polygon",
        {
            "points": pts(layout.cut),
            "fill": "none",
            "stroke": "black",
            "stroke-width": "0.4",
        },
    )
    ET.SubElement(
        g,
        "polygon",
        {
            "points": pts(layout.stitch),
            "fill": "none",
            "stroke": "black",
            "stroke-width": "0.25",
            "stroke-dasharray": "4 2",
        },
    )
    for point, outward in layout.notches:
        a = point - outward * 2.0
        b = point + outward * (layout.seam_allowance + 2.0)
        ET.SubElement(
            g,
            "line",
            {
                "x1": f"{a[0]:.3f}",
                "y1": f"{h - a[1]:.3f}",
                "x2": f"{b[0]:.3f}",
                "y2": f"{h - b[1]:.3f}",
                "stroke": "black",
                "stroke-width": "0.6",
            },
        )
    start, end = layout.grain_arrow
    ET.SubElement(
        g,
        "line",
        {
            "x1": f"{start[0]:.3f}",
            "y1": f"{h - start[1]:.3f}",
            "x2": f"{end[0]:.3f}",
            "y2": f"{h - end[1]:.3f}",
            "stroke": "black",
            "stroke-width": "0.4",
        },
    )
    for tip, sign in ((end, 1.0), (start, -1.0)):
        head = (
            f"{tip[0]:.3f},{h - tip[1]:.3f} "
            f"{tip[0] - 1.5:.3f},{h - (tip[1] - sign * 4.0):.3f} "
            f"{tip[0] + 1.5:.3f},{h - (tip[1] - sign * 4.0):.3f}"
        )
        ET.SubElement(g, "polygon", {"points": head, "fill": "black"})
    text = ET.SubElement(
        g,
        "text",
        {
            "x": f"{layout.label_pos[0]:.3f}",
            "y": f"{h - layout.label_pos[1]:.3f}",
            "font-size": "8",
            "font-family": "sans-serif",
        },
    )
    text.text = layout.label
    return g


def write_svg(layouts: list[PanelLayout], path: str) -> None:
    """One SVG sheet with all panels, in real-world millimetres."""
    bbox = sheet_bbox(layouts)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    ET.register_namespace("", SVG_NS)
    svg = ET.Element(
        "svg",
        {
            "xmlns": SVG_NS,
            "width": f"{w:.2f}mm",
            "height": f"{h:.2f}mm",
            "viewBox": f"{bbox[0]:.3f} 0 {w:.3f} {h:.3f}",
        },
    )
    svg.append(svg_content_group(layouts, flip_height=bbox[3]))
    ET.ElementTree(svg).write(path, xml_declaration=True, encoding="unicode")


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------


def write_dxf(layouts: list[PanelLayout], path: str) -> None:
    """DXF (mm) with layers CUT, STITCH, NOTCH, ANNOT."""
    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM
    for name, color in (("CUT", 1), ("STITCH", 3), ("NOTCH", 5), ("ANNOT", 7)):
        doc.layers.add(name, color=color)
    msp = doc.modelspace()

    for layout in layouts:
        msp.add_lwpolyline(
            layout.cut, close=True, dxfattribs={"layer": "CUT"}
        )
        msp.add_lwpolyline(
            layout.stitch, close=True, dxfattribs={"layer": "STITCH"}
        )
        for point, outward in layout.notches:
            a = point - outward * 2.0
            b = point + outward * (layout.seam_allowance + 2.0)
            msp.add_line(a, b, dxfattribs={"layer": "NOTCH"})
        start, end = layout.grain_arrow
        msp.add_line(start, end, dxfattribs={"layer": "ANNOT"})
        msp.add_text(
            layout.label,
            height=8.0,
            dxfattribs={"layer": "ANNOT"},
        ).set_placement(tuple(layout.label_pos))

    doc.saveas(path)
