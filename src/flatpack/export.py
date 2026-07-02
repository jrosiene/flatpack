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

from flatpack.meshutil import boundary_loops, unique_edges
from flatpack.seams import Panel

SVG_NS = "http://www.w3.org/2000/svg"

BARTACK_LENGTH = 12.0  # drawn size of a bar tack symbol, mm


@dataclass
class Mark2D:
    """A sewing mark placed on the flattened panel."""

    pos: np.ndarray  # (2,)
    direction: np.ndarray  # (2,) unit vector (bar tack orientation)
    type: str  # "bartack" | "attach"
    label: str


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
    darts: list[tuple[np.ndarray, np.ndarray]]  # per dart: two legs, apex first
    marks: list[Mark2D]


def layout_panel(
    panel: Panel,
    uv: np.ndarray,
    seam_allowance: float = 10.0,
) -> PanelLayout:
    """Build printable geometry for one flattened panel.

    Rotates the layout so the grainline (panel.spec.grain vertex pair, if
    given) points straight up, and translates it to the positive quadrant.
    The same rigid transform is applied to every feature (boundary, darts,
    marks) so they stay in register.
    """
    faces = np.asarray(panel.faces, dtype=np.int64)
    loop = boundary_loops(faces)[0]

    uv_t = uv @ _grain_rotation(panel, uv).T
    stitch = uv_t[loop].astype(float)

    polygon = Polygon(stitch)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    cut_polygon = polygon.buffer(
        seam_allowance, join_style="mitre", mitre_limit=4.0
    )
    cut = np.asarray(cut_polygon.exterior.coords)[:-1]

    # Shift everything to the positive quadrant with a small margin.
    shift = seam_allowance + 5.0 - np.min(cut, axis=0)
    uv_t = uv_t + shift
    stitch = stitch + shift
    cut = cut + shift
    polygon = Polygon(stitch)

    notches = _notches(panel, stitch, loop)

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
        darts=_dart_layouts(panel, uv_t),
        marks=_mark_layouts(panel, uv_t),
    )


def _grain_rotation(panel: Panel, uv: np.ndarray) -> np.ndarray:
    """2x2 rotation putting the panel's grainline along +y.

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
        centered = uv - uv.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
    angle = np.arctan2(direction[1], direction[0])
    rot = np.pi / 2 - angle
    c, s = np.cos(rot), np.sin(rot)
    return np.array([[c, -s], [s, c]])


def _dart_layouts(panel: Panel, uv_t: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two legs of each dart, as point arrays from apex to mouth.

    Opening the dart slit duplicated its interior vertices; the two copies
    of each are separated in uv by the dart intake. Copies are chained
    into legs by following actual mesh edges from the apex outward.
    """
    edges = {tuple(e) for e in unique_edges(np.asarray(panel.faces, dtype=np.int64))}

    def connected(a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in edges

    layouts = []
    for path in panel.darts:
        copies = [
            list(np.nonzero(panel.orig_vertex_index == v)[0]) for v in path
        ]
        if any(not c for c in copies):
            continue  # dart not (fully) inside this panel
        apex = copies[-1]
        leg1, leg2 = [apex[0]], [apex[-1]]
        for cands in reversed(copies[:-1]):
            if len(cands) == 1:
                leg1.append(cands[0])
                leg2.append(cands[0])
            elif connected(leg1[-1], cands[0]) or connected(leg2[-1], cands[1]):
                leg1.append(cands[0])
                leg2.append(cands[1])
            else:
                leg1.append(cands[1])
                leg2.append(cands[0])
        layouts.append((uv_t[leg1], uv_t[leg2]))
    return layouts


def _mark_layouts(panel: Panel, uv_t: np.ndarray) -> list[Mark2D]:
    marks = []
    for mark in panel.marks:
        local = panel.local_index(mark.vertex)
        pos = uv_t[local]
        direction = np.array([1.0, 0.0])
        if mark.toward is not None:
            try:
                d = uv_t[panel.local_index(mark.toward)] - pos
            except KeyError:
                d = None
            if d is not None and np.linalg.norm(d) > 1e-9:
                direction = d / np.linalg.norm(d)
        marks.append(
            Mark2D(pos=pos, direction=direction, type=mark.type, label=mark.label)
        )
    return marks


def _loop_is_ccw(loop: np.ndarray) -> bool:
    x, y = loop[:, 0], loop[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) > 0


def _notches(
    panel: Panel,
    stitch: np.ndarray,
    loop: np.ndarray,
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
    layout.darts = [(a + shift, b + shift) for a, b in layout.darts]
    for mark in layout.marks:
        mark.pos = mark.pos + shift


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
    for leg1, leg2 in layout.darts:
        # Dart legs: fold/stitch lines from the mouth to the apex, plus a
        # small circle at the apex so it survives cutting out.
        for leg in (leg1, leg2):
            ET.SubElement(
                g,
                "polyline",
                {
                    "points": pts(leg),
                    "fill": "none",
                    "stroke": "black",
                    "stroke-width": "0.3",
                    "stroke-dasharray": "6 2 1 2",
                },
            )
        apex = leg1[0]
        ET.SubElement(
            g,
            "circle",
            {
                "cx": f"{apex[0]:.3f}",
                "cy": f"{h - apex[1]:.3f}",
                "r": "1.5",
                "fill": "none",
                "stroke": "black",
                "stroke-width": "0.3",
            },
        )
    for mark in layout.marks:
        g.append(_mark_svg(mark, h))
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


def _mark_svg(mark: Mark2D, h: float) -> ET.Element:
    """Symbol for a sewing mark: bar tack = thick bar, attach = target."""
    g = ET.Element("g", {"class": f"mark-{mark.type}"})
    x, y = mark.pos
    if mark.type == "bartack":
        half = mark.direction * (BARTACK_LENGTH / 2)
        a, b = mark.pos - half, mark.pos + half
        ET.SubElement(
            g,
            "line",
            {
                "x1": f"{a[0]:.3f}",
                "y1": f"{h - a[1]:.3f}",
                "x2": f"{b[0]:.3f}",
                "y2": f"{h - b[1]:.3f}",
                "stroke": "black",
                "stroke-width": "2.5",
                "stroke-linecap": "butt",
            },
        )
    else:
        ET.SubElement(
            g,
            "circle",
            {
                "cx": f"{x:.3f}",
                "cy": f"{h - y:.3f}",
                "r": "3",
                "fill": "none",
                "stroke": "black",
                "stroke-width": "0.4",
            },
        )
        for dx, dy in ((4.5, 0), (0, 4.5)):
            ET.SubElement(
                g,
                "line",
                {
                    "x1": f"{x - dx:.3f}",
                    "y1": f"{h - (y - dy):.3f}",
                    "x2": f"{x + dx:.3f}",
                    "y2": f"{h - (y + dy):.3f}",
                    "stroke": "black",
                    "stroke-width": "0.4",
                },
            )
    if mark.label:
        text = ET.SubElement(
            g,
            "text",
            {
                "x": f"{x + 5:.3f}",
                "y": f"{h - y - 4:.3f}",
                "font-size": "5",
                "font-family": "sans-serif",
            },
        )
        text.text = mark.label
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
    for name, color in (
        ("CUT", 1),
        ("STITCH", 3),
        ("NOTCH", 5),
        ("DART", 6),
        ("MARK", 4),
        ("ANNOT", 7),
    ):
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
        for leg1, leg2 in layout.darts:
            msp.add_lwpolyline(leg1, dxfattribs={"layer": "DART"})
            msp.add_lwpolyline(leg2, dxfattribs={"layer": "DART"})
            msp.add_circle(tuple(leg1[0]), radius=1.5, dxfattribs={"layer": "DART"})
        for mark in layout.marks:
            if mark.type == "bartack":
                half = mark.direction * (BARTACK_LENGTH / 2)
                msp.add_line(
                    mark.pos - half, mark.pos + half, dxfattribs={"layer": "MARK"}
                )
            else:
                msp.add_circle(tuple(mark.pos), radius=3.0, dxfattribs={"layer": "MARK"})
                msp.add_line(
                    mark.pos - (4.5, 0), mark.pos + (4.5, 0), dxfattribs={"layer": "MARK"}
                )
                msp.add_line(
                    mark.pos - (0, 4.5), mark.pos + (0, 4.5), dxfattribs={"layer": "MARK"}
                )
            if mark.label:
                msp.add_text(
                    mark.label, height=5.0, dxfattribs={"layer": "MARK"}
                ).set_placement(tuple(mark.pos + (5.0, 4.0)))
        start, end = layout.grain_arrow
        msp.add_line(start, end, dxfattribs={"layer": "ANNOT"})
        msp.add_text(
            layout.label,
            height=8.0,
            dxfattribs={"layer": "ANNOT"},
        ).set_placement(tuple(layout.label_pos))

    doc.saveas(path)
