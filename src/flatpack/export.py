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
from shapely.geometry import LineString, Point, Polygon

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
class EdgeLabel:
    """Length annotation for one straight run of a panel's boundary."""

    pos: np.ndarray  # (2,) midpoint, nudged toward the panel interior
    angle_deg: float  # text rotation, kept upright
    length_mm: float  # true seam length along the boundary (not the chord)


def format_edge_length(length_mm: float, units: str) -> str:
    """Human formatting for edge labels: 'cm' or 'in'."""
    if units == "cm":
        return f"{length_mm / 10.0:.1f} cm"
    if units == "in":
        return f'{length_mm / 25.4:.2f}"'
    raise ValueError(f"unknown edge label units {units!r} (use 'cm' or 'in')")


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
    edge_labels: list[EdgeLabel]
    # Registration ticks along seams, matched between mating panels.
    seam_ticks: list[tuple[np.ndarray, np.ndarray]]  # (point on stitch, outward)


SEAM_MARKER_INTERVAL = 75.0  # target spacing of seam alignment ticks, mm
MIN_SEAM_FOR_MARK = 40.0  # shorter seams get no interior tick


def layout_panel(
    panel: Panel,
    uv: np.ndarray,
    seam_allowance: float = 10.0,
    seam_marker_interval: float = SEAM_MARKER_INTERVAL,
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
    seam_ticks = _seam_ticks(panel, stitch, loop, seam_marker_interval)

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
        edge_labels=_edge_labels(stitch, polygon),
        seam_ticks=seam_ticks,
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


def _edge_labels(
    stitch: np.ndarray,
    polygon: Polygon,
    corner_tolerance: float = 2.0,
    min_length: float = 25.0,
) -> list[EdgeLabel]:
    """Length annotations for the straight runs of the stitch boundary.

    The boundary polyline is simplified (Douglas-Peucker) to find its
    corners; each run between consecutive corners is one 'edge'. The
    printed length is the true polyline length along the run — for a
    straight edge that's the chord; for a gentle curve it's the seam
    length, which is what you'd measure with a tape. Runs shorter than
    min_length are skipped to avoid clutter.

    The ring is rotated to start at the sharpest corner first: the loop's
    arbitrary start vertex is always kept by the simplifier, and starting
    mid-edge would split one straight edge into two labels.
    """
    stitch = np.roll(stitch, -_sharpest_corner(stitch), axis=0)
    ring = np.vstack([stitch, stitch[:1]])
    simplified = np.asarray(
        LineString(ring).simplify(corner_tolerance).coords
    )

    # Douglas-Peucker keeps original vertices, so corners map back to
    # exact boundary indices. Keep first occurrences: the ring's closing
    # point duplicates index 0 and must map to it, with the wraparound
    # handled below.
    index_of: dict = {}
    for i, (x, y) in enumerate(ring):
        index_of.setdefault((round(x, 9), round(y, 9)), i)
    corner_indices = [
        index_of[(round(x, 9), round(y, 9))] for x, y in simplified
    ]

    seg_lengths = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])

    labels = []
    for a, b in zip(corner_indices[:-1], corner_indices[1:]):
        length = float(cumulative[b] - cumulative[a])
        if b <= a:  # segment closing the loop back to the start
            length += total
        chord = ring[b] - ring[a]
        chord_len = float(np.linalg.norm(chord))
        if length < min_length or chord_len < 1e-9:
            continue

        angle = np.degrees(np.arctan2(chord[1], chord[0]))
        if angle > 90.0 or angle <= -90.0:
            angle += 180.0 if angle <= -90.0 else -180.0

        midpoint = (ring[a] + ring[b]) / 2.0
        normal = np.array([-chord[1], chord[0]]) / chord_len
        inward = midpoint + normal * 5.0
        if not polygon.contains(Point(inward)):
            inward = midpoint - normal * 5.0
        labels.append(EdgeLabel(pos=inward, angle_deg=float(angle), length_mm=length))
    return labels


def _sharpest_corner(stitch: np.ndarray) -> int:
    """Index of the boundary vertex with the largest turning angle."""
    before = stitch - np.roll(stitch, 1, axis=0)
    after = np.roll(stitch, -1, axis=0) - stitch
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.einsum("ij,ij->i", before, after)
    return int(np.argmax(np.abs(np.arctan2(cross, dot))))


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


def _seam_ticks(
    panel: Panel,
    stitch: np.ndarray,
    loop: np.ndarray,
    interval: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Registration ticks along the panel's true seam edges.

    Ticks are placed at matching positions on the two panels that share a
    seam: each seam run is oriented from the end with the smaller *original*
    vertex index (so both panels agree on direction), its tick count comes
    from the run's 3D length (identical on both sides), and ticks sit at the
    same fractions along it — so tick k on one panel lines up with tick k on
    the other when the seam is sewn. Also works for a single panel whose two
    sides of an opened seam meet (e.g. a tube seam).
    """
    if not panel.seam_edges:
        return []
    if not _loop_is_ccw(stitch):
        stitch = stitch[::-1]
        loop = loop[::-1]

    n = len(loop)
    is_seam = np.zeros(n, dtype=bool)  # is_seam[i]: edge (loop[i], loop[i+1])
    for i in range(n):
        a, b = int(loop[i]), int(loop[(i + 1) % n])
        if (min(a, b), max(a, b)) in panel.seam_edges:
            is_seam[i] = True

    ticks: list[tuple[np.ndarray, np.ndarray]] = []
    for run in _cyclic_true_runs(is_seam):
        # run is a list of edge indices; its vertices are loop[i] for i in
        # run plus the final loop[run[-1]+1].
        vpos = [run[0]] + [(e + 1) % n for e in run]
        verts = [int(loop[i]) for i in vpos]
        pts2d = np.array([stitch[i] for i in vpos])
        origs = [int(panel.orig_vertex_index[v]) for v in verts]
        if origs[0] > origs[-1]:  # canonical: start from smaller original id
            verts, pts2d = verts[::-1], pts2d[::-1]

        pos3d = panel.vertices[verts]
        seg3 = np.linalg.norm(np.diff(pos3d, axis=0), axis=1)
        length3 = float(seg3.sum())
        if length3 < MIN_SEAM_FOR_MARK:
            continue
        count = max(1, round(length3 / interval) - 1)

        seg2 = np.linalg.norm(np.diff(pts2d, axis=0), axis=1)
        cum2 = np.concatenate([[0.0], np.cumsum(seg2)])
        total2 = float(cum2[-1])
        if total2 < 1e-9:
            continue
        for k in range(1, count + 1):
            target = k / (count + 1) * total2
            j = int(np.searchsorted(cum2, target)) - 1
            j = min(max(j, 0), len(seg2) - 1)
            frac = (target - cum2[j]) / max(seg2[j], 1e-12)
            point = pts2d[j] + frac * (pts2d[j + 1] - pts2d[j])
            tangent = pts2d[j + 1] - pts2d[j]
            tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
            outward = np.array([tangent[1], -tangent[0]])  # right of a CCW loop
            ticks.append((point, outward))
    return ticks


def _cyclic_true_runs(mask: np.ndarray) -> list[list[int]]:
    """Maximal runs of consecutive True indices in a cyclic boolean array."""
    n = len(mask)
    if mask.all():
        return [list(range(n))]
    if not mask.any():
        return []
    start = int(np.argmin(mask))  # a False position to break the cycle at
    runs, current = [], []
    for step in range(n):
        i = (start + step) % n
        if mask[i]:
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


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
    layout.seam_ticks = [(p + shift, d) for p, d in layout.seam_ticks]
    layout.grain_arrow = (layout.grain_arrow[0] + shift, layout.grain_arrow[1] + shift)
    layout.label_pos = layout.label_pos + shift
    layout.darts = [(a + shift, b + shift) for a, b in layout.darts]
    for mark in layout.marks:
        mark.pos = mark.pos + shift
    for label in layout.edge_labels:
        label.pos = label.pos + shift


def sheet_bbox(layouts: list[PanelLayout], margin: float = 5.0) -> np.ndarray:
    """[xmin, ymin, xmax, ymax] over all cut lines, padded by margin."""
    pts = np.concatenate([layout.cut for layout in layouts])
    return np.concatenate([pts.min(axis=0) - margin, pts.max(axis=0) + margin])


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------


def svg_content_group(
    layouts: list[PanelLayout],
    flip_height: float,
    edge_units: str | None = None,
    seam_markers: bool = True,
) -> ET.Element:
    """All panels as one SVG <g>. flip_height maps our y-up mm coordinates
    to SVG's y-down convention. edge_units ('cm' or 'in') turns on edge
    length labels; seam_markers draws the seam alignment ticks."""
    group = ET.Element("g", {"id": "pattern"})
    for layout in layouts:
        group.append(_panel_group(layout, flip_height, edge_units, seam_markers))
    return group


def _panel_group(
    layout: PanelLayout,
    h: float,
    edge_units: str | None = None,
    seam_markers: bool = True,
) -> ET.Element:
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
    if seam_markers:
        for point, outward in layout.seam_ticks:
            # A light tick from the stitch line out to the cut edge.
            a = point - outward * 1.0
            b = point + outward * layout.seam_allowance
            ET.SubElement(
                g,
                "line",
                {
                    "x1": f"{a[0]:.3f}",
                    "y1": f"{h - a[1]:.3f}",
                    "x2": f"{b[0]:.3f}",
                    "y2": f"{h - b[1]:.3f}",
                    "stroke": "black",
                    "stroke-width": "0.35",
                    "class": "seam-tick",
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
    if edge_units:
        for label in layout.edge_labels:
            text = ET.SubElement(
                g,
                "text",
                {
                    "x": f"{label.pos[0]:.3f}",
                    "y": f"{h - label.pos[1]:.3f}",
                    "font-size": "5",
                    "font-family": "sans-serif",
                    "text-anchor": "middle",
                    "class": "edge-length",
                    # SVG y is flipped, so the rotation flips sign too.
                    "transform": (
                        f"rotate({-label.angle_deg:.2f} "
                        f"{label.pos[0]:.3f} {h - label.pos[1]:.3f})"
                    ),
                },
            )
            text.text = format_edge_length(label.length_mm, edge_units)
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


def write_svg(
    layouts: list[PanelLayout],
    path: str,
    edge_units: str | None = None,
    seam_markers: bool = True,
) -> None:
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
    svg.append(
        svg_content_group(
            layouts, flip_height=bbox[3], edge_units=edge_units, seam_markers=seam_markers
        )
    )
    ET.ElementTree(svg).write(path, xml_declaration=True, encoding="unicode")


# ---------------------------------------------------------------------------
# DXF
# ---------------------------------------------------------------------------


def write_dxf(
    layouts: list[PanelLayout],
    path: str,
    edge_units: str | None = None,
    seam_markers: bool = True,
) -> None:
    """DXF (mm) with layers CUT, STITCH, NOTCH, SEAMTICK, DART, MARK, ANNOT."""
    doc = ezdxf.new(setup=True)
    doc.units = ezdxf.units.MM
    for name, color in (
        ("CUT", 1),
        ("STITCH", 3),
        ("NOTCH", 5),
        ("SEAMTICK", 2),
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
        if seam_markers:
            for point, outward in layout.seam_ticks:
                a = point - outward * 1.0
                b = point + outward * layout.seam_allowance
                msp.add_line(a, b, dxfattribs={"layer": "SEAMTICK"})
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
        if edge_units:
            for label in layout.edge_labels:
                msp.add_text(
                    format_edge_length(label.length_mm, edge_units),
                    height=5.0,
                    rotation=label.angle_deg,
                    dxfattribs={"layer": "ANNOT"},
                ).set_placement(tuple(label.pos))
        start, end = layout.grain_arrow
        msp.add_line(start, end, dxfattribs={"layer": "ANNOT"})
        msp.add_text(
            layout.label,
            height=8.0,
            dxfattribs={"layer": "ANNOT"},
        ).set_placement(tuple(layout.label_pos))

    doc.saveas(path)
