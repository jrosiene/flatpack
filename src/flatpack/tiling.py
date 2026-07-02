"""Tile a pattern sheet across letter/A4 pages for home printing.

Each page SVG has the size of the printable area (page minus printer
margin) in real millimetres — print at 100% / "actual size". The full
pattern is embedded in every page, shifted so that page's window shows
through; the SVG viewport clips the rest. Pages overlap by a configurable
glue strip and carry:

- a dashed glue line marking where the next page overlaps,
- corner crop marks,
- a page label like "B2" (column letter, row number) for assembly.
"""

from __future__ import annotations

import copy
import string
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flatpack.export import PanelLayout, SVG_NS, sheet_bbox, svg_content_group

# (width, height) of the paper in mm.
PAGE_SIZES_MM = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
}


@dataclass
class PageWindow:
    label: str  # e.g. "A1"
    x0: float  # window origin in sheet coordinates (y up)
    y0: float
    width: float
    height: float


def page_windows(
    bbox: np.ndarray,
    page: str = "letter",
    printer_margin: float = 10.0,
    overlap: float = 15.0,
) -> list[PageWindow]:
    """Grid of page windows covering bbox = [xmin, ymin, xmax, ymax].

    Columns are lettered left to right, rows numbered top to bottom (the
    order you'd tape them together).
    """
    paper_w, paper_h = PAGE_SIZES_MM[page]
    win_w = paper_w - 2 * printer_margin
    win_h = paper_h - 2 * printer_margin
    step_x = win_w - overlap
    step_y = win_h - overlap
    if step_x <= 0 or step_y <= 0:
        raise ValueError("overlap larger than the printable page area")

    total_w = bbox[2] - bbox[0]
    total_h = bbox[3] - bbox[1]
    cols = max(1, int(np.ceil((total_w - overlap) / step_x)))
    rows = max(1, int(np.ceil((total_h - overlap) / step_y)))

    windows = []
    for row in range(rows):
        for col in range(cols):
            label = f"{_column_letters(col)}{row + 1}"
            windows.append(
                PageWindow(
                    label=label,
                    x0=bbox[0] + col * step_x,
                    # Row 1 is the TOP row: highest y in our y-up coordinates.
                    y0=bbox[3] - win_h - row * step_y,
                    width=win_w,
                    height=win_h,
                )
            )
    return windows


def _column_letters(col: int) -> str:
    letters = string.ascii_uppercase
    result = ""
    col += 1
    while col:
        col, rem = divmod(col - 1, 26)
        result = letters[rem] + result
    return result


def write_tiled_svgs(
    layouts: list[PanelLayout],
    outdir: str | Path,
    page: str = "letter",
    printer_margin: float = 10.0,
    overlap: float = 15.0,
    prefix: str = "page",
    edge_units: str | None = None,
) -> list[Path]:
    """Write one SVG per page; returns the paths written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    bbox = sheet_bbox(layouts)
    windows = page_windows(bbox, page, printer_margin, overlap)
    content = svg_content_group(layouts, flip_height=bbox[3], edge_units=edge_units)

    ET.register_namespace("", SVG_NS)
    paths = []
    for win in windows:
        svg = ET.Element(
            "svg",
            {
                "xmlns": SVG_NS,
                "width": f"{win.width:.2f}mm",
                "height": f"{win.height:.2f}mm",
                "viewBox": f"0 0 {win.width:.3f} {win.height:.3f}",
            },
        )
        # Content coordinates are y-down with origin at the sheet's top-left
        # (bbox[0], bbox[3]); shift that page's window to the viewport.
        shift_x = -(win.x0 - bbox[0])
        shift_y = -(bbox[3] - (win.y0 + win.height))
        holder = ET.SubElement(
            svg, "g", {"transform": f"translate({shift_x:.3f} {shift_y:.3f})"}
        )
        holder.append(copy.deepcopy(content))
        _add_page_marks(svg, win, overlap)
        path = outdir / f"{prefix}_{win.label}.svg"
        ET.ElementTree(svg).write(path, xml_declaration=True, encoding="unicode")
        paths.append(path)
    return paths


def write_pattern_pdf(page_svg_paths: list[Path], out_pdf: str | Path) -> Path:
    """Bundle the tiled page SVGs into one print-ready multi-page PDF.

    One PDF page per tile, in the same order the pages were written (row by
    row), each sized to the printable area in real millimetres so it prints
    at 100%. The full pattern is clipped to the page window on the canvas
    (svglib does not clip to the SVG viewport by itself), so each page
    shows only its own tile. Vector output — crisp at any zoom.
    """
    from reportlab.pdfgen import canvas  # heavy imports, kept local
    from svglib.svglib import svg2rlg

    out_pdf = Path(out_pdf)
    pdf = canvas.Canvas(str(out_pdf))
    for svg_path in page_svg_paths:
        drawing = svg2rlg(str(svg_path))
        pdf.setPageSize((drawing.width, drawing.height))
        pdf.saveState()
        clip = pdf.beginPath()
        clip.rect(0, 0, drawing.width, drawing.height)
        pdf.clipPath(clip, stroke=0, fill=0)
        drawing.drawOn(pdf, 0, 0)
        pdf.restoreState()
        pdf.showPage()
    pdf.save()
    return out_pdf


def _add_page_marks(svg: ET.Element, win: PageWindow, overlap: float) -> None:
    marks = ET.SubElement(svg, "g", {"id": "page-marks"})
    w, h = win.width, win.height

    # Corner crop marks.
    for cx, cy, dx, dy in (
        (0, 0, 1, 1),
        (w, 0, -1, 1),
        (0, h, 1, -1),
        (w, h, -1, -1),
    ):
        for ex, ey in ((dx * 8, 0), (0, dy * 8)):
            ET.SubElement(
                marks,
                "line",
                {
                    "x1": f"{cx}",
                    "y1": f"{cy}",
                    "x2": f"{cx + ex}",
                    "y2": f"{cy + ey}",
                    "stroke": "black",
                    "stroke-width": "0.3",
                },
            )

    # Dashed glue lines: the strip beyond them is repeated on the next page.
    for x1, y1, x2, y2 in (
        (w - overlap, 0, w - overlap, h),  # right edge strip
        (0, h - overlap, w, h - overlap),  # bottom edge strip
    ):
        ET.SubElement(
            marks,
            "line",
            {
                "x1": f"{x1:.3f}",
                "y1": f"{y1:.3f}",
                "x2": f"{x2:.3f}",
                "y2": f"{y2:.3f}",
                "stroke": "grey",
                "stroke-width": "0.2",
                "stroke-dasharray": "6 3",
            },
        )

    label = ET.SubElement(
        marks,
        "text",
        {
            "x": "12",
            "y": "12",
            "font-size": "6",
            "font-family": "sans-serif",
            "fill": "grey",
        },
    )
    label.text = f"page {win.label}"
