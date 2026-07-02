"""Print-ready PDF bundling of the tiled pages."""

import glob

import pytest

from flatpack.seams import spec_from_dict
from flatpack.synthetic import make_sphere_patch


def test_process_writes_valid_multipage_pdf(tmp_path):
    from flatpack.pipeline import process

    mesh = make_sphere_patch(radius=200.0, half_width=120.0, n=25)
    n = 25
    spec = spec_from_dict({"seams": [{"name": "c", "path": [i * n + n // 2 for i in range(n)]}]})
    process(mesh, spec, tmp_path)

    pdf = tmp_path / "pattern_tiled.pdf"
    assert pdf.exists()
    data = pdf.read_bytes()
    assert data[:5] == b"%PDF-"

    n_pages = len(glob.glob(str(tmp_path / "page_*.svg")))
    assert n_pages > 1
    # One PDF page per tiled SVG page.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    assert len(doc) == n_pages


def test_pdf_pages_are_full_paper_size(tmp_path):
    """Each PDF page is the full paper size (so it prints at true scale
    without the printer rescaling to fit a smaller page)."""
    import pypdfium2 as pdfium

    from flatpack.pipeline import process

    mesh = make_sphere_patch(radius=150.0, half_width=80.0, n=15)
    n = 15
    spec = spec_from_dict({"seams": [{"name": "c", "path": [i * n + n // 2 for i in range(n)]}]})
    process(mesh, spec, tmp_path, page="letter")

    doc = pdfium.PdfDocument(str(tmp_path / "pattern_tiled.pdf"))
    width_pt, height_pt = doc[0].get_size()
    # Full US Letter: 215.9 x 279.4 mm in points.
    assert width_pt == pytest.approx(215.9 * 72 / 25.4, abs=1.0)
    assert height_pt == pytest.approx(279.4 * 72 / 25.4, abs=1.0)


def test_pdf_content_is_inside_the_printer_margin(tmp_path):
    """All ink sits inside the printer margin, not in the non-printable
    border - so nothing is clipped when printing at 100%."""
    import numpy as np
    import pypdfium2 as pdfium

    from flatpack.pipeline import process
    from flatpack.tiling import DEFAULT_PRINTER_MARGIN

    mesh = make_sphere_patch(radius=200.0, half_width=120.0, n=25)
    n = 25
    spec = spec_from_dict({"seams": [{"name": "c", "path": [i * n + n // 2 for i in range(n)]}]})
    process(mesh, spec, tmp_path, page="letter")

    doc = pdfium.PdfDocument(str(tmp_path / "pattern_tiled.pdf"))
    scale = 2.0  # render at 2 px/pt
    margin_px = DEFAULT_PRINTER_MARGIN * 72 / 25.4 * scale
    for page in doc:
        arr = page.render(scale=scale).to_numpy()
        ink = np.argwhere((arr[:, :, :3] < 100).any(axis=2))
        if ink.size == 0:
            continue
        top, left = ink.min(axis=0)
        bottom, right = ink.max(axis=0)
        h, w = arr.shape[:2]
        # Allow a couple of px of antialiasing slack.
        assert left >= margin_px - 3 and top >= margin_px - 3
        assert right <= w - margin_px + 3 and bottom <= h - margin_px + 3


def test_pages_show_distinct_tiles(tmp_path):
    """Each page is clipped to its own window, not the whole pattern."""
    import numpy as np
    import pypdfium2 as pdfium

    from flatpack.pipeline import process

    mesh = make_sphere_patch(radius=200.0, half_width=120.0, n=25)
    n = 25
    spec = spec_from_dict({"seams": [{"name": "c", "path": [i * n + n // 2 for i in range(n)]}]})
    process(mesh, spec, tmp_path)

    doc = pdfium.PdfDocument(str(tmp_path / "pattern_tiled.pdf"))
    ink = []
    for page in doc:
        arr = page.render(scale=0.4).to_numpy()
        ink.append(int((arr < 128).sum()))
    # If clipping failed, every page would carry the whole pattern and have
    # near-identical ink; require real variation between pages.
    assert max(ink) > 0
    assert np.std(ink) > 0.05 * np.mean(ink)
