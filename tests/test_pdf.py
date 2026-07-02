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


def test_pdf_pages_are_printable_size(tmp_path):
    """Each PDF page is the printable area (page minus margins) in points."""
    import pypdfium2 as pdfium

    from flatpack.pipeline import process

    mesh = make_sphere_patch(radius=150.0, half_width=80.0, n=15)
    n = 15
    spec = spec_from_dict({"seams": [{"name": "c", "path": [i * n + n // 2 for i in range(n)]}]})
    process(mesh, spec, tmp_path, page="letter")

    doc = pdfium.PdfDocument(str(tmp_path / "pattern_tiled.pdf"))
    width_pt, height_pt = doc[0].get_size()
    # Letter (215.9 x 279.4 mm) minus 10 mm margins each side, in points.
    assert width_pt == pytest.approx((215.9 - 20) * 72 / 25.4, abs=1.0)
    assert height_pt == pytest.approx((279.4 - 20) * 72 / 25.4, abs=1.0)


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
