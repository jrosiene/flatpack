"""Fabric-aware warp flagging of panels at split time."""

import numpy as np
import pytest

from flatpack.analysis import HIGH_FRACTION, analyze, analyze_panel, flatten_and_fit
from flatpack.seams import spec_from_dict, split_mesh
from flatpack.synthetic import make_open_tube, make_plane, make_sphere_patch

N = 21


def sphere_spec(fabric="rigid"):
    seam = [i * N + N // 2 for i in range(N)]
    return spec_from_dict(
        {
            "seams": [{"name": "c", "path": seam}],
            "panels": [{"name": "L", "anchor_face": 0, "fabric": fabric}],
        }
    )


def analyze_sphere(fabric, half_width=65.0):
    mesh = make_sphere_patch(radius=100.0, half_width=half_width, n=N)
    panel = next(p for p in split_mesh(mesh, sphere_spec(fabric)) if p.name == "L")
    return analyze_panel(panel)


def test_flat_panel_is_ok():
    mesh = make_plane(half_width=50.0, n=N)
    seam = [i * N + N // 2 for i in range(N)]
    spec = spec_from_dict({"seams": [{"name": "c", "path": seam}]})
    for panel in split_mesh(mesh, spec):
        a = analyze_panel(panel)
        assert a.severity == "ok"
        assert a.bad_area_fraction < 0.02
        assert "within fabric tolerance" in a.advice


def test_deeply_curved_panel_is_flagged_high():
    a = analyze_sphere("rigid")
    assert a.severity == "high"
    assert a.bad_area_fraction >= HIGH_FRACTION
    assert a.worst_point_3d is not None and len(a.worst_point_3d) == 3
    assert "dart" in a.advice or "relief" in a.advice


def test_stretch_fabric_helps_but_darts_remain():
    """A stretchier fabric absorbs stretch-short areas (relief) and so
    warps less overall, but excess material still needs darts."""
    rigid = analyze_sphere("rigid")
    stretch = analyze_sphere("ultrastretch")
    assert stretch.relief_area_fraction < rigid.relief_area_fraction
    assert stretch.bad_area_fraction < rigid.bad_area_fraction
    # Fabric stretch cannot remove excess material: darts are still needed.
    assert stretch.dart_area_fraction > 0.0


def test_severity_grades_with_curvature():
    gentle = analyze_sphere("rigid", half_width=25.0)
    steep = analyze_sphere("rigid", half_width=65.0)
    order = {"ok": 0, "marginal": 1, "high": 2, "error": 3}
    assert order[gentle.severity] < order[steep.severity]
    assert gentle.bad_area_fraction < steep.bad_area_fraction


def test_unflattenable_panel_reports_error():
    """A tube with no seam still wraps around; flag it rather than crash."""
    tube = make_open_tube(radius=80.0, height=300.0, n_around=24, n_height=9)
    a = analyze(tube, spec_from_dict({}))[0]
    assert a.severity == "error"
    assert "seam" in a.advice.lower()
    assert a.worst_point_3d is None


def test_worst_point_is_on_the_mesh():
    mesh = make_sphere_patch(radius=100.0, half_width=65.0, n=N)
    panel = next(p for p in split_mesh(mesh, sphere_spec("rigid")) if p.name == "L")
    a = analyze_panel(panel)
    # Worst point is a triangle centroid, so it lies within the panel bbox.
    lo = panel.vertices.min(axis=0)
    hi = panel.vertices.max(axis=0)
    assert np.all(np.array(a.worst_point_3d) >= lo - 1e-6)
    assert np.all(np.array(a.worst_point_3d) <= hi + 1e-6)


def test_pipeline_matches_analysis():
    """The generated report and the live warp analysis use the same
    flatten_and_fit, so their fabric-fit numbers agree."""
    mesh = make_sphere_patch(radius=100.0, half_width=65.0, n=N)
    panel = next(p for p in split_mesh(mesh, sphere_spec("silnylon")) if p.name == "L")
    flat, fabric, fit = flatten_and_fit(panel)
    a = analyze_panel(panel)
    weights = flat.distortion.area_3d / flat.distortion.area_3d.sum()
    expected_bad = float(weights[fit.needs_dart | fit.needs_relief].sum())
    assert a.bad_area_fraction == pytest.approx(expected_bad)
