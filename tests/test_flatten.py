"""Validate the LSCM flattening on surfaces with known behaviour."""

import numpy as np
import pytest

from flatpack.flatten import flatten, lscm
from flatpack.synthetic import (
    make_cylinder_patch,
    make_plane,
    make_saddle_patch,
    make_sphere_patch,
)


def test_plane_flattens_exactly():
    """Zero curvature: flattening is a rigid motion, distortion ~ none."""
    mesh = make_plane(half_width=50.0, n=11)
    result = flatten(mesh)
    assert np.allclose(result.distortion.sigma1, 1.0, atol=1e-8)
    assert np.allclose(result.distortion.sigma2, 1.0, atol=1e-8)


def test_cylinder_patch_is_developable():
    """Single curvature: must flatten with only discretisation-level error."""
    mesh = make_cylinder_patch(radius=100.0, half_width=60.0, n=31)
    result = flatten(mesh)
    strain = np.maximum(result.distortion.sigma1 - 1, 1 - result.distortion.sigma2)
    assert strain.max() < 5e-3


def test_sphere_patch_has_unavoidable_distortion():
    """Positive Gaussian curvature cannot flatten without distortion."""
    mesh = make_sphere_patch(radius=100.0, half_width=60.0, n=21)
    result = flatten(mesh)
    strain = np.maximum(result.distortion.sigma1 - 1, 1 - result.distortion.sigma2)
    assert strain.max() > 0.05  # clearly nonzero
    # ... but LSCM keeps it conformal: angles barely move even here.
    assert result.distortion.summary()["angle_error_deg_mean"] < 2.0
    # No inverted triangles.
    assert not result.distortion.flipped.any()


def test_sphere_distortion_grows_with_curvature():
    shallow = flatten(make_sphere_patch(radius=200.0, half_width=60.0, n=21))
    deep = flatten(make_sphere_patch(radius=100.0, half_width=60.0, n=21))
    worst = lambda r: (r.distortion.sigma1 - 1).max()
    assert worst(deep) > worst(shallow)


def test_saddle_patch_flattens_without_flips():
    mesh = make_saddle_patch(half_width=60.0, curvature=0.008, n=21)
    result = flatten(mesh)
    assert not result.distortion.flipped.any()
    strain = np.maximum(result.distortion.sigma1 - 1, 1 - result.distortion.sigma2)
    assert strain.max() > 0.01  # negative curvature still distorts


def test_total_area_is_preserved():
    mesh = make_sphere_patch(radius=100.0, half_width=60.0, n=21)
    result = flatten(mesh)
    uv_area = (result.distortion.area_ratio * result.distortion.area_3d).sum()
    assert uv_area == pytest.approx(mesh.area, rel=1e-6)


def test_closed_mesh_is_rejected():
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)
    with pytest.raises(ValueError, match="closed"):
        flatten(sphere)


def test_matches_libigl_reference():
    """Our LSCM should agree with libigl's on distortion statistics."""
    igl = pytest.importorskip("igl")
    from flatpack.distortion import distortion_report
    from flatpack.flatten import default_pins

    mesh = make_sphere_patch(radius=100.0, half_width=60.0, n=21)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    pins = default_pins(v, f)

    ours = lscm(v, f, pins=pins)

    b = np.array(pins, dtype=np.int64)
    bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    igl_out = igl.lscm(v, f, b, bc)
    # The bindings' return shape has varied across versions; find the uv array.
    candidates = igl_out if isinstance(igl_out, tuple) else (igl_out,)
    uv_igl = next(
        np.asarray(c)
        for c in candidates
        if isinstance(c, np.ndarray) and np.asarray(c).shape == (len(v), 2)
    )

    ours_report = distortion_report(v, f, ours)
    igl_report = distortion_report(v, f, uv_igl)
    # Conformal maps are unique up to similarity, so anisotropy (scale-free)
    # must match closely.
    assert np.allclose(
        ours_report.anisotropy, igl_report.anisotropy, rtol=1e-4, atol=1e-6
    )
