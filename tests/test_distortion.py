"""Distortion metrics against maps with known singular values."""

import numpy as np
import pytest

from flatpack.distortion import distortion_report


@pytest.fixture
def flat_square():
    """Two triangles forming a unit square in the z=0 plane."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


def test_identity_map(flat_square):
    vertices, faces = flat_square
    report = distortion_report(vertices, faces, vertices[:, :2])
    assert np.allclose(report.sigma1, 1.0)
    assert np.allclose(report.sigma2, 1.0)
    assert np.allclose(report.area_ratio, 1.0)
    assert np.allclose(report.max_angle_error_deg, 0.0, atol=1e-9)
    assert not report.flipped.any()


def test_pure_stretch_in_x(flat_square):
    vertices, faces = flat_square
    uv = vertices[:, :2] * np.array([1.25, 1.0])
    report = distortion_report(vertices, faces, uv)
    assert np.allclose(report.sigma1, 1.25)
    assert np.allclose(report.sigma2, 1.0)
    assert np.allclose(report.area_ratio, 1.25)
    # Max-stretch direction is +-x.
    assert np.allclose(np.abs(report.stretch_dir_uv[:, 0]), 1.0)


def test_uniform_scale_preserves_angles(flat_square):
    vertices, faces = flat_square
    report = distortion_report(vertices, faces, vertices[:, :2] * 2.0)
    assert np.allclose(report.anisotropy, 1.0)
    assert np.allclose(report.max_angle_error_deg, 0.0, atol=1e-9)
    assert np.allclose(report.area_ratio, 4.0)


def test_mirrored_map_is_flagged(flat_square):
    vertices, faces = flat_square
    uv = vertices[:, :2] * np.array([-1.0, 1.0])
    report = distortion_report(vertices, faces, uv)
    assert report.flipped.all()


def test_summary_and_worst_location(flat_square):
    vertices, faces = flat_square
    uv = vertices[:, :2].copy()
    uv[:, 0] *= 1.5  # both triangles stretched
    report = distortion_report(vertices, faces, uv)
    summary = report.summary()
    assert summary["triangles"] == 2
    assert summary["max_stretch_strain"] == pytest.approx(0.5)
    assert report.worst_triangle_uv().shape == (2,)
