"""Fabric capability model, fit checking, and anisotropic relaxation."""

import numpy as np
import pytest

from flatpack.fabric import (
    FABRICS,
    Fabric,
    fabric_fit,
    relax_for_fabric,
    relaxation_energy,
)
from flatpack.flatten import flatten
from flatpack.synthetic import make_sphere_patch


def test_capability_interpolates_between_axes():
    fabric = Fabric("test", stretch_along=0.30, stretch_cross=0.10)
    assert fabric.capability(np.array(0.0)) == pytest.approx(0.30)
    assert fabric.capability(np.array(np.pi / 2)) == pytest.approx(0.10)
    mid = fabric.capability(np.array(np.pi / 4))
    assert 0.10 < mid < 0.30


@pytest.fixture(scope="module")
def sphere_result():
    mesh = make_sphere_patch(radius=100.0, half_width=65.0, n=21)
    return mesh, flatten(mesh)


def test_stretch_fabric_needs_less_relief_than_woven(sphere_result):
    _, result = sphere_result
    woven = fabric_fit(result.distortion, FABRICS["silnylon"])
    stretchy = fabric_fit(result.distortion, FABRICS["ultrastretch"])
    assert (
        stretchy.needs_relief.sum() < woven.needs_relief.sum()
    ), "stretch fabric should absorb strain a woven cannot"
    # Excess material still needs darts regardless of stretch.
    assert (stretchy.needs_dart == woven.needs_dart).all()


def test_rigid_fabric_flags_all_significant_strain(sphere_result):
    _, result = sphere_result
    fit = fabric_fit(result.distortion, FABRICS["rigid"])
    strain = np.maximum(result.distortion.sigma1 - 1, 1 - result.distortion.sigma2)
    assert fit.ok.sum() == (strain <= 0.02).sum()


def test_fit_summary_counts_are_consistent(sphere_result):
    _, result = sphere_result
    fit = fabric_fit(result.distortion, FABRICS["silnylon"])
    s = fit.summary()
    assert s["triangles_ok"] + (fit.needs_relief | fit.needs_dart).sum() == len(
        result.distortion.sigma1
    )


def test_relaxation_reduces_weighted_energy(sphere_result):
    mesh, result = sphere_result
    fabric = FABRICS["ultrastretch"]
    before = relaxation_energy(mesh.vertices, mesh.faces, result.uv, fabric)
    relaxed = relax_for_fabric(mesh.vertices, mesh.faces, result.uv, fabric)
    after = relaxation_energy(mesh.vertices, mesh.faces, relaxed, fabric)
    assert after < before * 0.8, "relaxation should substantially reduce edge strain"


def test_relaxation_keeps_orientation(sphere_result):
    from flatpack.distortion import distortion_report

    mesh, result = sphere_result
    relaxed = relax_for_fabric(
        mesh.vertices, mesh.faces, result.uv, FABRICS["ultrastretch"]
    )
    report = distortion_report(mesh.vertices, mesh.faces, relaxed)
    assert not report.flipped.any()
