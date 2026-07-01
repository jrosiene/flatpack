"""Fabric model: how much strain a material can absorb, and in which direction.

A woven fabric like silnylon barely stretches in any direction; a 4-way
knit like UltraStretch can absorb a lot of strain, usually more along one
axis than the other. We model a fabric by its usable strain (fraction of
length) along its stretch axis and across it; capability at intermediate
angles is interpolated on an ellipse.

Two things are built on top of that model:

- fabric_fit: given a flattening's distortion report, decide per triangle
  whether the fabric can absorb the strain or whether the panel needs a
  dart (excess material) or a relief cut / stretch zone (missing material).
- relax_for_fabric: nudge the uv layout so edges the fabric *cannot*
  stretch match their true 3D length closely, letting the error accumulate
  in directions the fabric can absorb. This trades a little conformality
  (angle accuracy) for length accuracy where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flatpack.distortion import DistortionReport
from flatpack.meshutil import unique_edges

# Strain the seams can ease in without visible puckering, even in a
# non-stretch fabric. Used as the tolerance for both tension and
# compression before we flag a triangle.
EASE_TOLERANCE = 0.02


@dataclass(frozen=True)
class Fabric:
    """A fabric's usable stretch, as fractional strain (0.15 = 15%)."""

    name: str
    stretch_along: float  # usable strain along the stretch axis
    stretch_cross: float  # usable strain perpendicular to it

    def capability(self, angle_from_axis_rad: np.ndarray) -> np.ndarray:
        """Usable strain in a direction at the given angle from the stretch axis."""
        c = np.cos(angle_from_axis_rad)
        s = np.sin(angle_from_axis_rad)
        return np.sqrt((self.stretch_along * c) ** 2 + (self.stretch_cross * s) ** 2)


FABRICS = {
    "rigid": Fabric("rigid", 0.0, 0.0),
    "silnylon": Fabric("silnylon", 0.02, 0.02),
    "xpac": Fabric("xpac", 0.01, 0.01),
    "ultrastretch": Fabric("ultrastretch", 0.35, 0.20),
    "powermesh": Fabric("powermesh", 0.50, 0.50),
}


@dataclass
class FabricFit:
    """Where the fabric can and cannot absorb the flattening distortion."""

    fabric: Fabric
    needs_relief: np.ndarray  # bool per triangle: too much stretch required
    needs_dart: np.ndarray  # bool per triangle: too much excess material

    @property
    def ok(self) -> np.ndarray:
        return ~(self.needs_relief | self.needs_dart)

    def summary(self) -> dict:
        t = len(self.needs_relief)
        return {
            "fabric": self.fabric.name,
            "triangles_ok": int(self.ok.sum()),
            "triangles_needing_relief": int(self.needs_relief.sum()),
            "triangles_needing_dart": int(self.needs_dart.sum()),
            "fraction_ok": float(self.ok.sum() / t) if t else 1.0,
        }


def fabric_fit(
    report: DistortionReport,
    fabric: Fabric,
    stretch_axis_deg: float = 90.0,
) -> FabricFit:
    """Check the distortion of a flattened panel against a fabric.

    stretch_axis_deg is the direction of the fabric's stretch axis in uv
    coordinates, measured counterclockwise from +u. Panels are laid out
    grain-vertical by the exporter, so the default of 90 means "stretch
    along the grain".

    sigma is (uv length) / (3D length) along each principal direction:

    - sigma > 1: the cut piece is bigger than the surface. Excess material
      cannot be stretched away regardless of fabric; beyond normal sewing
      ease it must be darted or gathered out -> needs_dart.
    - sigma < 1: the cut piece is smaller than the surface, so the fabric
      must stretch by 1 - sigma in that direction to reach. If that exceeds
      the fabric's capability plus ease -> needs_relief (relief cut, gusset,
      or a different seam layout).
    """
    axis = np.radians(stretch_axis_deg)
    axis_vec = np.array([np.cos(axis), np.sin(axis)])

    # Angle between each triangle's max-stretch direction and the stretch
    # axis; sigma2 acts perpendicular to sigma1.
    cos_between = np.clip(np.abs(report.stretch_dir_uv @ axis_vec), 0.0, 1.0)
    angle_sigma1 = np.arccos(cos_between)
    angle_sigma2 = np.pi / 2 - angle_sigma1

    cap1 = fabric.capability(angle_sigma1)
    cap2 = fabric.capability(angle_sigma2)

    needs_dart = (report.sigma1 - 1.0 > EASE_TOLERANCE) | (
        report.sigma2 - 1.0 > EASE_TOLERANCE
    )
    needs_relief = (1.0 - report.sigma1 > EASE_TOLERANCE + cap1) | (
        1.0 - report.sigma2 > EASE_TOLERANCE + cap2
    )

    return FabricFit(fabric=fabric, needs_relief=needs_relief, needs_dart=needs_dart)


def relax_for_fabric(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    fabric: Fabric,
    stretch_axis_deg: float = 90.0,
    iterations: int = 100,
    step: float = 0.5,
) -> np.ndarray:
    """Anisotropic edge-length relaxation of a flattened layout.

    Every mesh edge wants its uv length to match its 3D length, weighted by
    how *stiff* the fabric is in that edge's direction: edges along a
    non-stretch direction are enforced strongly, edges along the stretch
    axis are allowed to stay wrong (the fabric will absorb it). This is a
    damped Jacobi iteration of the classic spring relaxation; returns new uv.

    Use relaxation_energy() to verify improvement.
    """
    uv = np.asarray(uv, dtype=float).copy()
    edges = unique_edges(np.asarray(faces, dtype=np.int64))
    rest = np.linalg.norm(
        np.asarray(vertices, float)[edges[:, 1]] - np.asarray(vertices, float)[edges[:, 0]],
        axis=1,
    )

    for _ in range(iterations):
        weights = _edge_stiffness(uv, edges, fabric, stretch_axis_deg)
        d = uv[edges[:, 1]] - uv[edges[:, 0]]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-12)
        # Move each endpoint half the error, scaled by stiffness.
        corr = ((length - rest) / length)[:, None] * d * 0.5 * weights[:, None]

        delta = np.zeros_like(uv)
        count = np.zeros(len(uv))
        np.add.at(delta, edges[:, 0], corr)
        np.add.at(delta, edges[:, 1], -corr)
        np.add.at(count, edges[:, 0], weights)
        np.add.at(count, edges[:, 1], weights)
        count = np.maximum(count, 1e-12)
        uv += step * delta / count[:, None]

    return uv


def relaxation_energy(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    fabric: Fabric,
    stretch_axis_deg: float = 90.0,
) -> float:
    """Stiffness-weighted squared edge-strain: the quantity relax_for_fabric reduces."""
    edges = unique_edges(np.asarray(faces, dtype=np.int64))
    rest = np.linalg.norm(
        np.asarray(vertices, float)[edges[:, 1]] - np.asarray(vertices, float)[edges[:, 0]],
        axis=1,
    )
    d = np.asarray(uv, float)[edges[:, 1]] - np.asarray(uv, float)[edges[:, 0]]
    strain = (np.linalg.norm(d, axis=1) - rest) / rest
    weights = _edge_stiffness(uv, edges, fabric, stretch_axis_deg)
    return float(np.sum(weights * strain**2))


def _edge_stiffness(
    uv: np.ndarray, edges: np.ndarray, fabric: Fabric, stretch_axis_deg: float
) -> np.ndarray:
    """Stiffness in (0, 1]: 1 where the fabric cannot stretch, lower where it can."""
    axis = np.radians(stretch_axis_deg)
    axis_vec = np.array([np.cos(axis), np.sin(axis)])
    d = np.asarray(uv, float)[edges[:, 1]] - np.asarray(uv, float)[edges[:, 0]]
    length = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
    cos_between = np.clip(np.abs((d / length[:, None]) @ axis_vec), 0.0, 1.0)
    capability = fabric.capability(np.arccos(cos_between))
    stiffness = EASE_TOLERANCE / (EASE_TOLERANCE + capability)
    return stiffness
