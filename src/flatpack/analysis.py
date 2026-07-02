"""Flag panels that cannot lie flat in their chosen fabric.

At split time — before drawing the whole pattern — it helps to know which
panels warp badly, so you can add a dart or another seam where the fabric
can't absorb the distortion. This module flattens each panel and grades
it against the fabric's stretch tolerance, returning a compact per-panel
verdict with the worst spot located in 3D (so the GUI can point at it).

The flatten-and-fit step here is the *same* code the exporter uses
(flatten_and_fit), so the live warp flags match the generated pattern's
report exactly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import trimesh

from flatpack.distortion import distortion_report
from flatpack.fabric import FABRICS, Fabric, FabricFit, fabric_fit, relax_for_fabric
from flatpack.flatten import FlattenResult, flatten
from flatpack.seams import Panel, SeamSpec, split_mesh

# Area fraction (weighted by 3D area) the fabric cannot accommodate, above
# which a panel is flagged. Below MARGINAL it flattens cleanly.
MARGINAL_FRACTION = 0.02
HIGH_FRACTION = 0.10


@dataclass
class PanelAnalysis:
    """Fabric-aware flattenability verdict for one panel."""

    name: str
    fabric: str
    severity: str  # "ok" | "marginal" | "high" | "error"
    bad_area_fraction: float  # fraction of area outside fabric tolerance
    dart_area_fraction: float  # excess material (needs a dart/gather)
    relief_area_fraction: float  # missing material (needs relief/stretch)
    max_stretch_pct: float
    max_compress_pct: float
    max_angle_error_deg: float
    worst_point_3d: list[float] | None  # mesh-space centroid of worst triangle
    advice: str

    def as_dict(self) -> dict:
        return asdict(self)


def flatten_and_fit(
    panel: Panel, relax: bool = True
) -> tuple[FlattenResult, Fabric, FabricFit]:
    """Flatten a panel and check it against its fabric.

    Shared by the exporter and the warp analysis so both agree. Applies
    the same fabric-aware relaxation the pattern uses for stretch fabrics.
    The stretch axis is measured from the grain (panels lay out
    grain-vertical, hence the 90 degree offset).
    """
    flat = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
    fabric = FABRICS[panel.spec.fabric]
    axis = 90.0 + panel.spec.stretch_axis_deg

    if relax and (fabric.stretch_along > 0 or fabric.stretch_cross > 0):
        uv = relax_for_fabric(
            panel.vertices, panel.faces, flat.uv, fabric, stretch_axis_deg=axis
        )
        flat = FlattenResult(
            uv=uv,
            distortion=distortion_report(panel.vertices, panel.faces, uv),
            pins=flat.pins,
        )

    fit = fabric_fit(flat.distortion, fabric, stretch_axis_deg=axis)
    return flat, fabric, fit


def analyze_panel(panel: Panel, relax: bool = True) -> PanelAnalysis:
    """Grade one panel's flattenability against its fabric."""
    try:
        flat, _, fit = flatten_and_fit(panel, relax=relax)
    except ValueError as exc:
        # A panel that will not flatten at all (still wraps like a tube,
        # or an annulus) is the extreme warp case: report why.
        return PanelAnalysis(
            name=panel.name,
            fabric=panel.spec.fabric,
            severity="error",
            bad_area_fraction=1.0,
            dart_area_fraction=0.0,
            relief_area_fraction=0.0,
            max_stretch_pct=float("nan"),
            max_compress_pct=float("nan"),
            max_angle_error_deg=float("nan"),
            worst_point_3d=None,
            advice=str(exc),
        )

    report = flat.distortion
    weights = report.area_3d / report.area_3d.sum()
    dart_fraction = float(weights[fit.needs_dart].sum())
    relief_fraction = float(weights[fit.needs_relief].sum())
    bad_fraction = float(weights[fit.needs_dart | fit.needs_relief].sum())

    if bad_fraction >= HIGH_FRACTION:
        severity = "high"
    elif bad_fraction >= MARGINAL_FRACTION:
        severity = "marginal"
    else:
        severity = "ok"

    worst = report.worst_triangle_index()
    worst_point = panel.vertices[panel.faces[worst]].mean(axis=0)

    return PanelAnalysis(
        name=panel.name,
        fabric=panel.spec.fabric,
        severity=severity,
        bad_area_fraction=bad_fraction,
        dart_area_fraction=dart_fraction,
        relief_area_fraction=relief_fraction,
        max_stretch_pct=float((report.sigma1 - 1.0).max() * 100.0),
        max_compress_pct=float((1.0 - report.sigma2).max() * 100.0),
        max_angle_error_deg=float(report.max_angle_error_deg.max()),
        worst_point_3d=[float(x) for x in worst_point],
        advice=_advice(severity, dart_fraction, relief_fraction),
    )


def analyze(mesh: trimesh.Trimesh, spec: SeamSpec, relax: bool = True) -> list[PanelAnalysis]:
    """Split the mesh and grade every panel's flattenability."""
    return [analyze_panel(panel, relax=relax) for panel in split_mesh(mesh, spec)]


def _advice(severity: str, dart_fraction: float, relief_fraction: float) -> str:
    if severity == "ok":
        return "flattens within fabric tolerance"
    if dart_fraction >= relief_fraction:
        primary = "add a dart (or gather) at the marked spot to take up excess material"
    else:
        primary = (
            "add a relief cut/seam at the marked spot, or choose a stretchier fabric"
        )
    if severity == "high":
        return primary + "; consider splitting this panel with another seam"
    return primary
