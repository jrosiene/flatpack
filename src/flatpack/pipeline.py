"""End-to-end pipeline: mesh + seam spec -> flattened, annotated pattern files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import trimesh

from flatpack.distortion import distortion_report
from flatpack.export import PanelLayout, layout_panel, pack_layouts, write_dxf, write_svg
from flatpack.fabric import FABRICS, FabricFit, fabric_fit, relax_for_fabric
from flatpack.flatten import FlattenResult, flatten
from flatpack.seams import Panel, SeamSpec, load_seam_spec, split_mesh
from flatpack.tiling import write_tiled_svgs


@dataclass
class PanelResult:
    panel: Panel
    flat: FlattenResult
    fit: FabricFit
    layout: PanelLayout


def process(
    mesh: trimesh.Trimesh,
    spec: SeamSpec,
    outdir: str | Path,
    page: str = "letter",
    relax: bool = True,
) -> list[PanelResult]:
    """Cut, flatten, check fabric fit, and write SVG/DXF plus tiled pages.

    Writes into outdir:
      pattern.svg      one sheet with all panels, true scale (mm)
      pattern.dxf      the same in DXF
      page_A1.svg ...  tiled pages for home printing
      report.json      distortion and fabric-fit summary per panel
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    panels = split_mesh(mesh, spec)
    results = []
    for panel in panels:
        flat = flatten(trimesh.Trimesh(panel.vertices, panel.faces, process=False))
        fabric = FABRICS[panel.spec.fabric]

        uv = flat.uv
        if relax and (fabric.stretch_along > 0 or fabric.stretch_cross > 0):
            # Panels are laid out grain-vertical; the fabric's stretch axis
            # sits at stretch_axis_deg from the grain.
            uv = relax_for_fabric(
                panel.vertices,
                panel.faces,
                uv,
                fabric,
                stretch_axis_deg=90.0 + panel.spec.stretch_axis_deg,
            )
            flat = FlattenResult(
                uv=uv,
                distortion=distortion_report(panel.vertices, panel.faces, uv),
                pins=flat.pins,
            )

        fit = fabric_fit(
            flat.distortion, fabric, stretch_axis_deg=90.0 + panel.spec.stretch_axis_deg
        )
        layout = layout_panel(panel, uv, seam_allowance=spec.seam_allowance)
        results.append(PanelResult(panel=panel, flat=flat, fit=fit, layout=layout))

    layouts = [r.layout for r in results]
    pack_layouts(layouts)
    edge_units = spec.edge_labels if spec.edge_labels in ("cm", "in") else None
    write_svg(layouts, str(outdir / "pattern.svg"), edge_units=edge_units)
    write_dxf(layouts, str(outdir / "pattern.dxf"), edge_units=edge_units)
    write_tiled_svgs(layouts, outdir, page=page, edge_units=edge_units)

    report = {
        r.panel.name: {
            "distortion": r.flat.distortion.summary(),
            "fabric_fit": r.fit.summary(),
            "worst_distortion_at_uv": [float(x) for x in r.flat.distortion.worst_triangle_uv()],
        }
        for r in results
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2))
    return results


def process_files(
    mesh_path: str | Path,
    seam_path: str | Path,
    outdir: str | Path,
    page: str = "letter",
    relax: bool = True,
) -> list[PanelResult]:
    # process=False keeps the file's vertex order intact: seam files refer
    # to vertices by index, so the mesh must not be re-welded on load.
    # Prefer OBJ/PLY over STL (STL does not store shared vertices at all).
    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    spec = load_seam_spec(seam_path)
    return process(mesh, spec, outdir, page=page, relax=relax)
